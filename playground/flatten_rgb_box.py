# flatten rgb box and set dimensions
flattened_nuclei = list()
max_dim_z = 0
max_dim_y = 0
max_dim_x = 0

# load nuclei data
for nucleus in seg.nuclei:
    flattened_nuclei.append(nucleus['cropped_rgb_box'].flatten())

    # set max dimensions
    if nucleus['cropped_rgb_box'].shape[0] > max_dim_z:
        max_dim_z = nucleus['cropped_rgb_box'].shape[0]
    if nucleus['cropped_rgb_box'].shape[1] > max_dim_y:
        max_dim_y = nucleus['cropped_rgb_box'].shape[1]
    if nucleus['cropped_rgb_box'].shape[2] > max_dim_x:
        max_dim_x = nucleus['cropped_rgb_box'].shape[2]

# calc dimension for array
dimension = max_dim_z * max_dim_y * max_dim_x * seg.nuclei[0]['cropped_rgb_box'].shape[3]
tpl_stack = np.zeros((max_dim_z, max_dim_y, max_dim_x, seg.nuclei[0]['cropped_rgb_box'].shape[3]))

# add data to nuclei
nucleus_data = np.zeros(shape=(len(seg.nuclei), dimension))
nucleus_target = np.zeros(shape=(len(seg.nuclei)))
for i, nucleus in enumerate(seg.nuclei):
     centred_nucleus = ImageHandler.add_stack_to_centre(nucleus['cropped_rgb_box'], tpl_stack)
     nucleus_data[i] = centred_nucleus.flatten()

     # match data to target
     nucleus_target[i] = int(corr.is_correction_nonuc(nucleus))
