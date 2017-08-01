"""
Extract pouch volume
"""

import skimage.io as io
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import numpy as np

# import nucleo_segment classes
from storage.image import ImageHandler
from processing.image import ImageProcessing
from frontend.figures.plot import Plot

# define directories
work_dir = '/Users/schiend/Desktop/Drive/Experiments/Carles/'

# load image - Z, C, Y, X
pouch_img = io.imread(
    work_dir + 'vgQE-dsRed_mLamin-A488_wL3-L2_late_sync_25C.lif - Series066.tif'
)

# reorder axes - C, Z, Y, X
pouch_stack = pouch_img.swapaxes(0, 1)[0, 20:21:1, :, :]

processing_steps = [
    ['EQU'],
    ['THR', 'OTSU', 100, 'no3D'],       # checked
    ['MED', 1],
    ['ERO', 'bin', 5],                   # checked
    ['FILL'],
    ['OPN', 'bin', 2],
    ['CONV_BIT', 16, '3D'],
    ['LABEL', 1, 'no3D']
]

check_params = True
do_post_process = False

results_titles = list()
results_titles.append('Original')
pouch_volumes = list()

if check_params is True:
    for i in range(10, 21, 1):
        processing_steps[5][2] = i
        pouch_volumes.append(ImageProcessing.apply_filters(processing_steps, pouch_stack, verbose=True))
        results_titles.append('OPN %i' % i)
else:
    pouch_volumes.append(ImageProcessing.apply_filters(processing_steps, pouch_stack, verbose=True))
    results_titles.append('Processed')

if do_post_process is True:
    # go through labels and filter small ones
    pouch_volume_filtered = np.zeros_like(pouch_volumes[0])
    union_pouch_volume = np.zeros_like(pouch_volumes[0])
    dilate_step = [
        ['DIL', 'bin', 50]
    ]

    for z in range(0, pouch_volumes[0].shape[0]):
        print('get props for z: %i' % z)

        z_props = regionprops(pouch_volumes[0][z])

        # a list of dilated images
        dilated_imgs = list()

        # go through and filter by size
        for props in z_props:
            if props['area'] > 1000:
                # add to image
                dilated_imgs.append(np.zeros_like(union_pouch_volume[z]))

                for i, coords in enumerate(props['coords']):
                    pouch_volume_filtered[z][int(coords[0]), int(coords[1])] = 1
                    dilated_imgs[-1][int(coords[0]), int(coords[1])] = 1

                # dilate
                dilated_imgs[-1] = ImageProcessing.apply_filters(dilate_step, dilated_imgs[-1], verbose=True)

        print('calc union for %i dilated images' % len(dilated_imgs))

        # go through all dilated images and add union
        for o, dilated_outer in enumerate(dilated_imgs):
            for i, dilated_inner in enumerate(dilated_imgs):
                if o != i:
                    union_pouch_volume[z] += np.logical_and(dilated_outer, dilated_inner)

    # add filtered and union to results
    pouch_volumes.append(pouch_volume_filtered)
    results_titles.append('Filtered')

    pouch_volumes.append(union_pouch_volume)
    results_titles.append('Union')

    # add union to filtered
    pouch_volume_final = pouch_volume_filtered + union_pouch_volume
    pouch_volume_final[pouch_volume_final > 1] = 1

    final_steps = [
        ['CLS', 'bin', 20],
        ['FILL']
    ]

    #for i in range(20, 41, 5):
    #        final_steps[0][2] = i
    #        pouch_volumes.append(ImageProcessing.apply_filters(final_steps, pouch_volume_final, verbose=True))
    #        results_titles.append('CLS %i' % i)

    pouch_volume_final = ImageProcessing.apply_filters(final_steps, pouch_volume_final, verbose=True)
    pouch_volumes.append(pouch_volume_final)
    results_titles.append('Final')

# show images
stack_fig = plt.figure(figsize=(20, 10))

# view the stack and the processing results
results = list()

results.append(pouch_stack)

for pouch_volume in pouch_volumes:
    results.append(pouch_volume)

Plot.show_stacks(stack_fig, results, range(0, 1, 1), img_title=results_titles)

#ImageHandler.save_stack_as_tiff(pouch_volume, work_dir + 'pouch_volume.tif')
