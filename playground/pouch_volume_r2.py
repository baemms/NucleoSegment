"""
Extract pouch volume
"""

import skimage.io as io
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import active_contour
import numpy as np

# import nucleo_segment classes
from storage.image import ImageHandler
from processing.image import ImageProcessing
from frontend.figures.plot import Plot

# define directories
work_dir = '/Users/schiend/Desktop/Drive/Experiments/Carles/'

# define files
pouch_files = [
    'Q4_late/vgQE-dsRed_mLamin-A488_wL3-L2 late sync_25C.lif - Series066.tif',
    #'Q4_early/vgQE-dsRed_mLamin-A488_eL3-L2 late sync_25C.lif - Series012.tif' # trachea in the way
    #'Q4_early/vgQE-dsRed_mLamin-A488_eL3-L2 late sync_25C.lif - Series015.tif' # tilted
    #'Q4_early/vgQE-dsRed_mLamin-A488_eL3-L2 late sync_25C.lif - Series020.tif' # ok
    #'Q4_early/vgQE-dsRed_mLamin-A488_eL3-L2 late sync_25C.lif - Series022.tif'  # best
    #'Q4_early/vgQE-dsRed_mLamin-A488_eL3-L2 late sync_25C.lif - Series027.tif' # too weak
]

pouch_thrs = [
    40,
    #?,
    #?,
    100
    #100
]

pouch_imgs = list()
pouch_stacks = list()

for file in pouch_files:
    print('--- LOAD %s' % file)

    # load image - Z, C, Y, X
    pouch_imgs.append(io.imread(
        work_dir + file
    ))

    if len(pouch_imgs[-1].shape) > 3:
        # reorder axes - C, Z, Y, X
        if pouch_imgs[-1].shape[1] < 10:
            pouch_stacks.append(pouch_imgs[-1].swapaxes(0, 1)[0, 40:41:1, :, :, 0])
        else:
            pouch_stacks.append(pouch_imgs[-1][:, :, :, 0])
    else:
        pouch_stacks.append(pouch_imgs[-1][20:61:10])

processing_steps = [
    ['EQU'],
    ['THR', 'OTSU', 100, 'no3D'],    # checked
    ['CLS', 'bin', 5],              # checked
    ['FILL'],
    ['OPN', 'bin', 2],              # checked
    ['CONV_BIT', 16, '3D'],
    ['LABEL', 1, 'no3D']
]

THR_STEP = 1

dilate_step = [
    ['DIL', 'bin', 50]
]

final_steps = [
    ['CLS', 'bin', 30],
    ['FILL']
]

area_threshold = 1000

check_params = False
active_contour = True
calc_union = False
save_as_stack = False
show_stack = True
show_range = range(0, 1, 1)

###
# Go through defined images and process
###

results = list()
results_titles = list()

for i, pouch_stack in enumerate(pouch_stacks):
    print('--- PROCESS %i' % i)

    results.append(pouch_stack)
    results_titles.append('Original')

    pouch_volumes = list()

    if check_params is True:
        for i in range(0, 51, 10):
            processing_steps[1][2] = i
            pouch_volumes.append(ImageProcessing.apply_filters(processing_steps, pouch_stack, verbose=True))
            results.append(pouch_volumes[-1])
            results_titles.append('THR %i' % i)
    else:
        # set threshold
        processing_steps[THR_STEP][2] = pouch_thrs[i]

        pouch_volumes.append(ImageProcessing.apply_filters(processing_steps, pouch_stack, verbose=True))
        results.append(pouch_volumes[-1])
        results_titles.append('Threshold')

    # NEXT: Test active contours
    # ERROR: TypeError: 'bool' object is not callable
    if active_contour is True:
        s = np.linspace(0, 2*np.pi, 400)
        x = 600 + 300*np.cos(s)
        y = 585 + 300*np.sin(s)
        init = np.array([x, y]).T

        snake_img = active_contour(pouch_imgs[-1][0], init, alpha=0.015, beta=10, gamma=0.001)
        results.append(snake_img)
        results_titles('Snake')

    if calc_union is True:
        # go through labels and filter small ones
        pouch_volume_filtered = np.zeros_like(pouch_volumes[-1])
        union_pouch_volume = np.zeros_like(pouch_volumes[-1])

        for z in range(0, pouch_volumes[-1].shape[0]):
            print('- GET PROPS for z: %i' % z)

            z_props = regionprops(pouch_volumes[-1][z])

            # a list of dilated images
            dilated_imgs = list()

            # go through and filter by size
            for props in z_props:
                if props['area'] > area_threshold:
                    # add to image
                    dilated_imgs.append(np.zeros_like(union_pouch_volume[z]))

                    for i, coords in enumerate(props['coords']):
                        pouch_volume_filtered[z][int(coords[0]), int(coords[1])] = 1
                        dilated_imgs[-1][int(coords[0]), int(coords[1])] = 1

                    # dilate
                    dilated_imgs[-1] = ImageProcessing.apply_filters(dilate_step, dilated_imgs[-1], verbose=True)

            print('- CALC UNION for %i dilated images' % len(dilated_imgs))

            # go through all dilated images and add union
            for o, dilated_outer in enumerate(dilated_imgs):
                for i, dilated_inner in enumerate(dilated_imgs):
                    if o != i:
                        union_pouch_volume[z] += np.logical_and(dilated_outer, dilated_inner)

        # add filtered and union to results
        pouch_volumes.append(pouch_volume_filtered)
        results.append(pouch_volumes[-1])
        results_titles.append('Filtered')

        pouch_volumes.append(union_pouch_volume)
        results.append(pouch_volumes[-1])
        results_titles.append('Union')

        # add union to filtered
        pouch_volume_final = pouch_volume_filtered + union_pouch_volume
        pouch_volume_final[pouch_volume_final > 1] = 1

        #for i in range(20, 41, 5):
        #        final_steps[0][2] = i
        #        pouch_volumes.append(ImageProcessing.apply_filters(final_steps, pouch_volume_final, verbose=True))
        #        results_titles.append('CLS %i' % i)

        pouch_volume_final = ImageProcessing.apply_filters(final_steps, pouch_volume_final, verbose=True)

        results.append(pouch_volume_final)
        results_titles.append('Volume #%i' % i)

    if save_as_stack is True:
        ImageHandler.save_stack_as_tiff(pouch_volume_final, work_dir + ('%s_pv.tif' % pouch_files[i]))

    if show_stack is True:
        # show images
        stack_fig = plt.figure(figsize=(20, 10))

        # view the stack and the processing results

        cmap = list()
        for i in range(0, len(results)):
            cmap.append('hot')

        Plot.show_stacks(stack_fig, results, show_range, img_title=results_titles, colour_map=cmap)
