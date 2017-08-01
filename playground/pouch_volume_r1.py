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
pouch_stack = pouch_img.swapaxes(0, 1)[0, 20:61:10, :, :]

processing_steps = [
    ['EQU'],
    ['THR', 'OTSU', 40, 'no3D'],    # checked
    ['CLS', 'bin', 5],              # checked
    ['FILL'],
    ['OPN', 'bin', 2],              # checked
    ['CONV_BIT', 16, '3D'],
    ['LABEL', 1, 'no3D']
]

check_params = True

results_titles = list()
results_titles.append('Original')
pouch_volumes = list()

if check_params is True:
    for i in range(0, 51, 10):
        processing_steps[1][2] = i
        pouch_volumes.append(ImageProcessing.apply_filters(processing_steps, pouch_stack, verbose=True))
        results_titles.append('THR %i' % i)
else:
    pouch_volumes.append(ImageProcessing.apply_filters(processing_steps, pouch_stack, verbose=True))
    results_titles.append('Processed')

# show images
stack_fig = plt.figure(figsize=(20, 10))

# view the stack and the processing results
results = list()

results.append(pouch_stack)

for pouch_volume in pouch_volumes:
    results.append(pouch_volume)

cmap = list()
for i in range(0, len(results)):
    cmap.append('hot')

Plot.show_stacks(stack_fig, results, range(0, 4, 1), img_title=results_titles, colour_map=cmap)

ImageHandler.save_stack_as_tiff(pouch_volumes[0], work_dir + 'pouch_volume.tif')
