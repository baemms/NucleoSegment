import matplotlib
import sys
import numpy as np
import math
import csv
import re
import os
import getopt
from decimal import Decimal
from random import choice

# set Qt4 for matplot
matplotlib.use('Qt4Agg')

# Qt libraries
from PyQt4 import QtGui

# import classes
from storage.image import ImageHandler
from processing.image import ImageProcessing
from processing.segmentation import Segmentation
from processing.filter import Dilation
from frontend.figures.plot import Plot

import storage.config as cfg

# show the editor to choose images
app = QtGui.QApplication(sys.argv)

# update fonts
import matplotlib.pylab as plt
params = {'axes.titlesize': cfg.fontsdict_plot['fontsize'],
          'xtick.labelsize': cfg.fontsdict_plot['fontsize'],
          'ytick.labelsize': cfg.fontsdict_plot['fontsize']}

plt.rcParams.update(params)
#selected_info_IDs = ['N1-19-9', 'N1-19-22', 'N1-19-23', 'N1-19-24']
selected_info_IDs = ['N1-19-23']

# open window to select nuclei criteria
loaded_infos = ImageHandler.load_image_infos()
infos = list()

# processing steps
processing_steps = [['DIL', 'y', 2],
                    ['CONV_BIT', 16]]

# calculate compactness?
calc_compactness = False

for info in loaded_infos:
    info_ID = info['ID']
    if info_ID in selected_info_IDs:
        infos.append(info)

for info in infos:
    seg = Segmentation(info)
    seg.load()

    img_path = seg.get_results_dir().tmp + 'compactness_examples.tif'

    voxel_size = float(info['voxel_size'])
    voxel_volume = voxel_size ** 3
    voxel_surface = voxel_size ** 2

    print('LOAD', info['ID'])

    # go through nuclei and add to stack
    nIDs = seg.nuclei.get_nIDs(only_accepted=True)

    example_counter = np.zeros((5 * 5)).astype(np.int8)
    stack_examples = list()
    examples_to_show = 5

    for x in range(0, len(example_counter)):
        stack_examples.append(list())

        for y in range(0, examples_to_show):
            stack_examples[-1].append(dict())

    examples_finished = False

    num_nIDs = len(nIDs)
    print('NUCLEI', num_nIDs)

    # calculate compactness
    if calc_compactness is True:
        for i, nID in enumerate(nIDs):
            surface = seg.nuclei.get_nucleus_surface(nID)
            volume = seg.nuclei.get_nucleus_volume(nID)
            contact_surface = seg.nuclei.get_nucleus_contact_surface(nID)
            neighbours = seg.nuclei.get_nucleus_neighbours(nID)

            compactness = contact_surface/surface

            closeness = 0
            if neighbours > 0:
                closeness = compactness/neighbours

            sphericity = ((math.pi ** (1/3)) * ((6 * volume * voxel_volume) ** (2/3)))/(surface * voxel_surface)

            #seg.nuclei.set_nucleus_compactness(compactness, nID)
            #seg.nuclei.set_nucleus_closeness(closeness, nID)
            seg.nuclei.set_nucleus_sphericity(sphericity, nID)

            #print('COMPACTNESS/NEIGHBOURS/CLOSENESS', compactness, neighbours, closeness)
            if (i % 100) == 0:
                print('VOLUME/SURFACE/SPHERITICITY', volume, surface, sphericity)
                print('DONE %i/%i' % (i, num_nIDs))

        # save nuclei
        seg.nuclei.save(seg.get_results_dir().nuclei_params_corr)

        print('SAVED')
    else:
        # load stacks
        stack_contact = ImageHandler.load_tiff_as_stack(seg.get_results_dir().tmp + 'contact_surface.tif')

        # get examples
        print('GET EXAMPLES')
        while examples_finished is False and len(nIDs) > 0:
            # get a random nucleus
            nID = choice(nIDs)
            nIDs.remove(nID)

            compactness = seg.nuclei.get_nucleus_compactness(nID)
            closeness = seg.nuclei.get_nucleus_closeness(nID)
            neighbours = seg.nuclei.get_nucleus_neighbours(nID)
            sphericity = seg.nuclei.get_nucleus_sphericity(nID)

            # save example
            compactness = round(Decimal(compactness), 2)
            closeness = round(Decimal(closeness), 2)

            nucleus_value = 0

            # compactness
            if compactness < 0.05:
                nucleus_value = 1
            elif 0.24 < compactness < 0.26:
                nucleus_value = 2
            elif 0.49 < compactness < 0.51:
                nucleus_value = 3
            elif 0.74 < compactness < 0.76:
                nucleus_value = 4
            elif compactness > 0.95:
                nucleus_value = 5

            if nucleus_value > 0:
                # closeness
                if neighbours < 3:
                    nucleus_value += 0
                elif 3 <= neighbours <= 4:
                    nucleus_value += (1 * 5)
                elif 5 <= neighbours <= 6:
                    nucleus_value += (2 * 5)
                elif 7 <= neighbours <= 8:
                    nucleus_value += (3 * 5)
                elif neighbours > 8:
                    nucleus_value += (4 * 5)

            if nucleus_value > 0 and example_counter[(nucleus_value - 1)] < examples_to_show:
                # create examples
                cur_stack = stack_examples[(nucleus_value - 1)][example_counter[(nucleus_value - 1)]]

                #seg.nuclei.add_nucleus_to_stack(nID, stack_examples, nucleus_value=nucleus_value)
                nucleus_centroids = seg.nuclei.get_nucleus_centroids(nID)
                nucleus_areas = seg.nuclei.get_nucleus_areas(nID)

                stack_nucleus = seg.nuclei.add_nucleus_to_stack(nID, np.zeros_like(seg.stacks.lamin), 1)

                cur_stack['lamin'] = Plot.get_nucleus_box(
                    nucleus_centroids, nucleus_areas, seg.stacks.lamin, 10)
                cur_stack['contact'] = Plot.get_nucleus_box(
                    nucleus_centroids, nucleus_areas, stack_contact, 10)
                cur_stack['nucleus'] = Plot.get_nucleus_box(
                    nucleus_centroids, nucleus_areas, stack_nucleus, 10)
                cur_stack['sphericity'] = sphericity

                example_counter[(nucleus_value - 1)] += 1

            # got all examples?
            if len(example_counter[example_counter >= examples_to_show]) == len(example_counter):
                examples_finished = True

        # save stack
        print('SAVE STACKS')

        fiji_array = ''

        for i, examples in enumerate(stack_examples):
            for j, stacks in enumerate(examples):
                if len(stacks) > 0:
                    fiji_array += '\'%i_%i_xy\',' % (i, j)

                    cur_sphericity = stacks['sphericity']

                    print('ex_%i_%i_xy %.2f' % (i, j, sphericity))

                    for key, value in stacks.items():
                        if key != 'sphericity':
                            ImageHandler.save_stack_as_tiff(value[int(value.shape[0]/2)],
                                                            seg.get_results_dir().tmp
                                                            + 'ex_%i_%i_xy_%s' % (i, j, key))

                    fiji_array += '\'%i_%i_yz\',' % (i, j)

                    for key, value in stacks.items():
                        if key != 'sphericity':
                            ImageHandler.save_stack_as_tiff(value[:, int(value.shape[1]/2), :],
                                                            seg.get_results_dir().tmp
                                                            + 'ex_%i_%i_yz_%s' % (i, j, key))

        print('DONE')
        print('FIJI ARRAY:')
        print(fiji_array)
