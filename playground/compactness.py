import matplotlib
import sys
import numpy as np
import math
import csv
import re
import os
import getopt

# set Qt4 for matplot
matplotlib.use('Qt4Agg')

# Qt libraries
from PyQt4 import QtGui

# import classes
from storage.image import ImageHandler
from processing.image import ImageProcessing
from processing.segmentation import Segmentation
from processing.filter import Dilation

import storage.config as cfg

# show the editor to choose images
app = QtGui.QApplication(sys.argv)

# update fonts
import matplotlib.pylab as plt
params = {'axes.titlesize': cfg.fontsdict_plot['fontsize'],
          'xtick.labelsize': cfg.fontsdict_plot['fontsize'],
          'ytick.labelsize': cfg.fontsdict_plot['fontsize']}

plt.rcParams.update(params)
selected_info_IDs = ['N1-19-9', 'N1-19-22', 'N1-19-23', 'N1-19-24']
#selected_info_IDs = ['N1-19-9', 'N1-19-23']

# open window to select nuclei criteria
loaded_infos = ImageHandler.load_image_infos()
infos = list()

# processing steps
processing_steps = [['DIL', 'y', 2],
                    ['CONV_BIT', 16]]

for info in loaded_infos:
    info_ID = info['ID']
    if info_ID in selected_info_IDs:
        infos.append(info)

for info in infos:
    seg = Segmentation(info)
    seg.load()

    img_path = seg.get_results_dir().tmp + 'contact_surface.tif'

    print('LOAD', info['ID'])

    # go through nuclei and add to stack
    nIDs = seg.nuclei.get_nIDs(only_accepted=True)

    stack_contact_surface = np.zeros_like(seg.stacks.lamin)

    num_nIDs = len(nIDs)

    # go through nIDs
    for i, nID in enumerate(nIDs):
        print('PROCESS %i/%i' % (i, num_nIDs))

        stack_tmp_self = np.zeros_like(seg.stacks.lamin)
        stack_tmp_others = np.zeros_like(seg.stacks.lamin)
        stack_tmp_contact = np.zeros_like(seg.stacks.lamin)

        # get nucleus bounding-box
        nuclei_bbox = seg.nuclei.get_expanded_bbox_for_nucleus(nID, only_horizontal=False)

        # get all neighbouring nuclei
        neighbour_nIDs = seg.nuclei.get_nID_by_pos_range(nuclei_bbox)

        # delete yourself
        if nID in neighbour_nIDs:
            neighbour_nIDs.remove(nID)

        # add nuclei to stack
        stack_tmp_self = seg.nuclei.add_nucleus_to_stack(nID, stack_tmp_self, nucleus_value=1)

        for neighbour_nID in neighbour_nIDs:
            seg.nuclei.set_nucleus_colour(neighbour_nID, neighbour_nID)

        stack_tmp_others = seg.nuclei.add_nuclei_to_stack(stack_tmp_others, nIDs=neighbour_nIDs)

        # dilate and combine
        stack_tmp_contact = np.logical_and(
            ImageProcessing.apply_filters(processing_steps, stack_tmp_self),
            ImageProcessing.apply_filters(processing_steps, stack_tmp_others))
        neighbours_stack = (stack_tmp_others * stack_tmp_contact)
        neighbours = set(neighbours_stack.ravel().tolist())

        if len(neighbours) > 0:
            neighbours.remove(0)

        #print('NEIGHBOURS', neighbours)

        # get contact surface
        contact_surface = np.sum(stack_tmp_contact)
        nucleus_surface = seg.nuclei.get_nucleus_surface(nID)

        compactness = contact_surface / nucleus_surface

        #print('CONTACT SURFACE', contact_surface, nucleus_surface, compactness)

        seg.nuclei.set_nucleus_contact_surface(contact_surface, nID)
        seg.nuclei.set_nucleus_neighbours(len(neighbours), nID)
        seg.nuclei.set_nucleus_compactness(compactness, nID)

        # add to stack
        stack_contact_surface += stack_tmp_contact

        #ImageHandler.save_stack_as_tiff(stack_tmp_self, seg.get_results_dir().tmp + '%i_self.tif' % nID)
        #ImageHandler.save_stack_as_tiff(stack_tmp_others, seg.get_results_dir().tmp + '%i_others.tif' % nID)

    # save nuclei
    seg.nuclei.save(seg.get_results_dir().nuclei_params_corr)

    # make all one
    stack_contact_surface[stack_contact_surface > 0] = 1

    # save stack

    print('SAVE')

    ImageHandler.save_stack_as_tiff(stack_contact_surface, img_path)
