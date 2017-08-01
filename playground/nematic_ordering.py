"""
Main module to start the segmentation pipeline
"""

import matplotlib
import sys
import numpy as np
import math
import csv
import re
import heapq  # to determine nsmallest values
import os

from collections import OrderedDict

# set Qt4 for matplot
matplotlib.use('Qt4Agg')

# Qt libraries
from PyQt4 import QtGui

# matplot
from skimage import data, io
from matplotlib import pyplot as plt

import numpy as np

# import classes
from storage.image import ImageHandler
from processing.segmentation import Segmentation
from processing.correction import Correction
from processing.classifier import Classifier
from processing.image import ImageProcessing

from frontend.gui.nuc_segment import NucleoSegment
from frontend.gui.nuc_process import NucleoProcess
from frontend.gui.merge_criteria import MergeCriteria
from frontend.gui.nuc_criteria import NucleiCriteria
from frontend.gui.nuc_select import NucleoSelect

import storage.config as cfg

# show the editor to choose images
app = QtGui.QApplication(sys.argv)

# update fonts
import matplotlib.pylab as plt
params = {'axes.titlesize': cfg.fontsdict_plot['fontsize'],
          'xtick.labelsize': cfg.fontsdict_plot['fontsize'],
          'ytick.labelsize': cfg.fontsdict_plot['fontsize']}

plt.rcParams.update(params)

# select a specific info
selected_info_IDs = ['N1-19-9']

# open window to select nuclei criteria
loaded_infos = ImageHandler.load_image_infos()
infos = list()

for info in loaded_infos:
    info_ID = info['ID']
    if info_ID in selected_info_IDs:
        infos.append(info)

for info in infos:
    seg = Segmentation(info)
    seg.load()

    # go through nuclei and add them to a stack with
    # nuclei_in_direction as value
    nIDs = seg.nuclei.get_nIDs(only_accepted=True)

    layer_stack = np.zeros_like(seg.stacks.nuclei)

    for nID in nIDs:
        # get nuclei in direction
        apical = seg.nuclei.get_nucleus_apical_distance(nID)

        if np.isnan(apical):
            apical = 0

        # add to stack
        seg.nuclei.add_nucleus_to_stack(nID, layer_stack, nucleus_value=apical)

    # save stack as tif
    ImageHandler.save_stack_as_tiff(layer_stack, seg.get_results_dir().tmp + 'nuclei_layers.tif')
