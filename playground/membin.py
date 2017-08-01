"""
Main module to start the segmentation pipeline
"""

import matplotlib
import sys
import numpy as np
import math

# set Qt4 for matplot
matplotlib.use('Qt4Agg')

# Qt libraries
from PyQt4 import QtGui

# matplot
from skimage import data, io
from matplotlib import pyplot as plt

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

# open window to select nuclei criteria
infos = ImageHandler.load_image_infos()

# select a specific info
selected_info_IDs = ['N1-19-24']

for info in infos:
    if info['ID'] in selected_info_IDs:
        seg = Segmentation(info)
        seg.load()

processing_steps = [
    ['EQU'],
    ['THR', 'OTSU', 100, 'no3D'],
    ['CLS', 'bin', 2],
    ['FILL'],
    ['OPN', 'bin', 2],
    ['CONV_BIT', 16, '3D']
]

# apply processing steps
seg.stacks.membin = ImageProcessing.apply_filters(
    processing_steps, seg.stacks.membrane, verbose=cfg.general_verbose)

ImageHandler.save_stack_as_tiff(seg.stacks.membin, seg.get_results_dir().tmp + cfg.file_stack_membin)
