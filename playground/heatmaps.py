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
selected_info_IDs = ['N1-19-9', 'N1-19-23']
selected_infos = dict()
segs = dict()

# maps
density_map = dict()
volume_map = dict()

# lists
density_list = dict()
volume_list = dict()

min_den = -1
max_den = -1
min_vol = -1
max_vol = -1

# resolution
dim = 100

for info in infos:
    if info['ID'] in selected_info_IDs:
        selected_infos[info['ID']] = info
        segs[info['ID']] = Segmentation(info)
        segs[info['ID']].load()

        # prepare maps
        density_map[info['ID']] = np.zeros_like(segs[info['ID']].stacks.lamin[0])
        volume_map[info['ID']] = np.zeros_like(segs[info['ID']].stacks.lamin[0])

        # prepare lists
        density_list[info['ID']] = list()
        volume_list[info['ID']] = list()

for info_ID, seg in segs.items():
    # tile image
    for y in range(int(density_map[info_ID].shape[0]/dim)):
        for x in range(int(density_map[info_ID].shape[1]/dim)):
            min_y = (y * dim)
            min_x = (x * dim)
            max_y = ((y * dim) + dim)
            max_x = ((x * dim) + dim)

            pos_range = np.array([
                0, min_y, min_x,
                seg.stacks.lamin.shape[0], max_y, max_x,
            ])

            # get nuclei in square
            nIDs_in_square = seg.nuclei.get_nID_by_pos_range(pos_range)

            volumes = list()

            # get parameter for nuclei
            for nID in nIDs_in_square:
                volumes.append(seg.nuclei.get_nucleus_volume(nID))

            # get averages and set maps
            if len(nIDs_in_square) > 0:
                density_list[info_ID].append(len(nIDs_in_square))

            if len(volumes) > 0:
                volume_list[info_ID].append(sum(volumes)/len(volumes))

# get min and max
for info_ID, seg in segs.items():
    if min_den < 0 or min_den > min(density_list[info_ID]):
        min_den = min(density_list[info_ID])

    if max_den < 0 or max_den < max(density_list[info_ID]):
        max_den = max(density_list[info_ID])

    if min_vol < 0 or min_vol > min(volume_list[info_ID]):
        min_vol = min(volume_list[info_ID])

    if max_vol < 0 or max_vol < max(volume_list[info_ID]):
        max_vol = max(volume_list[info_ID])

# prepare maps
for info_ID, seg in segs.items():
    counter = 0
    for y in range(int(density_map[info_ID].shape[0]/dim)):
        for x in range(int(density_map[info_ID].shape[1]/dim)):
            print('TEST LIST', density_list, counter)

            min_y = (y * dim)
            min_x = (x * dim)
            max_y = ((y * dim) + dim)
            max_x = ((x * dim) + dim)

            density_map[info_ID][min_y:max_y, min_x:max_x] = ((density_list[info_ID][counter] - min_den)/(max_den - min_den)) * 255
            volume_map[info_ID][min_y:max_y, min_x:max_x] = ((volume_list[info_ID][counter] - min_vol)/(max_vol - min_vol)) * 255

            counter += 1

print('Plot')

fig = plt.figure()

i = 0
for info_ID, seg in segs.items():
    a = fig.add_subplot(2, 4, 1 + (2 * i))
    imgplot = plt.imshow(density_map[info_ID])
    a.set_title('densities')

    a = fig.add_subplot(2, 4, 2 + (2 * i))
    imgplot = plt.imshow(volume_map[info_ID])
    a.set_title('volumes')

    i += 1

i = 0
for info_ID, seg in segs.items():
    # plot data
    a = fig.add_subplot(2, 4, 5 + (2 * i))
    plt.hist(density_list[info_ID], bins=100, color='blue')
    a.set_title('densities')

    a = fig.add_subplot(2, 4, 6 + (2 * i))
    plt.hist(volume_list[info_ID], bins=100, color='blue')
    a.set_title('volumes')

    i += 1
