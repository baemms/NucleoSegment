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
from processing.segmentation import Segmentation

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
#selected_info_IDs = ['N1-19-9']

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

    print('LOAD', info['ID'])

    # go through nuclei and add to stack
    nIDs = seg.nuclei.get_nIDs(only_accepted=True)

    stack_apical_dist = np.zeros_like(seg.stacks.lamin)
    stack_pseudo_layer = np.zeros_like(seg.stacks.lamin)

    # get average nuclei height
    all_depths = seg.nuclei.data_frames['data_params'].get_vals_from_col('depth', only_accepted=True)

    # get average
    avg_depth = np.mean(all_depths)

    print('MEAN DEPTH', avg_depth)

    for nID in nIDs:
        # get layer
        apical_dist = seg.nuclei.get_nucleus_apical_distance(nID)

        if not math.isnan(apical_dist):
            stack_apical_dist = seg.nuclei.add_nucleus_to_stack(nID, stack_apical_dist, nucleus_value=int(apical_dist))
            stack_pseudo_layer = seg.nuclei.add_nucleus_to_stack(nID, stack_pseudo_layer,
                                                                 nucleus_value=int(apical_dist/avg_depth))

    # save stack
    img_path_apical_dist = seg.get_results_dir().tmp + 'apical_dist.tif'
    img_path_pseudo_layer = seg.get_results_dir().tmp + 'pseudo_layer.tif'

    print('SAVE')

    ImageHandler.save_stack_as_tiff(stack_apical_dist, img_path_apical_dist)
    ImageHandler.save_stack_as_tiff(stack_pseudo_layer, img_path_pseudo_layer)
