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

    stack_volume = np.zeros_like(seg.stacks.lamin).astype(np.int8)

    all_volumes = seg.nuclei.data_frames['data_params'].get_vals_from_col('volume', only_accepted=True)

    min_val = float(max(all_volumes))
    max_val = float(min(all_volumes))

    for nID in nIDs:
        volume = seg.nuclei.get_nucleus_volume(nID)

        colour = int((volume - min_val)/(max_val - min_val) * 255)

        seg.nuclei.set_nucleus_colour(colour, nID)

    stack_volume = seg.nuclei.add_nuclei_to_stack(stack_volume, only_accepted=True)

    # save stack
    img_path = seg.get_results_dir().tmp + 'volume.tif'

    print('SAVE')

    ImageHandler.save_stack_as_tiff(stack_volume, img_path)
