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
selected_info_ID = 'N1-19-9'
#selected_info_ID = None
selected_info = None

for info in infos:
    if info['ID'] == selected_info_ID:
        selected_info = info

# define steps
# 0: do not execute; 1: GUI; 2: silent processing; 3: silent processing and force reload
processing = dict()
processing['start'] = 0
processing['process'] = 0
processing['merge'] = 0
processing['nuclei'] = 0
processing['select'] = 0
processing['train'] = 0
processing['rev_merge'] = 0
processing['playground'] = 1

# start
if processing['start'] == 1:
    exp_id = None
    if selected_info_ID is not None:
        exp_id = selected_info_ID

    test_window = NucleoSegment(exp_id=exp_id)
    test_window.show()
    test_window.raise_()
    test_window.activateWindow()
    sys.exit(app.exec_())

# process
if processing['process'] == 1:
    test_window = NucleoProcess(selected_info)
    test_window.show()
    test_window.raise_()
    test_window.activateWindow()
    sys.exit(app.exec_())
elif processing['process'] == 2:
    print('=== Process ===')
    seg = Segmentation(selected_info)
    seg.segment(process=True, merge=False, filter=False)
    seg.get_label_props()
    seg.save(force_labels_props_raw=True)
    del(seg)

# merge
if processing['merge'] == 1:
    test_window = MergeCriteria(selected_info)
    test_window.show()
    test_window.raise_()
    test_window.activateWindow()
    sys.exit(app.exec_())
elif processing['merge'] >= 2:
    force_reload = False
    if processing['merge'] == 3:
        force_reload = True

    print('=== Merge ===')
    seg = Segmentation(selected_info)
    seg.load(force_props_load=force_reload)
    seg.segment(process=False, merge=True, filter=False)
    seg.save(force_nuclei_raw=True)
    del(seg)

# nuclei
if processing['nuclei'] == 1:
    test_window = NucleiCriteria(selected_info)
    test_window.show()
    test_window.raise_()
    test_window.activateWindow()
    sys.exit(app.exec_())
elif processing['nuclei'] >= 2:
    force_reload = False
    if processing['nuclei'] == 3:
        force_reload = True

    print('=== Filter ===')
    seg = Segmentation(selected_info)
    seg.load(force_nuclei_load=force_reload)
    seg.get_nuclei_criteria()
    removed_nuclei = seg.nuclei.filter_nuclei()
    seg.save(force_nuclei_stack_rebuild=True)

    print('save correction')
    # add nuclei to correction
    corr = Correction(seg)
    corr.add_nuclei_to_correction_filtered(removed_nuclei, add_to_stack=False)
    corr.update_correction_stacks()
    corr.save_corrections()

    del(seg)
    del(corr)

# select
if processing['select'] == 1:
    print('=== Select ===')
    test_window = NucleoSelect(selected_info)
    test_window.show()
    test_window.raise_()
    test_window.activateWindow()
    sys.exit(app.exec_())

# train
if processing['train'] == 2:
    print('=== Train ===')
    seg = Segmentation(selected_info)
    seg.load()

    # train
    clf = Classifier(seg)
    clf.train_with_exts()

    # save
    clf.save_classifier()
    seg.save()

    del(seg)
    del(clf)

# merge revisions
if processing['rev_merge'] == 2:
    print('=== Merge revision ===')
    # get revs from ID
    revs_to_merge = ImageHandler.get_revs_by_expnum(
        ImageHandler.extract_expnum_from_ID(selected_info['ID'])
    )

    # go through and merge revs with parent
    seg = Segmentation(selected_info)
    seg.load()

    # merge
    for rev_info in revs_to_merge:
        print(rev_info['ID'])
        seg.merge_parent_with_rev(rev_info)

    # save merge
    seg.save_merge_segmentation()

    seg.save()
    del(seg)

# playground
if processing['playground'] == 1:
    seg = Segmentation(selected_info)
    seg.load()

    """
    print('TEST LOCAL DENSITY')
    density_map = np.zeros_like(seg.stacks.lamin[0])
    volume_map = np.zeros_like(seg.stacks.lamin[0])
    depth_map = np.zeros_like(seg.stacks.lamin[0])
    volume_depth_map = np.zeros_like(seg.stacks.lamin[0])
    apical_dist_map = np.zeros_like(seg.stacks.lamin[0])
    dim = 10

    # tile image
    for y in range(int(density_map.shape[0]/dim)):
        for x in range(int(density_map.shape[1]/dim)):
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
            depths = list()
            volume_depths = list()
            apical_dists = list()

            print('TEST DENSITY', pos_range)

            # get parameter for nuclei
            for nID in nIDs_in_square:
                volumes.append(seg.nuclei.get_nucleus_volume(nID))
                depths.append(seg.nuclei.get_nucleus_depth(nID))
                volume_depths.append(seg.nuclei.get_nucleus_volume_depth_ratio(nID))

                apical_dist = seg.nuclei.get_nucleus_apical_distance(nID)

                if apical_dist is not None and math.isnan(apical_dist) is False:
                    apical_dists.append(apical_dist)

            # get averages and set maps
            density_map[min_y:max_y, min_x:max_x] = len(nIDs_in_square)

            if len(volumes) > 0:
                volume_map[min_y:max_y, min_x:max_x] = sum(volumes)/len(volumes)

            if len(depths) > 0:
                depth_map[min_y:max_y, min_x:max_x] = sum(depths)/len(depths)

            if len(volume_depths) > 0:
                volume_depth_map[min_y:max_y, min_x:max_x] = sum(volume_depths)/len(volume_depths)

            if len(apical_dists) > 0:
                apical_dist_map[min_y:max_y, min_x:max_x] = sum(apical_dists)/len(apical_dists)

    fig = plt.figure()

    a = fig.add_subplot(2, 3, 1)
    imgplot = plt.imshow(density_map)
    a.set_title('densities')

    a = fig.add_subplot(2, 3, 2)
    imgplot = plt.imshow(volume_map)
    a.set_title('volumes')

    a = fig.add_subplot(2, 3, 3)
    imgplot = plt.imshow(depth_map)
    a.set_title('depths')

    a = fig.add_subplot(2, 3, 4)
    imgplot = plt.imshow(volume_depth_map)
    a.set_title('volume_depths')

    a = fig.add_subplot(2, 3, 5)
    imgplot = plt.imshow(apical_dist_map)
    a.set_title('apical_dists')
    """

    print('TEST RECALC PARAMS')
    seg.nuclei.calc_nuclei_params(only_accepted=True)
    seg.save()

    #print('TEST RECALC NUCLEI')
    #corr = Correction(seg)
    #corr.apply_corrections()
