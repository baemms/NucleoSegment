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
selected_info_IDs = ['N1-19-9', 'N1-19-22', 'N1-19-23', 'N1-19-24']
#selected_info_IDs = ['N1-19-9']
params_to_get = ['nuclei_number', 'volume',  'between_space', 'sphericity',
                 'layer_height', 'volumes_per_layer_height', 'nuclei_per_layer_height',
                 'nuclei_in_direction', 'layer_height_per_layer', 'nuclei_per_layer', 'volumes_per_layer',
                 'membin_stack_volume', 'nuclei_stack_volume', 'density', 'nuclei_per_volume',
                 'layer_height_per_membin_stack_volume', 'avg_apical_dist', 'avg_pile', 'apical_dist_bin']
#params_to_get = ['nuclei_number', 'volume', 'avg_pile']
#layers_to_get = ['nuclei_number', 'volume', 'between_space', 'sphericity']
layers_to_get = list()
layers_to_get_count = 3

# add layers to params
for layer_param in layers_to_get:
    for layer_id in range(1, layers_to_get_count + 1):
        params_to_get.append('layer_' + layer_param + '_' + str(layer_id))

limit_nvals = 5
save_csv = True
plot_figures = True
only_centre = True

# window tiling for averaging data
tiling_dim = 50
tiling_count = 5
tiling_offset = int(tiling_dim/tiling_count)

# open window to select nuclei criteria
loaded_infos = ImageHandler.load_image_infos()
infos = list()

for info in loaded_infos:
    info_ID = info['ID']
    if info_ID in selected_info_IDs:
        infos.append(info)
        print('TEST APPEND', info)

img_maps = dict()
val_maps = dict()
tiling_count_map = dict()
tiling_maps = dict()
limits = dict()
limit_lists = dict()

for info in infos:
    seg = Segmentation(info)
    seg.load()

    # preparations for params
    if 'between_space' in params_to_get:
        # get binary stack without nuclei
        filled_minus_nuclei = ImageHandler.load_tiff_as_stack(seg.get_results_dir().tmp + 'filled_minus_nuclei.tif')

        if filled_minus_nuclei is None:
            filled_stack = np.full((seg.stacks.lamin.shape[0],
                                    seg.stacks.lamin.shape[1],
                                    seg.stacks.lamin.shape[2]), 1, dtype=np.uint8)

            filled_minus_nuclei = seg.nuclei.add_nuclei_to_stack(filled_stack, nucleus_value=0,
                                                                 only_accepted=True)
            ImageHandler.save_stack_as_tiff(filled_minus_nuclei,
                                            seg.get_results_dir().tmp + 'filled_minus_nuclei.tif')

    count_lists = dict()

    # init maps, lists and tiles
    for param in params_to_get:
        if (param in img_maps) is False:
            img_maps[param] = dict()
            val_maps[param] = dict()
            tiling_count_map[param] = dict()
            tiling_maps[param] = dict()
            limits[param] = [-1, -1]
            limit_lists[param] = [list(), list()]

        img_maps[param][info['ID']] = np.zeros_like(seg.stacks.lamin[0])
        val_maps[param][info['ID']] = np.zeros_like(seg.stacks.lamin[0])
        tiling_count_map[param][info['ID']] = np.zeros_like(seg.stacks.lamin[0])

        # init tiling matrices
        tiling_maps[param][info['ID']] = list()

        for tile in range(tiling_count):
            tiling_maps[param][info['ID']].append(np.zeros_like(seg.stacks.lamin[0]))

    # go through the image and get parameters from nuclei
    for tile in range(tiling_count):
        print('new tile', info['ID'], tile)

        for y in range(int(img_maps[params_to_get[0]][info['ID']].shape[0]/tiling_dim)):
            for x in range(int(img_maps[params_to_get[0]][info['ID']].shape[1]/tiling_dim)):
                # calculate coordinates for information
                min_y = (y * tiling_dim) + (tile * tiling_offset)
                min_x = (x * tiling_dim) + (tile * tiling_offset)
                max_y = ((y * tiling_dim) + tiling_dim) + (tile * tiling_offset)
                max_x = ((x * tiling_dim) + tiling_dim) + (tile * tiling_offset)

                pos_range = np.array([
                    0, min_y, min_x,
                    seg.stacks.lamin.shape[0], max_y, max_x,
                ])

                print('get params for', info['ID'], pos_range)

                # get nuclei in square
                nIDs_in_square = seg.nuclei.get_nID_by_pos_range(pos_range, only_by_centre=only_centre)

                # get parameter for nuclei
                for param in params_to_get[1:]:
                    count_lists[param] = list()

                    # prepare layer stats
                    is_layer_stats = False
                    is_param_layer = False

                    if re.match(r'layer_.*_[0-9]*$', param) is not None:
                        is_layer_stats = True

                        # extract param and layer from string
                        layer_search = re.search(r'_[0-9]*$', param)
                        param_search = re.match(r'layer_.*_', param)

                        layer_id = int(layer_search.group()[1:])
                        layer_param = param_search.group()[6:-1]

                    # prepare for between space
                    if param == 'between_space' or re.match(r'layer_between_space_[0-9]*$', param) is not None:
                        bts_top_z = -1
                        bts_bottom_z = -1

                    for nID in nIDs_in_square:
                        if hasattr(seg.nuclei, 'get_nucleus_' + param):
                            val = getattr(seg.nuclei, 'get_nucleus_' + param)(nID)

                            if param == 'mami_axis':
                                val *= 100

                            if val is not None and math.isnan(val) is False:
                                count_lists[param].append(val)
                        else:
                            # check layer stats
                            if is_layer_stats is True:
                                # get layer from nucleus
                                if seg.nuclei.get_nucleus_nuclei_in_direction(nID) == layer_id:
                                    is_param_layer = True

                                    if layer_param == 'between_space':
                                        nucleus_centre = seg.nuclei.get_nucleus_centre(nID)

                                        if bts_top_z < 0 or nucleus_centre[0] < bts_top_z:
                                            bts_top_z = nucleus_centre[0]

                                        if bts_bottom_z < 0 or nucleus_centre[0] > bts_bottom_z:
                                            bts_bottom_z = nucleus_centre[0]
                                    if layer_param == 'nuclei_number':
                                        count_lists[param].append(1)
                                    elif layer_param == 'volume':
                                        count_lists[param].append(seg.nuclei.get_nucleus_volume(nID))
                                    elif layer_param == 'sphericity':
                                        surface = seg.nuclei.get_nucleus_surface(nID)
                                        volume = seg.nuclei.get_nucleus_volume(nID)

                                        sphericity = (math.pow(math.pi, (1/3)) * math.pow((6 * volume), (2/3)))/surface
                                        count_lists[param].append((sphericity * 100))

                            # check overall stats
                            if param == 'between_space':
                                nucleus_centre = seg.nuclei.get_nucleus_centre(nID)

                                if bts_top_z < 0 or nucleus_centre[0] < bts_top_z:
                                    bts_top_z = nucleus_centre[0]

                                if bts_bottom_z < 0 or nucleus_centre[0] > bts_bottom_z:
                                    bts_bottom_z = nucleus_centre[0]
                            elif param == 'sphericity':
                                # get surface and volume and calculate sphericity
                                surface = seg.nuclei.get_nucleus_surface(nID)
                                volume = seg.nuclei.get_nucleus_volume(nID)

                                sphericity = (math.pow(math.pi, (1/3)) * math.pow((6 * volume), (2/3)))/surface
                                count_lists[param].append((sphericity * 100))
                            elif param == 'layer_height':
                                # get apical distance
                                count_lists[param].append(seg.nuclei.get_nucleus_apical_distance(nID))
                            elif param == 'apical_dist_bin':
                                # get apical distance and nuclei in direction
                                apical_distance = seg.nuclei.get_nucleus_apical_distance(nID)
                                nuclei_in_direction = seg.nuclei.get_nucleus_nuclei_in_direction(nID)

                                if nuclei_in_direction > 0:
                                    count_lists[param].append(apical_distance/nuclei_in_direction)
                            elif param == 'avg_pile':
                                # add pile information from nucleus
                                count_lists[param].append(seg.nuclei.get_nucleus_nuclei_in_direction(nID))
                            elif param == 'avg_apical_dist':
                                # add pile information from nucleus
                                count_lists[param].append(seg.nuclei.get_nucleus_apical_distance(nID))

                    # store top and bottom z
                    if param == 'between_space' or (is_param_layer is True and layer_param == 'between_space'):
                        count_lists[param].append((bts_top_z, bts_bottom_z))

                    # dummies
                    if param == 'nuclei_per_layer'\
                            or param == 'volumes_per_layer'\
                            or param == 'membin_stack_volume'\
                            or param == 'nuclei_stack_volume' \
                            or param == 'density'\
                            or param == 'nuclei_per_layer_height'\
                            or param == 'volumes_per_layer_height'\
                            or param == 'layer_height_per_layer':
                        count_lists[param].append(1)

                #####
                # Use values from the count lists and average them into tiling maps
                # Then use the tiling maps to create an average for the actual maps
                #####

                # value for this square
                nIDs_in_tile = 0
                vals_for_tile = dict()

                # get averages and set maps
                if len(nIDs_in_square) > 0:
                    nIDs_in_tile = len(nIDs_in_square)

                tiling_maps[params_to_get[0]][info['ID']][tile][min_y:max_y, min_x:max_x] = nIDs_in_tile
                tiling_count_map[params_to_get[0]][info['ID']][min_y:max_y, min_x:max_x] += 1

                for param in params_to_get[1:]:
                    if len(count_lists[param]) > 0:
                        #print('TEST COUNT', count_lists[param])

                        # prepare layer height
                        if param == 'layer_height' \
                                or param == 'volumes_per_layer_height'\
                                or param == 'nuclei_per_layer_height'\
                                or param == 'layer_height_per_layer':
                            if len(count_lists['layer_height']) > 0:
                                layer_height = max(count_lists['layer_height'])
                            else:
                                layer_height = -1

                        if param == 'nuclei_in_direction':
                            vals_for_tile[param] = max(count_lists[param])
                        elif param == 'nuclei_per_layer':
                            if len(count_lists['nuclei_in_direction']) > 0:
                                vals_for_tile[param] = len(nIDs_in_square)/max(count_lists['nuclei_in_direction'])
                            else:
                                vals_for_tile[param] = 0
                        elif param == 'volumes_per_layer':
                            if len(count_lists['nuclei_in_direction']) > 0:
                                vals_for_tile[param] = \
                                    sum(count_lists['volume'])/max(count_lists['nuclei_in_direction'])
                            else:
                                vals_for_tile[param] = 0
                        elif param == 'between_space' or re.match(r'layer_between_space_[0-9]*$', param) is not None:
                            # get all values from top to bottom
                            between_space = np.sum(filled_minus_nuclei[
                                                   count_lists[param][0][0]:count_lists[param][0][1],
                                                   min_y:max_y, min_x:max_x
                                                   ])

                            space_per_nucleus = between_space/len(nIDs_in_square)

                            if np.isnan(space_per_nucleus):
                                space_per_nucleus = -1

                            vals_for_tile[param] = space_per_nucleus
                        elif re.match(r'layer_nuclei_number_[0-9]*$', param) is not None:
                            vals_for_tile[param] = sum(count_lists[param])
                        elif re.match(r'layer_volume_[0-9]*$', param) is not None:
                            vals_for_tile[param] = sum(count_lists[param])/len(count_lists[param])
                        elif re.match(r'layer_sphericity_[0-9]*$', param) is not None:
                            vals_for_tile[param] = sum(count_lists[param])/len(count_lists[param])
                        elif param == 'layer_height':
                            if layer_height >= 0:
                                vals_for_tile[param] = layer_height
                            else:
                                vals_for_tile[param] = 0
                        elif param == 'apical_dist_bin':
                            # take average
                            vals_for_tile[param] = sum(count_lists[param])/len(count_lists[param])
                        elif param == 'avg_pile':
                            # take average
                            vals_for_tile[param] = sum(count_lists[param])/len(count_lists[param])
                        elif param == 'avg_apical_dist':
                            # take average
                            vals_for_tile[param] = sum(count_lists[param])/len(count_lists[param])
                        elif param == 'volumes_per_layer_height':
                            volumes = sum(count_lists['volume'])

                            if layer_height >= 0:
                                vals_for_tile[param] = volumes/layer_height
                            else:
                                vals_for_tile[param] = 0
                        elif param == 'nuclei_per_layer_height':
                            nuclei_count = len(nIDs_in_square)

                            if layer_height >= 0:
                                vals_for_tile[param] = nuclei_count/layer_height
                            else:
                                vals_for_tile[param] = 0
                        elif param == 'layer_height_per_layer':
                            if layer_height >= 0 and len(count_lists['nuclei_in_direction']) > 0:
                                vals_for_tile[param] = max(count_lists['nuclei_in_direction'])/layer_height
                            else:
                                vals_for_tile[param] = 0
                        elif param == 'membin_stack_volume':
                            if seg.has_membin() is False:
                                seg.create_membin()

                            membin_volume = np.sum(seg.stacks.membin[:, min_y:max_y, min_x:max_x])

                            vals_for_tile[param] = membin_volume
                        elif param == 'nuclei_stack_volume':
                            nuclei_stack = seg.stacks.nuclei[:, min_y:max_y, min_x:max_x]

                            # convert values to ones
                            nuclei_stack[nuclei_stack > 0] = 1

                            nuclei_volume = np.sum(nuclei_stack)

                            vals_for_tile[param] = nuclei_volume
                        elif param == 'density':
                            membin_volume = np.sum(seg.stacks.membin[:, min_y:max_y, min_x:max_x])
                            nuclei_stack = seg.stacks.nuclei[:, min_y:max_y, min_x:max_x]

                            # convert values to ones
                            nuclei_stack[nuclei_stack > 0] = 1

                            nuclei_volume = np.sum(nuclei_stack)

                            if membin_volume > 0:
                                vals_for_tile[param] = nuclei_volume/membin_volume
                            else:
                                vals_for_tile[param] = 0
                        else:
                            vals_for_tile[param] = sum(count_lists[param])/len(count_lists[param])
                    else:
                        vals_for_tile[param] = 0

                    # set tile value for param
                    if vals_for_tile[param] > 0:
                        tiling_maps[param][info['ID']][tile][min_y:max_y, min_x:max_x] = vals_for_tile[param]

                    # count tiles as sampled
                    tiling_count_map[param][info['ID']][min_y:max_y, min_x:max_x] += 1
    # free memory
    del(seg)

# get min and max and average tiling maps
for param in params_to_get:
    for info in infos:
        # average tiling maps
        for tile in range(tiling_count):
            val_maps[param][info['ID']] += tiling_maps[param][info['ID']][tile]
        val_maps[param][info['ID']] = val_maps[param][info['ID']]/tiling_count_map[param][info['ID']]

        # account for nan ie/ no values counted for this part of the image
        val_maps[param][info['ID']][np.isnan(val_maps[param][info['ID']])] = 0

        over_zero_array = val_maps[param][info['ID']][val_maps[param][info['ID']] > 0]

        smallest = 0
        largest = 0

        if len(over_zero_array) > 0:
            smallest = val_maps[param][info['ID']][val_maps[param][info['ID']] > 0].min()
            largest = val_maps[param][info['ID']][val_maps[param][info['ID']] > 0].max()

        limit_lists[param][0].append(smallest)
        limit_lists[param][1].append(largest)

    limits[param][0] = sum(limit_lists[param][0])/len(limit_lists[param][0])
    limits[param][1] = sum(limit_lists[param][1])/len(limit_lists[param][1])

# adjust limits for layers for them to be equal
for layer_param in layers_to_get:
    smallest = -1
    largest = -1

    # go through layers and get the limits
    for layer_id in range(1, layers_to_get_count + 1):
        cur_param = 'layer_' + layer_param + '_' + str(layer_id)

        if smallest < 0 or smallest > limits[cur_param][0]:
            smallest = limits[cur_param][0]

        if largest < 0 or largest < limits[cur_param][1]:
            largest = limits[cur_param][1]

    # set limits
    for layer_id in range(1, layers_to_get_count + 1):
        cur_param = 'layer_' + layer_param + '_' + str(layer_id)

        limits[cur_param][0] = smallest
        limits[cur_param][1] = largest

# prepare directories
dir_analysis = 'analysis' + os.sep
dir_maps = dir_analysis + 'maps_d' + str(tiling_dim) + os.sep
dir_lists = dir_analysis + 'lists_d' + str(tiling_dim) + os.sep

# create directories
ImageHandler.create_dir(dir_maps)
ImageHandler.create_dir(dir_lists)

# build average lists
avg_lists = dict()
for param in params_to_get:
    avg_lists[param] = dict()

    for info_ID in selected_info_IDs:
        avg_lists[param][info_ID] = list()

        for y in range(1, int(img_maps[param][info_ID].shape[0]), tiling_offset):
            for x in range(1, int(img_maps[param][info_ID].shape[1]), tiling_offset):
                # get value from value maps
                avg_lists[param][info_ID].append(val_maps[param][info_ID][y, x])

# prepare maps
for info in infos:
    for param in params_to_get:
        # get divisor
        divisor = (limits[param][1] - limits[param][0])

        if divisor <= 0:
            divisor = 1

        # set relative value in image maps according to limits
        img_maps[param][info['ID']] = (val_maps[param][info['ID']] - limits[param][0])/divisor

        # account for value below zero or above threshold
        img_maps[param][info['ID']][img_maps[param][info['ID']] < 0] = 0
        img_maps[param][info['ID']][img_maps[param][info['ID']] > 1] = 1

        # set colour
        img_maps[param][info['ID']] = img_maps[param][info['ID']] * 255

        # convert data type
        img_maps[param][info['ID']] = img_maps[param][info['ID']].astype(np.uint8)

    # save maps
    print('Save maps for %s' % (info['ID']))

    for param in params_to_get:
        print('\t%s' % param)
        ImageHandler.save_stack_as_tiff(img_maps[param][info['ID']],
                                        dir_maps + info['ID'] + '_' + param + '.tif')

    if save_csv is True:
        # save lists
        print('Save correlation lists for %s' % (info['ID']))

        # combine lists
        rows = list()

        # add headings
        rows.append(list())

        for param in params_to_get:
            rows[-1].append(param)

        # add values
        for i in range(len(avg_lists[params_to_get[0]][info['ID']])):
            rows.append(list())

            for param in params_to_get:
                rows[-1].append(avg_lists[param][info['ID']][i])

        # save to csv
        with open(dir_lists + 'corr_' + info['ID'] + '.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file,
                                    delimiter=cfg.CSV_DEL,
                                    quotechar=cfg.CSV_QUOT)

            for row in rows:
                csv_writer.writerow(row)

        csv_file.close()

if save_csv is True:
    for param in params_to_get:
        print('Save histogram lists for %s' % (param))
        row_name = ''
        write_to_disc = True
        is_param_layer = False
        row_offset = 0

        # adjust csv for layers
        if re.match(r'layer_.*_[0-9]*$', param) is not None:
            is_param_layer = True

            # extract param and layer from string
            layer_id = int(re.search(r'_[0-9]*$', param).group()[1:])
            layer_param = re.match(r'layer_.*_', param).group()[6:-1]

            # set row name
            row_name = layer_id

            if layer_id < layers_to_get_count:
                write_to_disc = False

        if is_param_layer is False or (is_param_layer is True and layer_id == 1):
            # save lists for histograms
            rows = list()

            # add headings
            rows.append(['Name'])

            for info_ID in selected_info_IDs:
                rows[-1].append(info_ID)

        if is_param_layer is True:
            # adjust row offset for next layer
            if layer_id > 1:
                row_offset = len(rows) - 1

        for i, info_ID in enumerate(selected_info_IDs):
            for j, val in enumerate(avg_lists[param][info_ID]):
                # build row to write
                if (j + row_offset) >= (len(rows) - 1):
                    rows.append([row_name] + (len(selected_info_IDs)*[np.nan]))

                if val == 0:
                    val = np.nan

                rows[(j + 1 + row_offset)][i + 1] = val

        if write_to_disc is True:
            file_name = param

            if is_param_layer is True:
                file_name = 'layer_' + layer_param

            # save to csv
            with open(dir_lists + 'hist_' + file_name + '.csv', 'w') as csv_file:
                csv_writer = csv.writer(csv_file,
                                        delimiter=cfg.CSV_DEL,
                                        quotechar=cfg.CSV_QUOT)

                for row in rows:
                    csv_writer.writerow(row)

            csv_file.close()

if plot_figures == True:
    print('Plot')

    fig = plt.figure()

    # figures
    i = 0
    for info in infos:
        for j, param in enumerate(params_to_get):
            a = fig.add_subplot(2, len(params_to_get)*len(selected_info_IDs), (j+1)+(len(selected_info_IDs) * i))
            plt.imshow(img_maps[param][info['ID']])
            a.set_title(param)

        i += 1

    """
    # histograms
    i = 0
    for info in infos:
        # plot data
        for j, param in enumerate(params_to_get):
            print('TEST PARAM', param, avg_lists[param], limits[param])
            a = fig.add_subplot(2, len(params_to_get)*len(selected_info_IDs), (j+1)+(len(params_to_get)*len(selected_info_IDs))+(len(selected_info_IDs) * i))

            dist = int((limits[param][1] - limits[param][0])/100)

            if dist > 0:
                bin_val = dist
            else:
                bin_val = 1

            print('TEST BIN', bin_val, range(int(limits[param][0]), int(limits[param][1] + 1), bin_val))

            plt.hist(avg_lists[param][info['ID']], bins=50, color='blue')
            a.set_title(param)

        i += 1
    """
