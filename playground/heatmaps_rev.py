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
selected_info_IDs = ['N1-19-9', 'N1-19-22', 'N1-19-23']
params_to_get = ['nuclei_number', 'volume',  'between_space', 'sphericity',
                 'layer_height', 'volumes_per_layer_height', 'nuclei_per_layer_height',
                 'nuclei_in_direction', 'layer_height_per_layer', 'nuclei_per_layer', 'volumes_per_layer',
                 'membin_stack_volume', 'nuclei_stack_volume', 'density', 'nuclei_per_volume',
                 'layer_height_per_membin_stack_volume']
#layers_to_get = ['nuclei_number', 'volume', 'between_space', 'sphericity']
layers_to_get = list()
layers_to_get_count = 3

# add layers to params
for layer_param in layers_to_get:
    for layer_id in range(1, layers_to_get_count + 1):
        params_to_get.append('layer_' + layer_param + '_' + str(layer_id))

dim = 20
limit_nvals = 5
save_csv = True
plot_figures = False
only_centre = False

# open window to select nuclei criteria
loaded_infos = ImageHandler.load_image_infos()
infos = list()

for info in loaded_infos:
    info_ID = info['ID']
    if info_ID in selected_info_IDs:
        infos.append(info)
        print('TEST APPEND', info)

maps = dict()
avg_lists = dict()
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

    # init maps and lists
    for param in params_to_get:
        if (param in maps) is False:
            maps[param] = dict()
            avg_lists[param] = dict()
            limits[param] = [-1, -1]
            limit_lists[param] = [list(), list()]

        maps[param][info['ID']] = np.zeros_like(seg.stacks.lamin[0])
        avg_lists[param][info['ID']] = list()

    # go through the image and get parameters from nuclei
    for y in range(int(maps[params_to_get[0]][info['ID']].shape[0]/dim)):
        for x in range(int(maps[params_to_get[0]][info['ID']].shape[1]/dim)):
            min_y = (y * dim)
            min_x = (x * dim)
            max_y = ((y * dim) + dim)
            max_x = ((x * dim) + dim)

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

            # get averages and set maps
            if len(nIDs_in_square) > 0:
                avg_lists[params_to_get[0]][info['ID']].append(len(nIDs_in_square))
            else:
                avg_lists[params_to_get[0]][info['ID']].append(0)

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
                        avg_lists[param][info['ID']].append(max(count_lists[param]))
                    elif param == 'nuclei_per_layer':
                        if len(count_lists['nuclei_in_direction']) > 0:
                            avg_lists[param][info['ID']].append(
                                len(nIDs_in_square)/max(count_lists['nuclei_in_direction'])
                            )
                        else:
                            avg_lists[param][info['ID']].append(0)
                    elif param == 'volumes_per_layer':
                        if len(count_lists['nuclei_in_direction']) > 0:
                            avg_lists[param][info['ID']].append(
                                sum(count_lists['volume'])/max(count_lists['nuclei_in_direction'])
                            )
                        else:
                            avg_lists[param][info['ID']].append(0)
                    elif param == 'between_space' or re.match(r'layer_between_space_[0-9]*$', param) is not None:
                        # get all values from top to bottom
                        between_space = np.sum(filled_minus_nuclei[
                                               count_lists[param][0][0]:count_lists[param][0][1],
                                               min_y:max_y, min_x:max_x
                                               ])

                        space_per_nucleus = between_space/len(nIDs_in_square)

                        if np.isnan(space_per_nucleus):
                            space_per_nucleus = -1

                        avg_lists[param][info['ID']].append(space_per_nucleus)
                    elif re.match(r'layer_nuclei_number_[0-9]*$', param) is not None:
                        avg_lists[param][info['ID']].append(sum(count_lists[param]))
                    elif re.match(r'layer_volume_[0-9]*$', param) is not None:
                        avg_lists[param][info['ID']].append(sum(count_lists[param])/len(count_lists[param]))
                    elif re.match(r'layer_sphericity_[0-9]*$', param) is not None:
                        avg_lists[param][info['ID']].append(sum(count_lists[param])/len(count_lists[param]))
                    elif param == 'layer_height':
                        if layer_height >= 0:
                            avg_lists[param][info['ID']].append(layer_height)
                        else:
                            avg_lists[param][info['ID']].append(0)
                    elif param == 'volumes_per_layer_height':
                        volumes = sum(count_lists['volume'])

                        if layer_height >= 0:
                            avg_lists[param][info['ID']].append(volumes/layer_height)
                        else:
                            avg_lists[param][info['ID']].append(0)
                    elif param == 'nuclei_per_layer_height':
                        nuclei_count = len(nIDs_in_square)

                        if layer_height >= 0:
                            avg_lists[param][info['ID']].append(nuclei_count/layer_height)
                        else:
                            avg_lists[param][info['ID']].append(0)
                    elif param == 'layer_height_per_layer':
                        if layer_height >= 0 and len(count_lists['nuclei_in_direction']) > 0:
                            avg_lists[param][info['ID']].append(max(count_lists['nuclei_in_direction'])/layer_height)
                        else:
                            avg_lists[param][info['ID']].append(0)
                    elif param == 'membin_stack_volume':
                        if seg.has_membin() is False:
                            seg.create_membin()

                        membin_volume = np.sum(seg.stacks.membin[:, min_y:max_y, min_x:max_x])

                        avg_lists[param][info['ID']].append(membin_volume)
                    elif param == 'nuclei_stack_volume':
                        nuclei_stack = seg.stacks.nuclei[:, min_y:max_y, min_x:max_x]

                        # convert values to ones
                        nuclei_stack[nuclei_stack > 0] = 1

                        nuclei_volume = np.sum(nuclei_stack)

                        avg_lists[param][info['ID']].append(nuclei_volume)
                    elif param == 'density':
                        membin_volume = np.sum(seg.stacks.membin[:, min_y:max_y, min_x:max_x])
                        nuclei_stack = seg.stacks.nuclei[:, min_y:max_y, min_x:max_x]

                        # convert values to ones
                        nuclei_stack[nuclei_stack > 0] = 1

                        nuclei_volume = np.sum(nuclei_stack)

                        if membin_volume > 0:
                            avg_lists[param][info['ID']].append(nuclei_volume/membin_volume)
                        else:
                            avg_lists[param][info['ID']].append(0)
                    else:
                        avg_lists[param][info['ID']].append(sum(count_lists[param])/len(count_lists[param]))
                else:
                    avg_lists[param][info['ID']].append(0)
    # free memory
    del(seg)

# get min and max
for param in params_to_get:
    for info in infos:
        smallest = heapq.nsmallest(limit_nvals, avg_lists[param][info['ID']])
        largest = heapq.nlargest(limit_nvals, avg_lists[param][info['ID']])

        for val in smallest:
            limit_lists[param][0].append(val)
        for val in largest:
            limit_lists[param][1].append(val)

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
dir_maps = dir_analysis + 'maps_d' + str(dim) + os.sep
dir_lists = dir_analysis + 'lists_d' + str(dim) + os.sep

# create directories
ImageHandler.create_dir(dir_maps)
ImageHandler.create_dir(dir_lists)

# to store colour values in list
colour_vals = OrderedDict()

# prepare maps
for info in infos:
    counter = 0
    for y in range(int(maps[params_to_get[0]][info['ID']].shape[0]/dim)):
        for x in range(int(maps[params_to_get[0]][info['ID']].shape[1]/dim)):
            min_y = (y * dim)
            min_x = (x * dim)
            max_y = ((y * dim) + dim)
            max_x = ((x * dim) + dim)

            print('set map for', info['ID'], min_y, min_x, max_y, max_x)

            for param in params_to_get:
                cur_val = avg_lists[param][info['ID']][counter]

                if cur_val > limits[param][1]:
                    rel_val = 1
                elif cur_val < limits[param][0]:
                    rel_val = 0
                else:
                    divisor = (limits[param][1] - limits[param][0])

                    if divisor <= 0:
                        divisor = 1

                    rel_val = ((cur_val - limits[param][0])/divisor)

                colour_val = rel_val * 255

                maps[param][info['ID']][min_y:max_y, min_x:max_x] = colour_val

                # add colour value to list
                if param not in colour_vals.keys():
                    colour_vals[param] = OrderedDict()

                if info['ID'] not in colour_vals[param].keys():
                    colour_vals[param][info['ID']] = list()

                colour_vals[param][info['ID']].append(colour_val)

            counter += 1

    # save maps
    print('Save maps for %s' % (info['ID']))

    for param in params_to_get:
        print('\t%s' % param)
        ImageHandler.save_stack_as_tiff(maps[param][info['ID']],
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

    # save csv for colour values
    for param, param_vals in colour_vals.items():
        print('Save colour histograms for %s' % (param))
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

        info_counter = 0
        for info_ID, colour_vals in param_vals.items():
            for j, val in enumerate(colour_vals):
                # build row to write
                if (j + row_offset) >= (len(rows) - 1):
                    rows.append([row_name] + (len(selected_info_IDs)*[np.nan]))

                if val == 0:
                    val = np.nan

                rows[(j + 1 + row_offset)][info_counter + 1] = val

            info_counter += 1

        if write_to_disc is True:
            file_name = param

            if is_param_layer is True:
                file_name = 'layer_' + layer_param

            # save to csv
            with open(dir_lists + 'hist_colour_' + file_name + '.csv', 'w') as csv_file:
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
            plt.imshow(maps[param][info['ID']])
            a.set_title(param)

        i += 1

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
