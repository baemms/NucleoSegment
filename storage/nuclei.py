"""
Manage nuclei storage, creation and queries
"""

import numpy as np
import random as rdm
import pickle
import os
import math

from processing.filter import Dilation, Equalise
import threading

from storage.images import Image
from storage.lookup_frame import LookupFrame
from frontend.figures.plot import Plot

import storage.config as cfg

class Nuclei:

    def __init__(self, segmentation):
        """
        Init nuclei dataframes
        """
        self.segmentation = segmentation

        self.data_frames = dict()

        # init lookup frames for data storage
        for col in cfg.pd_struct_nuclei_cols.items():
            self.data_frames[col[0]] = LookupFrame(self, col[1])

        # init dicts for formats that are not convenient for dataframes
        self.extra = dict()
        for extra_info in ['img_projections', 'img_boxes', 'infos']:
            self.extra[extra_info] = dict()

        # init dicts
        for projection in ['z', 'y', 'x']:
            self.extra['img_projections'][projection] = dict()

        self.img_boxes = dict()
        for box in ['bw', 'lamin', 'labels', 'dapi', 'membrane',
                    'crop_lamin', 'crop_dapi', 'crop_membrane',
                    'crop_rgb']:
            self.img_boxes[box] = dict()

        for info in ['img', 'lamin_slice']:
            self.extra['infos'][info] = dict()

    def create_nucleus(self, z, nucleus_label):
        """
        Create dataframe entries for new nucleus
        and create a new nID and return

        :param z:
        :param nucleus_label:
        :return:
        """
        # fill parameter columns in data frames
        # create dummy nucleus
        dummy_params = np.full((1, len(cfg.pd_struct_nuclei_cols['data_params'])), -1, dtype=np.int8)

        # create a new nucleus and get the ID
        nID = self.data_frames['data_params'].create_data_row(dummy_params)

        # add colour!
        self.set_nucleus_colour(nucleus_label['colour'], nID)

        self.add_to_nucleus(z, nucleus_label, nID)

        return nID

    def add_to_nucleus(self, z, nucleus_label, nID, remove_before=False):
        """
        Add data to a nucleus

        :param z:
        :param nucleus_label:
        :param nID:
        :param remove_before:
        :return:
        """
        # get data
        data_centroid = np.array([
            [z, nucleus_label['centroid'][0], nucleus_label['centroid'][1]]
        ])
        data_bbox = np.array([
            [z,
             nucleus_label['bbox'][0], nucleus_label['bbox'][1],
             nucleus_label['bbox'][2], nucleus_label['bbox'][3]]
        ])
        data_z_params = np.array([
            [z, nucleus_label['area'], nucleus_label['perimeter']]
        ])

        # create z array for coords
        data_coords_z = np.full((len(nucleus_label['coords']), 1), z, dtype=np.uint16)

        # add z to coords
        data_coords = np.c_[data_coords_z, nucleus_label['coords']]

        # delete data from frames
        if remove_before is True:
            self.del_from_nucleus(z, nID)

        # add data to frames
        self.data_frames['data_centroid'].add_data_row(nID, data_centroid)
        self.data_frames['data_bbox'].add_data_row(nID, data_bbox)
        self.data_frames['data_z_params'].add_data_row(nID, data_z_params)
        self.data_frames['data_coords'].add_data_row(nID, data_coords)

    def del_from_nucleus(self, z, nID):
        """
        Map more data to a nucleus

        :param label:
        :return:
        """
        # delete data from frames
        self.data_frames['data_centroid'].del_data_row(nID, 'z', z)
        self.data_frames['data_bbox'].del_data_row(nID, 'z', z)
        self.data_frames['data_z_params'].del_data_row(nID, 'z', z)
        self.data_frames['data_coords'].del_data_row(nID, 'z', z)

    def sort_vals_by_z(self):
        """
        Sort all values by z

        :return:
        """
        self.data_frames['data_centroid'].sort_by_col('z', sort_by_nID=True, asc=[True])
        self.data_frames['data_bbox'].sort_by_col('z', sort_by_nID=True, asc=[True])
        self.data_frames['data_z_params'].sort_by_col('z', sort_by_nID=True, asc=[True])
        self.data_frames['data_coords'].sort_by_col('z', sort_by_nID=True, asc=[True])

    def postprocess_for_nuclei(self, dummy_frame):
        """
        Make final changes to nucleus after merging

        :param dummy_frame:
        :return:
        """
        for nID in self.get_nIDs():
            self.postprocess_for_nucleus(dummy_frame, nID)

    def postprocess_for_nucleus(self, dummy_frame, nID):
        """
        Apply postprocessing for nucleus

        :param nID:
        :param dummy_frame:
        :return:
        """
        # dilate nuclei
        dilate = Dilation()

        # go through coordinates and add to frame
        curr_z = -1
        prev_z = -2

        frame = None
        nucleus_coords = self.get_nucleus_coords(nID)
        new_coords = list()

        for coords in nucleus_coords:
            # init frame
            if curr_z != prev_z:
                if frame is not None:
                    # dilate
                    dilated = dilate.apply(frame, {'size': cfg.merge_post_dil, 'bin': None})

                    # get coordinates
                    coords_dilated = np.argwhere(dilated)

                    # add to coordinates list
                    for coord in coords_dilated:
                        new_coords.append([coords[0], coord[0], coord[1]])

                frame = np.zeros_like(dummy_frame)

            frame[int(coords[1]), int(coords[2])] = 1

        # update coords
        self.set_nucleus_coords(np.array(new_coords), nID)

    def update_nucleus(self, new_nucleus):
        """
        Update nucleus in nuclei

        :param new_nucleus:
        :return:
        """
        self.set_nucleus_by_id(new_nucleus['nID'], new_nucleus)

    def remerge_nucleus(self, nID, corr_start, corr_stop,
                        merge_depth=False, force_raw_labels_props=False):
        """
        Remerge nucleus while ignoring the merge depth parameter

        :return:
        """
        nucleus_coords = self.get_nucleus_coords(nID)

        # get first and last plane of the nucleus and coordinates
        nucleus_start = int(float(nucleus_coords[0, 0]))
        nucleus_stop = int(float(nucleus_coords[-1, 0]))

        print('nID-%i OLD: %i - %i NEW: %i - %i' % (nID, nucleus_start, nucleus_stop, corr_start, corr_stop))

        # correction needed?
        corr_needed = True

        if nucleus_start == corr_start and nucleus_stop == corr_stop:
            corr_needed = False

        if corr_needed:
            # correction top
            if nucleus_start > corr_start:
                # merge up
                print('merge up (%i - %i)' % (nucleus_start, corr_start))
                self.segmentation.remerge_nucleus_part(nID, nucleus_start, corr_start,
                                                       merge_depth=merge_depth, force_raw_labels_props=force_raw_labels_props)
            elif nucleus_start < corr_start:
                # delete planes
                print('delete top (%i - %i)' % (corr_start, nucleus_start))
                self.segmentation.delete_nucleus_part(nID, corr_start, nucleus_start)

            # correct bottom
            if nucleus_stop < corr_stop:
                # merge down
                print('merge down (%i - %i)' % (nucleus_stop, corr_stop))
                self.segmentation.remerge_nucleus_part(nID, nucleus_stop, corr_stop,
                                                       merge_depth=merge_depth, force_raw_labels_props=force_raw_labels_props)
            elif nucleus_stop > corr_stop:
                # delete planes
                print('delete bottom (%i - %i)' % (corr_stop, nucleus_stop))
                self.segmentation.delete_nucleus_part(nID, corr_stop, nucleus_stop)

            # reorder z parameters
            self.sort_vals_by_z()

            nucleus_coords = self.get_nucleus_coords(nID)
            nucleus_start = nucleus_coords[0, 0]
            nucleus_stop = nucleus_coords[-1, 0]

            print('nID-%i RESULT: %i - %i' % (nID, nucleus_start, nucleus_stop))

            # postprocess
            #self.postprocess_for_nucleus(self.segmentation.stacks.nuclei[0], nID)

            # recalculate params
            self.calc_nucleus_params(nID)

            params_path = self.segmentation.get_results_dir().nuclei_params_corr
            self.save(params_path)

    def filter_nuclei(self):
        """
        Filter nuclei on specific requirements

        :return:
        """
        filtered_nuclei = list()
        to_be_removed_nuclei = list()

        print('Filter nuclei')

        for i, nID in enumerate(self.get_nIDs()):
            if i % 100 == 0:
                print('\t%i' % i)

            criteria_met = True

            # go through nuclei criteria
            for label_filter in cfg.filter_criteria_nuclei:
                # TODO look up params in params and classifier data frames
                if self.segmentation.is_param_in_criteria_range(
                        self.get_nucleus_param_dict(nID), label_filter) is False:
                    criteria_met = False

            # if all criteria are met, add the nucleus to the filtered list
            if criteria_met is True:
                filtered_nuclei.append(nID)
            else:
                to_be_removed_nuclei.append(nID)

        # how nuclei survived the filtering?
        print('Nuclei accepted: %i; rejected: %i'
              % (len(filtered_nuclei), len(self.get_nIDs()) - len(filtered_nuclei)))

        self.reject_nuclei(to_be_removed_nuclei)

        return to_be_removed_nuclei

    def reject_nuclei(self, to_remove):
        """
        Mark nuclei as rejected

        :param to_remove:
        :return:
        """
        for nID in to_remove:
            self.reject_nucleus(nID)

    def get_nucleus_param_dict(self, nID):
        """
        create a dictionary with nucleus parameters

        :param nID:
        :return:
        """
        nID_params = dict()

        # get all parameters from single valued fields
        for data_frame in ['data_params', 'data_params', 'data_params']:
            for col in cfg.pd_struct_nuclei_cols[data_frame]:
                nID_params[col] = self.get_param_from_nucleus(data_frame, col, nID)

        return nID_params

    def get_validated_params_for_nucleus(self, nucleus):
        """
        Return validated params for nucleus

        :param nucleus:
        :return:
        """
        # TODO how to best return a validated collection for the params
        # get labels for nucleus
        nucleus_labels = self.build_label_from_nucleus(nucleus)

        # build dict for params
        validated_nucleus_params = dict()
        validated_labels_params = dict()

        # validate nuclei
        for nuclei_filter in cfg.filter_criteria_nuclei:
            cur_filter = nuclei_filter.lower()

            if self.is_param_in_nucleus(cur_filter) is True:
                failed_filter = False

                if self.is_param_in_criteria_range(nucleus, nuclei_filter) is False:
                    failed_filter = True

                validated_nucleus_params[cur_filter] = dict()
                validated_nucleus_params[cur_filter]['value'] = nucleus[cur_filter]
                validated_nucleus_params[cur_filter]['error'] = failed_filter

        # validate labels
        for i, labels_props in enumerate(nucleus_labels):
            for labels_filter in cfg.filter_criteria_labels:
                cur_filter = labels_filter.lower()

                if i == 0:
                    validated_labels_params[cur_filter] = dict()

                failed_filter = False

                if self.is_param_in_criteria_range(labels_props, labels_filter) is False:
                    failed_filter = True

                if i == 0 or labels_props[cur_filter] < validated_labels_params[cur_filter]['min']:
                    validated_labels_params[cur_filter]['min'] = labels_props[cur_filter]

                if i == 0 or labels_props[cur_filter] > validated_labels_params[cur_filter]['max']:
                    validated_labels_params[cur_filter]['max'] = labels_props[cur_filter]

                validated_labels_params[cur_filter]['error'] = failed_filter

        nucleus_props = [validated_nucleus_params, validated_labels_params]

        return nucleus_props

    @staticmethod
    def create_nID_by_nuclei(nucleus, nuclei, offset=0):
        """
        Generate a new nucleus ID based on the nuclei list given

        :param nucleus:
        :param nuclei:
        :param offset:
        :return:
        """
        nucleus['nID'] = len(nuclei) + offset

        return nucleus

    def convert_nucleus_label_props_to_lists(self, nucleus):
        """
        Convert nucleus label props to lists for easier editing

        :param nucleus:
        :return:
        """
        for prop in cfg.label_props_to_get_keys:
            if prop in nucleus and type(nucleus[prop]) is np.ndarray:
                nucleus[prop] = nucleus[prop].tolist()

        return nucleus

    def get_array_lists_for_nucleus_label_props(self, nucleus, add_props=None):
        """
        Get lists ready for panda conversion for label props

        :param nucleus:
        :param add_props:
        :return:
        """
        props_to_convert = cfg.label_props_to_get_keys
        props_list = dict()

        if add_props is not None:
            props_to_convert += add_props

        for prop in props_to_convert:
            if prop in nucleus:
                if type(nucleus[prop]) is list:
                    if type(nucleus[prop][0]) is tuple:
                        data = list()
                        # 1D tuple
                        if type(nucleus[prop][0][1]) is tuple:
                            for prop_z in nucleus[prop]:
                                data.append(list())

                                data[-1].append(prop_z[0])
                                for prop_val in prop_z[1]:
                                    data[-1].append(prop_val)
                        # 2D array
                        elif type(nucleus[prop][0][1]) is np.ndarray:
                            for prop_z in nucleus[prop]:
                                for prop_frame in prop_z[1]:
                                    data.append(list())

                                    data[-1].append(prop_z[0])
                                    for prop_val in prop_frame:
                                        data[-1].append(float(prop_val))
                        # single value
                        else:
                            for prop_z in nucleus[prop]:
                                data.append([prop_z[0], prop_z[1]])

                props_list[prop] = data

        return props_list

    def get_label_props_from_array_lists(self, nucleus, data_struct):
        """
        Convert array lists to label props

        :param nucleus:
        :param data_struct:
        :return:
        """

        for prop in cfg.pd_struct_nuclei_cols:
            # get prop from struct
            data_frame = getattr(data_struct, prop)

            # slice nID from struct
            data_nID = data_frame.loc[nucleus['nID']]

            if prop in cfg.pd_struct_nuclei_col_types:
                # go through columns and map data back into list structures
                if cfg.pd_struct_nuclei_col_types[prop] == cfg.PD_STRUCT_TYPE_VAL:
                    # single values
                    for kID, value in data_nID.iteritems():
                        # take the column id and use for dict value assignment
                        nucleus[kID] = value
                elif cfg.pd_struct_nuclei_col_types[prop] == cfg.PD_STRUCT_TYPE_TUPLE:
                    # tuple
                    new_tuple = tuple()

                    # go through items
                    for kID, value in data_nID.iteritems():
                        # add to new tuple
                        new_tuple += (value, )

                    key = prop.split('_')[-1]
                    nucleus[key] = new_tuple
                elif cfg.pd_struct_nuclei_col_types[prop] in [cfg.PD_STRUCT_TYPE_1D, cfg.PD_STRUCT_TYPE_1D_VAL]:
                    # 1D
                    z_list = list()
                    val_list = list()

                    # go through cols
                    for id_col, data_col in data_nID.iteritems():
                        counter = 0

                        # go through rows
                        for kID, value in data_col.iteritems():
                            if counter < len(z_list):
                                # add to values
                                if cfg.pd_struct_nuclei_col_types[prop] == cfg.PD_STRUCT_TYPE_1D:
                                    val_list[counter].append(value)
                                else:
                                    val_list.append(value)
                            else:
                                # create new entry for z and values
                                z_list.append(int(value))
                                if cfg.pd_struct_nuclei_col_types[prop] == cfg.PD_STRUCT_TYPE_1D:
                                    val_list.append(list())

                            counter += 1

                    # build up array
                    new_list = list()
                    for i, z in enumerate(z_list):
                        # multiple values in list?
                        if type(val_list[i]) == list and len(val_list[i]) > 1:
                            new_value = tuple()
                            for val in val_list[i]:
                                new_value += (val, )
                        else:
                            new_value = val_list[i]

                        new_list.append((z, new_value))
                    key = prop.split('_')[-1]
                    nucleus[key] = new_list
                elif cfg.pd_struct_nuclei_col_types[prop] == cfg.PD_STRUCT_TYPE_2D:
                    # 2D
                    z_list = list()

                    # initialise z
                    prev_z = -1

                    for kID, value in data_nID['z'].iteritems():
                        curr_z = int(value)

                        # append to previous z?
                        if prev_z != curr_z:
                            # create new entry for z and values
                            z_list.append((curr_z, list()))

                        # remember z
                        prev_z = curr_z

                    # go through z and init the respective values
                    for zID, z_tuple in enumerate(z_list):
                        # get data frame for z
                        data_z = data_nID[data_nID['z'].isin([z_tuple[0]])]

                        # go through cols
                        for id_col, data_col in data_z.iteritems():
                            if id_col != 'z':
                                counter = 0

                                # go through rows
                                for kID, value in data_col.iteritems():
                                    if counter < len(z_list[zID][1]):
                                        # add to existing values
                                        z_list[zID][1][counter] += (value,)
                                    else:
                                        # add to values
                                        z_list[zID][1].append((value,))

                                    counter += 1

                    key = prop.split('_')[-1]
                    nucleus[key] = z_list

        return nucleus

    def build_nuclei_from_dataframe(self, data_struct):
        """
        Build nuclei from dataframe

        :param data_struct:
        :return:
        """
        nuclei = list()

        # get all IDs and build nuclei
        for nID in data_struct.data_params.index.values:
            print('load %i' % nID)
            nuclei.append(self.build_nucleus_from_dataframe(data_struct, nID))

        return nuclei

    def build_nucleus_from_dataframe(self, data_struct, nID):
        """
        Build a nucleus object from a given dataframe

        :param data_struct:
        :param nID:
        :return:
        """
        nucleus = dict()

        nucleus['nID'] = nID

        # set label values
        nucleus = self.get_label_props_from_array_lists(nucleus, data_struct)

        # build image information

        # projections
        #for key in data_struct.img_projection.columns.values:
        #    nucleus['projection_%s' % key] = data_struct.img_projection[key].loc[nID]

        # preview image
        #nucleus['img'] = self.extra['img_projections']['y'][nID]

        # create boxes for nucleus
        nucleus = self.build_nucleus_boxes(nucleus)

        return nucleus

    def get_nIDs(self, only_accepted=False):
        """
        Get IDs from nuclei

        :return:
        """

        nIDs = self.data_frames['data_params'].get_nIDs(only_accepted=only_accepted)

        # filter nIDs by acceptance status
        #if only_accepted == True:
        #    # FIX: only every 2nd element is tested
        #    # re-typing it as a list helped
        #    for nID in list(nIDs):
        #        if self.is_nucleus_rejected(nID) is True:
        #            nIDs.remove(nID)

        return nIDs

    def calc_nuclei_params(self, only_accepted=False, limit_to=-1, selected_nIDs=None):
        """
        Calculate params for nuclei

        :param only_accepted:
        :param limit_to:
        :return:
        """
        # threading parameters
        max_threads = cfg.merge_threads_nuc_params

        # filter for selected nIDs
        if selected_nIDs is not None:
            nIDs = selected_nIDs
        else:
            nIDs = self.get_nIDs(only_accepted=only_accepted)

        # limit nIDs?
        if limit_to > 0:
            nIDs = nIDs[0:limit_to]

        nIDit = iter(nIDs)

        print('Calc nucleus params for %i' % len(nIDs))

        # check if there is a membrane segmentation
        if self.segmentation.has_membin() is False:
            self.segmentation.create_membin()

        hasnext = True

        while hasnext is True:
            nuc_threads = []

            # start threads
            for i in range(0, max_threads):
                try:
                    nID = next(nIDit)

                    if nID % 10 == 0:
                        print('\t%i' % nID)

                    thread = threading.Thread(target=self.calc_nucleus_params, args=(nID, ))
                    thread.start()
                    nuc_threads.append(thread)
                except StopIteration:
                    hasnext = False

            # wait to finish
            for thread in nuc_threads:
                thread.join()

    def add_nucleus_param(self, nID, param, value):
        """
        Add parameter to nucleus

        :param nID:
        :param param:
        :param value:
        :return:
        """
        # get nucleus
        nucleus = self.get_nucleus_by_id(nID)

        # add param to nucleus or update
        nucleus[param] = value

        # store nucleus again
        self.nuclei[self.get_listID(nID)] = nucleus

    def add_nucleus_to_list(self, nucleus):
        """
        Add nucleus to list

        :param nucleus:
        :return:
        """
        if self.is_nucleus_in_nuclei(nucleus['nID']) is False:
            self.nuclei.append(nucleus)

        # rebuild lookup tables
        self.lookup.rebuild_tables(self.nuclei)

    def del_nucleus_from_list(self, nucleus):
        """
        Del nucleus from list

        :param nucleus:
        :return:
        """
        if self.is_nucleus_in_nuclei(nucleus['nID']) is False:
            self.nuclei.pop(self.get_listID_by_nID(nucleus['nID']))

        # rebuild lookup tables
        self.lookup.rebuild_tables(self.nuclei)

    def is_param_in_nuclei(self, param):
        """
        Is param part of the nuclei?

        :param param:
        :return:
        """
        is_in_nuclei = False

        # take a nucleus and see if this is part of the nucleus dict
        nucleus_dict = self.get_nucleus_param_dict(0)

        if param in nucleus_dict:
            is_in_nuclei = True

        return is_in_nuclei

    def calc_nucleus_params(self, nID):
        """
        Calculate parameters for nucleus

        :param nID:
        :return:
        """
        # volume
        nucleus_volume = self.get_nucleus_areas(nID)[:, 1].sum()

        self.set_nucleus_volume(nucleus_volume, nID)

        # surface
        # get top and bottom from area
        nucleus_area = self.get_nucleus_areas(nID)
        nucleus_top = nucleus_area[0, 1]
        nucleus_btm = nucleus_area[-1, 1]

        # get sum from perimeter
        nucleus_perimeter = self.get_nucleus_perimeters(nID)

        # calculate surface
        nucleus_surface = nucleus_top + nucleus_btm + nucleus_perimeter[1:-1, 1].sum()

        self.set_nucleus_surface(nucleus_surface, nID)
        self.set_nucleus_surface_volume_ratio((nucleus_surface/nucleus_volume), nID)

        # calculate nucleus depth based on plane by plane basis
        nucleus_depth = 0

        # go through centroids and calculate distance between them in 3D space
        # with simple pythagoras
        nucleus_centroids = self.get_nucleus_centroids(nID)

        prev_z = -1
        prev_y = -1
        prev_x = -1

        for centroid in nucleus_centroids:
            # set current information
            cur_z = centroid[0]
            cur_y = centroid[1]
            cur_x = centroid[2]

            # do the calculation
            if prev_z >= 0 and prev_y >= 0 and prev_x >= 0:
                nucleus_depth += self.calc_3D_distance(np.array([cur_z, cur_y, cur_x]),
                                                       np.array([prev_z, prev_y, prev_x]))

            # set as previous infor for next iteration
            prev_z = cur_z
            prev_y = cur_y
            prev_x = cur_x

        self.set_nucleus_depth(nucleus_depth, nID)
        self.set_nucleus_volume_depth_ratio((nucleus_volume/nucleus_depth), nID)

        # create box
        self.build_nucleus_boxes(nID)

        # create projections
        self.create_projections(nID)

        # draw nucleus on stack
        nucleus_stack = self.add_nucleus_to_stack(nID, np.zeros_like(self.segmentation.stacks.lamin), 1)

        # calculate DAPI and membrane intensity
        lamin_signal = self.segmentation.stacks.lamin * nucleus_stack
        dapi_signal = self.segmentation.stacks.dapi * nucleus_stack
        membrane_signal = self.segmentation.stacks.membrane * nucleus_stack

        # get DAPI and membrane image
        #dapi_box = Plot.get_nucleus_box(nucleus, dapi_signal, cfg.criteria_select_nuc_box_offset)
        #membrane_box = Plot.get_nucleus_box(nucleus, membrane_signal, cfg.criteria_select_nuc_box_offset)

        lamin_mean = np.mean(lamin_signal.ravel()[np.flatnonzero(lamin_signal)])
        dapi_mean = np.mean(dapi_signal.ravel()[np.flatnonzero(dapi_signal)])
        membrane_mean = np.mean(membrane_signal.ravel()[np.flatnonzero(membrane_signal)])

        #nucleus['img'] = dapi_box[:, round(membrane_box.shape[1]/2), :]

        self.set_nucleus_lamin_int(lamin_mean, nID)
        self.set_nucleus_dapi_int(dapi_mean, nID)
        self.set_nucleus_membrane_int(membrane_mean, nID)

        # centre of nucleus by taking the average of centroids
        # take average
        nucleus_centre = np.array([
            nucleus_centroids[int(len(nucleus_centroids)/2)][0],
            int(sum(nucleus_centroids[:, 1]) / len(nucleus_centroids[:, 1])),
            int(sum(nucleus_centroids[:, 2]) / len(nucleus_centroids[:, 2]))
        ])

        self.set_nucleus_centre(nucleus_centre, nID)

        # calculate distance to image borders
        edge_distances = [
            nucleus_centre[0], nucleus_stack.shape[0] - nucleus_centre[0],
            nucleus_centre[1], nucleus_stack.shape[1] - nucleus_centre[1],
            nucleus_centre[2], nucleus_stack.shape[2] - nucleus_centre[2]
        ]

        self.set_classifier_edge_dist(min(edge_distances), nID)

        # take ratio of average area of top to bottom
        area_total = list()
        for area in self.get_nucleus_areas(nID):
            area_total.append(area[1])

        area_top = area_total[0:int(len(area_total)/2)]
        area_bottom = area_total[int(len(area_total)/2):len(area_total)]
        area_ratio = float(sum(area_top)/sum(area_bottom))

        self.set_classifier_topbot_ratio(area_ratio, nID)
        self.set_classifier_depth_ratio(float(sum(area_total)/self.get_nucleus_depth(nID)), nID)

        # TODO calculate donut for nucleus
        # donut_props = self.calc_donuts_for_nucleus(nID, self.segmentation.stacks)

        # set donut props for nucleus
        # for key in donut_props.keys():
        #     nucleus[key] = donut_props[key]

        # calc nucleus box
        nucleus_coords = self.get_nucleus_coords(nID)
        nucleus_bboxes = self.get_nucleus_bboxes(nID)
        nucleus_bbox = np.array([
            nucleus_coords[0, 0],
            min(nucleus_bboxes[:, 1]),
            min(nucleus_bboxes[:, 2]),
            nucleus_coords[-1, 0],
            max(nucleus_bboxes[:, 3]),
            max(nucleus_bboxes[:, 4])
        ])

        self.set_nucleus_bbox(nucleus_bbox, nID)

        # get neighbouring nuclei
        neighbour_nIDs = self.get_nID_by_pos_range(
            self.get_expanded_bbox_for_nucleus(nID, only_horizontal=True), only_accepted=True)

        # count neighbouring nuclei - minus the nucleus itself
        self.set_nucleus_neighbours((len(neighbour_nIDs) - 1), nID)

        # calculate average distance to neighbouring nuclei
        neighbours_distances = list()
        for neighbour_nID in neighbour_nIDs:
            if neighbour_nID != nID:
                # calc distance
                neighbour_centre = self.get_nucleus_centre(neighbour_nID)

                neighbours_distances.append(
                    self.calc_3D_distance(nucleus_centre, neighbour_centre)
                )

        avg_distance = sum(neighbours_distances)/len(neighbours_distances)

        self.set_nucleus_neighbours_distance(avg_distance, nID)

        # calculate direction vector of nucleus
        vec_top = nucleus_centroids[0]
        vec_bottom = nucleus_centroids[-1]

        vec_direction = (
            vec_top[0] - vec_bottom[0],
            vec_top[1] - vec_bottom[1],
            vec_top[2] - vec_bottom[2]
        )

        # adjust vector to z and prepare for saving
        vec_direction = np.array([
            vec_direction[0]/vec_direction[0],
            vec_direction[1]/vec_direction[0],
            vec_direction[2]/vec_direction[0]
        ])

        self.set_nucleus_direction(vec_direction, nID)

        # calculate apical distance
        # go from nucleus centre up and check if there is still a membrane signal
        is_in_membrane_dim = True
        cur_rel_z = 0
        next_pos = nucleus_centre
        last_pos_on_membrane = None

        # for counting nuclei_in_direction
        nuclei_in_direction_list = list()

        while is_in_membrane_dim is True:
            # is point on membrane signal?
            if self.segmentation.stacks.membin[next_pos[0], next_pos[1], next_pos[2]] > 0:
                last_pos_on_membrane = next_pos

            # get the nucleus at this position
            nID_at_pos = self.get_nID_by_pos(next_pos)
            if nID_at_pos is not None:
                nuclei_in_direction_list.append(nID_at_pos)

            # calculate next position
            cur_rel_z += 1

            next_pos = (
                int(nucleus_centre[0] - (cur_rel_z * vec_direction[0])),
                int(nucleus_centre[1] - (cur_rel_z * vec_direction[1])),
                int(nucleus_centre[2] - (cur_rel_z * vec_direction[2]))
            )

            # test if next point is in membrane dimensions
            for i in range(3):
                if next_pos[i] < 0 or next_pos[i] >= self.segmentation.stacks.membin.shape[i]:
                    is_in_membrane_dim = False

        # calculate distance to last position and nucleus centre
        apical_distance = self.calc_3D_distance(nucleus_centre, last_pos_on_membrane)

        self.set_nucleus_apical_distance(apical_distance, nID)

        # delete yourself and get unique list of nuclei in direction and count
        nuclei_in_direction_list.remove(nID)
        nuclei_in_direction_count = len(set(nuclei_in_direction_list))

        self.set_nucleus_nuclei_in_direction(nuclei_in_direction_count, nID)

        # calculate ellipsoid for nucleus
        # rotate direction vector 90DEG around X
        vec_direction_orth = self.segmentation.rot_vector(vec_direction, 90, 2)

        # rotate orthogonal vector 180DEG around Z and get the last
        # points on the nucleus and calculate the distance between them
        diameters = list()
        dia_points = list()

        # add nucleus to stack
        #nucleus_on_stack = self.add_nucleus_to_stack(nID, np.zeros_like(self.segmentation.stacks.lamin),
        #                                             nucleus_value=1)

        for cur_rot in range(cfg.nucleus_calc_elps_rot, 181, cfg.nucleus_calc_elps_rot):
            # roate vector
            vec_direction_orth_rot = self.segmentation.rot_vector(vec_direction_orth, cur_rot, 0)

            # get last point in +/- direction
            cur_dist = 0
            last_plus_pos = None
            last_minus_pos = None

            next_pos = nucleus_centre

            # is the position on the nucleus?
            while self.get_nID_by_pos(next_pos) == nID:
            #while nucleus_on_stack[next_pos[0], next_pos[1], next_pos[2]] > 0:
                last_plus_pos = next_pos

                # calculate next position
                cur_dist += 1

                next_pos = (
                    int(nucleus_centre[0] + (cur_dist * vec_direction_orth_rot[0])),
                    int(nucleus_centre[1] + (cur_dist * vec_direction_orth_rot[1])),
                    int(nucleus_centre[2] + (cur_dist * vec_direction_orth_rot[2]))
                )

            cur_dist = 0
            next_pos = nucleus_centre

            # is the position on the nucleus?
            while self.get_nID_by_pos(next_pos) == nID:
                last_minus_pos = next_pos

                # calculate next position
                cur_dist += 1

                next_pos = (
                    int(nucleus_centre[0] - (cur_dist * vec_direction_orth_rot[0])),
                    int(nucleus_centre[1] - (cur_dist * vec_direction_orth_rot[1])),
                    int(nucleus_centre[2] - (cur_dist * vec_direction_orth_rot[2]))
                )

            # calculate distance
            if last_plus_pos is not None and last_minus_pos is not None:
                diameter = self.calc_3D_distance(last_plus_pos, last_minus_pos)

                diameters.append(diameter)
                dia_points.append((last_plus_pos, last_minus_pos))

        # get min and max from radii
        min_diameter = min(diameters)
        max_diameter = max(diameters)

        min_index = diameters.index(min_diameter)
        max_index = diameters.index(max_diameter)

        # get vectors
        if dia_points[min_index][0][1] < dia_points[min_index][1][1]:
            vec_min = np.array([
                (dia_points[min_index][1][0] - dia_points[min_index][0][0]),
                (dia_points[min_index][1][1] - dia_points[min_index][0][1]),
                (dia_points[min_index][1][2] - dia_points[min_index][0][2])
            ])
        else:
            vec_min = np.array([
                (dia_points[min_index][0][0] - dia_points[min_index][1][0]),
                (dia_points[min_index][0][1] - dia_points[min_index][1][1]),
                (dia_points[min_index][0][2] - dia_points[min_index][1][2])
            ])

        if dia_points[max_index][0][1] < dia_points[max_index][1][1]:
            vec_max = np.array([
                (dia_points[max_index][1][0] - dia_points[max_index][0][0]),
                (dia_points[max_index][1][1] - dia_points[max_index][0][1]),
                (dia_points[max_index][1][2] - dia_points[max_index][0][2])
            ])
        else:
            vec_max = np.array([
                (dia_points[max_index][0][0] - dia_points[max_index][1][0]),
                (dia_points[max_index][0][1] - dia_points[max_index][1][1]),
                (dia_points[max_index][0][2] - dia_points[max_index][1][2])
            ])

        # calculate volumes of min and max ellipsoid
        elps_vol = (4/3) * math.pi * (nucleus_depth/2) * (min_diameter/2) * (max_diameter/2)

        # set major and minor axis
        self.set_nucleus_major_axis(max_diameter, nID)
        self.set_nucleus_minor_axis(min_diameter, nID)
        self.set_nucleus_mami_axis((max_diameter/min_diameter), nID)

        # get orientations and add 90 to have values from 0 - 180
        # the orientation function returns values from -90 to 90
        direction_orientation = self.calc_orientation_of_vector(vec_direction) + 90
        min_orientation = self.calc_orientation_of_vector(vec_min) + 90
        max_orientation = self.calc_orientation_of_vector(vec_max) + 90

        self.set_nucleus_direction_orientation(direction_orientation, nID)
        self.set_nucleus_minor_axis_orientation(min_orientation, nID)
        self.set_nucleus_major_axis_orientation(max_orientation, nID)

    def calc_orientation_of_vector(self, array_vector, project_to=0):
        """
        Calculate orientation of array vector

        :param project_to:
        :return:
        """

        angle = None

        if project_to == 0:
            angle = math.degrees(math.atan(array_vector[2]/array_vector[1]))
        elif project_to == 1:
            angle = math.degrees(math.atan(array_vector[2]/array_vector[0]))
        elif project_to == 2:
            angle = math.degrees(math.atan(array_vector[1]/array_vector[0]))

        return angle

    def calc_3D_distance(self, p1, p2):
        """
        Calculate distance in 3D between two points

        :param p1:
        :param p2:
        :return:
        """
        a = abs(int(p1[0]) - int(p2[0]))
        b = abs(int(p1[1]) - int(p2[1]))
        c = abs(int(p1[2]) - int(p2[2]))

        return math.sqrt(a**2 + (math.sqrt(b**2 + c**2))**2)

    def create_projections(self, nID):
        """
        Create projections for nID and store in extra dict

        :param nID:
        :return:
        """
        equalise_filter = Equalise()

        self.extra['img_projections']['z'][nID] = np.zeros_like(self.img_boxes['bw'][nID][0, :, :])
        for z in range(0, self.img_boxes['bw'][nID].shape[0]):
            self.extra['img_projections']['z'][nID] += self.img_boxes['bw'][nID][z, :, :]

        self.extra['img_projections']['z'][nID] = equalise_filter.apply(self.extra['img_projections']['z'][nID])

        self.extra['img_projections']['y'][nID] = np.zeros_like(self.img_boxes['bw'][nID][:, 0, :])
        for y in range(0, self.img_boxes['bw'][nID].shape[1]):
            self.extra['img_projections']['y'][nID] += self.img_boxes['bw'][nID][:, y, :]

        self.extra['img_projections']['y'][nID] = equalise_filter.apply(self.extra['img_projections']['y'][nID])

        self.extra['img_projections']['x'][nID] = np.zeros_like(self.img_boxes['bw'][nID][:, :, 0])
        for x in range(0, self.img_boxes['bw'][nID].shape[2]):
            self.extra['img_projections']['x'][nID] += self.img_boxes['bw'][nID][:, :, x]

        self.extra['img_projections']['x'][nID] = equalise_filter.apply(self.extra['img_projections']['x'][nID])

        self.extra['infos']['img'][nID] = self.extra['img_projections']['y'][nID]
        self.extra['infos']['lamin_slice'][nID] = \
            self.img_boxes['lamin'][nID][:,round(self.img_boxes['lamin'][nID].shape[1] / 2), :]

    def build_bw_nuclei(self):
        """
        Build black/white version of nuclei

        :param nuclei:
        :return:
        """
        bw_nuclei = np.zeros_like(self.segmentation.stacks.nuclei)
        bw_nuclei[self.segmentation.stacks.nuclei > 0] = 1

        return bw_nuclei

    def build_nucleus_boxes(self, nID):
        """
        Build nucleus boxes for display

        :param nID:
        :return:
        """
        bw_nuclei = self.build_bw_nuclei()

        # get required nucleus params
        nucleus_centroids = self.get_nucleus_centroids(nID)
        nucleus_areas = self.get_nucleus_areas(nID)

        bw_stack = self.add_nucleus_to_stack(nID, np.zeros_like(bw_nuclei), 1)
        self.img_boxes['bw'][nID] = Plot.get_nucleus_box(nucleus_centroids,
                                                         nucleus_areas,
                                                         bw_stack,
                                                         cfg.criteria_select_nuc_box_offset)
        self.img_boxes['lamin'][nID] = Plot.get_nucleus_box(nucleus_centroids,
                                                            nucleus_areas,
                                                            self.segmentation.stacks.lamin,
                                                            cfg.criteria_select_nuc_box_offset)
        self.img_boxes['labels'][nID] = Plot.get_nucleus_box(nucleus_centroids,
                                                             nucleus_areas,
                                                             self.segmentation.stacks.labels,
                                                             cfg.criteria_select_nuc_box_offset)
        self.img_boxes['membrane'][nID] = Plot.get_nucleus_box(nucleus_centroids,
                                                               nucleus_areas,
                                                               self.segmentation.stacks.membrane,
                                                               cfg.criteria_select_nuc_box_offset)
        self.img_boxes['dapi'][nID] = Plot.get_nucleus_box(nucleus_centroids,
                                                           nucleus_areas,
                                                           self.segmentation.stacks.dapi,
                                                           cfg.criteria_select_nuc_box_offset)

        # create cropped boxes for signals
        self.img_boxes['crop_lamin'][nID] = self.img_boxes['lamin'][nID] * self.img_boxes['bw'][nID]
        self.img_boxes['crop_membrane'][nID] = self.img_boxes['membrane'][nID] * self.img_boxes['bw'][nID]
        self.img_boxes['crop_dapi'][nID] = self.img_boxes['dapi'][nID] * self.img_boxes['bw'][nID]

        # create RGB version
        self.img_boxes['crop_rgb'][nID] = np.zeros(shape=(self.img_boxes['crop_lamin'][nID].shape[0],
                                                          self.img_boxes['crop_lamin'][nID].shape[1],
                                                          self.img_boxes['crop_lamin'][nID].shape[2], 4))
        self.img_boxes['crop_rgb'][nID][:, :, :, 0] = self.img_boxes['crop_membrane'][nID].astype(float) / 255
        self.img_boxes['crop_rgb'][nID][:, :, :, 1] = self.img_boxes['crop_lamin'][nID].astype(float) / 255
        self.img_boxes['crop_rgb'][nID][:, :, :, 2] = self.img_boxes['crop_dapi'][nID].astype(float) / 255
        self.img_boxes['crop_rgb'][nID][:, :, :, 3] = 0.0

        # set alpha '1' if one channel is over one
        for chn in range(0, self.img_boxes['crop_rgb'][nID].shape[3]):
            self.img_boxes['crop_rgb'][nID][:, :, :, 3][self.img_boxes['crop_rgb'][nID][:, :, :, chn] > 0] = 1.0

    def add_nuclei_to_stack(self, stack, nucleus_value=None, nIDs=None, only_accepted=False):
        """
        Add nuclei to a stack based on its coordinates

        :param nucleus:
        :param stack:
        :return:
        """
        print('Add nuclei to stack')

        # go through nuclei
        if nIDs is None:
            print('\tget nIDs')
            nIDs = self.get_nIDs(only_accepted=only_accepted)

        # get coords
        print('\tget coords')
        nuc_coords = self.get_nucleus_coords(nIDs)

        if nuc_coords is not None:
            # adjust for nan values
            nuc_coords = nuc_coords[nuc_coords[:, 0] >= 0]

            # go through nuclei and add
            prog_bar = int(len(nuc_coords) / 4)
            for i, coords in enumerate(nuc_coords):
                if i % prog_bar == 0:
                    print('\t%i' % i)
                stack[int(coords[0])][int(coords[1]), int(coords[2])] = coords[3]

        return stack

    def add_nucleus_to_stack(self, nID, stack, nucleus_value=None):
        """
        Add a single nucleus to a stack

        :param nucleus:
        :param stack:
        :return:
        """
        if self.is_nucleus_in_nuclei(nID):
            # colour choice
            if nucleus_value is None:
                nucleus_value = self.get_nucleus_colour(nID)
            elif nucleus_value < 0:
                nucleus_value = rdm.randrange(0, 255)

            # iterate through the coordinates
            nuc_coords = self.get_nucleus_coords(nID)
            for coords in nuc_coords:
                stack[int(coords[0])][int(coords[1]), int(coords[2])] = nucleus_value

        return stack

    def calc_donuts_for_nucleus(self, nID, stacks):
        """
        Calculate donuts for all labels of the nucleus
        and return the mean intensities

        :param nID:
        :param stacks:
        :return:
        """
        # to store mean intensities of donut
        donut_means = dict()

        # go through each label of the nucleus and draw a donut
        for z, coords in enumerate(nID['coords']):
            # build label
            label = self.build_label_from_nucleus(nID, z)[0]

            # get images
            imgs = Image()
            imgs.lamin = stacks.lamin[coords[0]]
            imgs.dapi = stacks.dapi[coords[0]]
            imgs.membrane = stacks.membrane[coords[0]]

            # calculate donut
            calc_props = self.segmentation.calc_donut_for_label(label, imgs,
                                                           dilero_param=cfg.merge_lamin_donut_ring)

            # init means
            if z == 0:
                for key in calc_props.keys():
                    donut_means[key] = list()

            # append to list
            for key in donut_means.keys():
                donut_means[key].append(calc_props[key])

        # calculate mean of the means
        donut_props = dict()
        for key in donut_means.keys():
            donut_props[key] = np.mean(donut_means[key])

        return donut_props

    def calc_donut_for_label(self, label_props, imgs, dilero_param=1):
        """
        Calculate donut for label

        :param label_props
        :param stacks:
        :param dilero_param:
        :return:
        """

        # to store return props
        donut_props = {
            'donut_lamin': 0.00,
            'donut_dapi': 0.00,
            'donut_membrane': 0.00,
            'donut_ratio': 0.00
        }

        # apply filter to extract lamin signal per plane
        donut_dilate = list()
        donut_dilate.append(('DIL', 'y', dilero_param))

        donut_erode = list()
        donut_erode.append(('ERO', 'y', dilero_param))

        # draw label on image
        label_img = Segmentation.add_label_to_img(label_props, np.zeros_like(imgs.lamin), 1)

        # create donut core
        donut_core_img = ImageProcessing.apply_filters(donut_erode, label_img).astype(int)

        # create donut ring
        donut_ring_img = ImageProcessing.apply_filters(donut_dilate, label_img).astype(int)
        donut_ring_img -= label_img

        # get properties for donut
        donut_core_raw_props = regionprops(donut_core_img)
        donut_ring_raw_props = regionprops(donut_ring_img)

        if len(donut_core_raw_props) > 0:
            donut_core_props_coords = donut_core_raw_props[0].coords
            donut_ring_props_coords = donut_ring_raw_props[0].coords

            # get intensities for donut core
            donut_core_dapi = np.zeros(len(donut_core_props_coords))
            donut_core_membrane = np.zeros(len(donut_core_props_coords))

            for i, coord in enumerate(donut_core_props_coords):
                donut_core_dapi[i] = imgs.dapi[coord[0]][coord[1]]
                donut_core_membrane[i] = imgs.membrane[coord[0]][coord[1]]

            # get lamin intensity for donut ring
            donut_ring_lamin = np.zeros(len(donut_ring_props_coords))

            for i, coord in enumerate(donut_ring_props_coords):
                donut_ring_lamin[i] = imgs.lamin[coord[0]][coord[1]]

            # calculate mean intensities
            donut_core_dapi_mean = np.mean(donut_core_dapi)
            donut_core_membrane_mean = np.mean(donut_core_membrane)
            donut_ring_lamin_mean = np.mean(donut_ring_lamin)

            if donut_core_membrane_mean > 0:
                lamin_donut_ratio = round(
                    (donut_ring_lamin_mean + donut_core_dapi_mean)/donut_core_membrane_mean, 2)
            else:
                lamin_donut_ratio = 0

            # store mean intensities
            donut_props = {
                'donut_lamin': donut_ring_lamin_mean,
                'donut_dapi': donut_core_dapi_mean,
                'donut_membrane': donut_core_membrane_mean,
                'donut_ratio': lamin_donut_ratio
            }

        return donut_props

    def get_nID_by_pos(self, pos, only_accepted=True):
        """
        Return nucleus which have the position

        :param pos:
        :return:
        """
        nID = None

        # look through coordinates and get the index
        vals = self.data_frames['data_coords'].data_frame[
            (self.data_frames['data_coords'].data_frame.z == pos[0])
            & (self.data_frames['data_coords'].data_frame.y == pos[1])
            & (self.data_frames['data_coords'].data_frame.x == pos[2])
        ].index.values

        if len(vals) > 0:
            nID = vals[0]

        # check for rejection
        if nID is not None and only_accepted is True:
            if self.segmentation.nuclei.is_nucleus_rejected(nID):
                nID = None

        return nID

    def get_nID_by_pos_range(self, pos_range, only_accepted=True):
        """
        Return nuclei which are in the position range

        :param pos_range:
        :return:
        """
        # look through coordinates and get the index
        nIDs = list(self.data_frames['data_coords'].data_frame[
            (self.data_frames['data_coords'].data_frame.z >= pos_range[0])
            & (self.data_frames['data_coords'].data_frame.z <= pos_range[3])
            & (self.data_frames['data_coords'].data_frame.y >= pos_range[1])
            & (self.data_frames['data_coords'].data_frame.y <= pos_range[4])
            & (self.data_frames['data_coords'].data_frame.x >= pos_range[2])
            & (self.data_frames['data_coords'].data_frame.x <= pos_range[5])
        ].index.values)

        unique_nIDs = list(np.unique(nIDs))
        ret_nIDs = list()

        # check for rejected nuclei
        if only_accepted is True:
            for nID in unique_nIDs:
                if self.segmentation.nuclei.is_nucleus_rejected(nID) is False:
                    ret_nIDs.append(nID)
        else:
            ret_nIDs = unique_nIDs

        return ret_nIDs

    def get_overlapping_nuclei(self, nucleus):
        """
        Does the nucleus has an overlap?

        :param pos:
        :return:
        """
        # to store overlapping nuclei
        overlapping_nuclei = list()

        # get only nuclei in the vicinity of the bbox
        #nuclei = self.get_nuclei_by_pos_range(self.get_expanded_bbox_for_nucleus(nucleus))

        # lookup nuclei in vicinity of the centroid +/- value
        for i, centroid in enumerate(nucleus['centroid']):
            nuclei_in_range = list()

            # define range for lookup
            pos_range = [
                centroid[0], centroid[1][0] - 3, centroid[1][1] - 3,
                centroid[0], centroid[1][0] + 3, centroid[1][1] + 3,
            ]

            nuclei_in_range = self.get_nuclei_by_pos_range(pos_range)

            overlapping_nuclei += nuclei_in_range

        # go through nuclei
        #for coords in nucleus['coords']:
        #    for coord in coords[1]:
        #        # create pos
        #        pos = (coords[0], coord[0], coord[1])
        #
        #        # look up overlapping nucleus
        #        overlapping_nucleus = self.get_nucleus_by_pos_in_nuclei(pos, self.lookup)
        #
        #        # add nucleus to list
        #        if overlapping_nucleus is not None:
        #            # is nucleus already in list?
        #            is_in_list = False
        #
        #            for cur_nucleus in overlapping_nuclei:
        #               if cur_nucleus['nID'] == overlapping_nucleus['nID']:
        #                    is_in_list = True
        #
        #            if is_in_list is False:
        #                overlapping_nuclei.append(overlapping_nucleus)

        return overlapping_nuclei

    def get_expanded_bbox_for_nucleus(self, nID, only_horizontal=False):
        """
        Return expanded bbox for nucleus

        :param nID:
        :param only_horizontal:
        :return:
        """
        nuc_bbox = self.get_nucleus_bbox(nID)

        # add offset
        offset = cfg.nuclei_bbox_range

        if only_horizontal is True:
            start_dim = 1
        else:
            start_dim = 0

        for i in range(start_dim, 3):
            nuc_bbox[i] -= offset

            if nuc_bbox[i] < 0:
                nuc_bbox[i] = 0

        for i in range(start_dim + 3, 6):
            nuc_bbox[i] += offset

            if nuc_bbox[i] >= self.segmentation.stacks.nuclei.shape[i - 3]:
                nuc_bbox[i] = self.segmentation.stacks.nuclei.shape[i - 3]

        return nuc_bbox

    def get_raw_nucleus_by_pos(self, pos):
        """
        Return nucleus which have the position

        :param pos:
        :return:
        """
        return self.get_nucleus_by_pos_in_nuclei(pos, self.get_raw_lookup())

    def get_raw_nuclei_by_pos_range(self, pos):
        """
        Return nuclei which are in the position range

        :param pos:
        :return:
        """
        return self.get_nuclei_by_pos_range_in_nuclei(pos, self.get_raw_lookup())

    def is_nucleus_in_nuclei(self, nID):
        """
        Look if the nucleus is in the nuclei list

        :param nucleus:
        :return:
        """
        in_list = False

        if self.data_frames['data_params'].is_nID_in_data_frame(nID):
            in_list = True

        return in_list

    def set_nucleus_by_id(self, nID, nucleus):
        """
        Cycle through nuclei and return the nucleus with a specific ID

        :param nID:
        :param nucleus:
        :return:
        """
        for id, cur_nucleus in enumerate(self.nuclei):
            if cur_nucleus['nID'] == nID:
                self.nuclei[id] = nucleus
                break

        return nucleus

    def remove_nucleus(self, nID):
        """
        Remove nucleus

        :param nID:
        :return:
        """
        # go through nuclei list and delete the respective nucleus
        for id, cur_nucleus in enumerate(self.nuclei):
            if cur_nucleus['nID'] == nID:
                self.nuclei.pop(id)
                break

    def save(self, path, projections_only=False):
        """
        Save nuclei data

        :param path:
        :return:
        """
        if projections_only is False:
            # save all dataframes as csv
            for key, lookup_frame in self.data_frames.items():
                lookup_frame.save_as_csv(path + 'df_%s.csv' % key)

        # pickle image information and general infos for nuclei
        with open(path + 'nuclei_extras.dat', "wb") as fin:
            pickle.dump(self.extra, fin)

        # TODO image boxes are too big
        #with open(path + 'nuclei_img_boxes.dat', "wb") as fin:
        #    pickle.dump(self.img_boxes, fin)

    def load(self, path, force_extra_recalc=False):
        """
        load nuclei data

        :param path:
        :return:
        """
        # mark if nuclei params were loaded
        nuclei_params_loaded = False

        # load all dataframes from csv
        for key, lookup_frame in self.data_frames.items():
            csv_file = path + 'df_%s.csv' % key

            if os.path.isfile(csv_file):
                lookup_frame.load_from_csv(csv_file)

                nuclei_params_loaded = True

        extra_file = path + 'nuclei_extras.dat'
        img_boxes_file = path + 'nuclei_img_boxes.dat'

        if force_extra_recalc is False and os.path.isfile(extra_file) and os.stat(extra_file).st_size > 0:
            # load pickled image information and general infos for nuclei
            with open(extra_file, "rb") as fin:
                self.extra = pickle.load(fin)

            # load img boxes
            if os.path.isfile(img_boxes_file) and os.stat(img_boxes_file).st_size > 0:
                with open(img_boxes_file, "rb") as fin:
                    self.img_boxes = pickle.load(fin)
        elif force_extra_recalc is True or nuclei_params_loaded is True:
            print('Recalculate extras')

            # create nuclei stack
            #self.segmentation.update(save=True, calc_nuclei_params=False)

            # calculate the extras as all the params are loaded
            for i, nID in enumerate(self.get_nIDs()):
                if (i % 100) == 0:
                    print('\t%i' % i)

                # calculate extras
                self.build_nucleus_boxes(nID)
                self.create_projections(nID)

            # get dirs
            dirs = self.segmentation.get_results_dir()

            print('save extras')
            # save projections
            self.save(dirs.nuclei_params_raw, projections_only=True)

    def sort_nuclei(self, col='nID', asc=True, inplace=True):
        """
        Sort nuclei by param

        :param col:
        :param asc:
        :param inplace:
        :return:
        """
        return self.data_frames['data_params'].sort_by_col(col, asc=asc, inplace=inplace)

    def get_nID_from_sID(self, sID, only_accepted=False):
        """
        Get nID from sequence ID

        :param sID:
        :return:
        """
        nIDs = self.get_nIDs(only_accepted=only_accepted)
        nID = None

        if sID >= 0 and sID < len(nIDs):
            nID = nIDs[sID]

        return nID

    def get_sID_from_nID(self, nID, only_accepted=False):
        """
        Get sequence ID from nID

        :param nID:
        :return:
        """
        nIDs = self.get_nIDs(only_accepted=only_accepted)
        sID = None

        if nID in nIDs:
            sID = nIDs.index(nID)

        return sID

    """
    Getters

    The data is handled as in a database
    """

    def get_param_list_from_nuclei(self, col=None, data_frame='data_params', only_accepted=False,
                                   create_dict=False, sort_by=None):
        """
        Get param from nucleus

        :param col:
        :param data_frame:
        :return:
        """
        # get values
        if col is None:
            # add nID to frame
            self.data_frames[data_frame].add_col_from_index()

            vals = self.data_frames[data_frame].get_vals()

            # create dict with column names
            if create_dict is True:
                val_list = list()

                # go through values and map into dictionary
                for x in range(0, vals.shape[0]):
                    row = vals[x]

                    # add dict
                    val_list.append(dict())

                    # add params
                    for y, col in enumerate(self.data_frames[data_frame].data_frame.columns):
                        val_list[-1][col] = row[y]

                    # add img
                    val_list[-1]['img'] = self.get_extra_infos(val_list[-1]['nID'], 'img')

            vals = val_list

            # del nID from frame
            self.data_frames[data_frame].del_col_from_index()
        else:
            if sort_by is not None:
                sorted_frame = self.data_frames[data_frame].sort_by_col(col)
                vals = self.data_frames[data_frame].get_vals_from_col(col, only_accepted=only_accepted,
                                                                      data_frame=sorted_frame)

            else:
                vals = self.data_frames[data_frame].get_vals_from_col(col, only_accepted=only_accepted)

        return vals

    def get_param_from_nucleus(self, data_frame, col, nID, multi_rows=False, z=-1,
                               join_params=None):
        """
        Get param from nucleus

        :param data_frame:
        :param col:
        :param nID:
        :param multi_rows:
        :return:
        """
        val = None

        if self.is_nucleus_in_nuclei(nID):
            if multi_rows is False:
                val = self.data_frames[data_frame].get_val_from_col_for_nID(col, nID, join_params=join_params)
            else:
                val = self.data_frames[data_frame].get_vals_from_col_for_nID(col, nID, join_params=join_params)

        # return only values from a certain z plane
        # if a value has a z plane, then it is the
        # first column
        if z >= 0:
            # extract z if more than one
            if len(val.shape) > 1:
                val = val[val[:, 0] == z]
            else:
                # check if z matches
                if val[0] != z:
                    val = None

        return val

    def get_nucleus_colour(self, nID):
        """
        Get colour of nucleus

        :param nID:
        :return:
        """
        colour = self.get_param_from_nucleus('data_params', 'colour', nID)

        if np.isnan(colour):
            colour = 0

        return colour

    def is_nucleus_rejected(self, nID):
        """
        Is nucleus rejected?

        :param nID:
        :return:
        """
        rejected = False

        nID_rejected = self.get_param_from_nucleus('data_params', 'rejected', nID)

        if nID_rejected is not None and nID_rejected != 'Nan' and nID_rejected > 0:
            rejected = True

        return rejected

    def get_accepted_nIDs(self):
        """
        Return only accepted nIDs

        :return:
        """
        accepted_nIDs = list(self.data_frames['data_params'].data_frame[
                                 self.data_frames['data_params'].data_frame.rejected < 1
                             ].index.values)
        return accepted_nIDs

    def get_param(self, param, nID):
        """
        Get data param from nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', param, nID)

    def get_nucleus_volume(self, nID):
        """
        Get volume of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'volume', nID)

    def get_nucleus_surface(self, nID):
        """
        Get surface of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'surface', nID)

    def get_nucleus_depth(self, nID):
        """
        Get depth of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'depth', nID)

    def get_nucleus_bbox(self, nID):
        """
        Get bbox of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'nuc_bbox', nID)

    def get_nucleus_volume_depth_ratio(self, nID):
        """
        Get volume depth ratio of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'volume_depth_ratio', nID)

    def get_nucleus_surface_volume_ratio(self, nID):
        """
        Get surface volume ratio of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'surface_volume_ratio', nID)

    def get_nucleus_neighbours(self, nID):
        """
        Get neighbours of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'neighbours', nID)

    def get_nucleus_neighbours_distance(self, nID):
        """
        Get neighbours distance of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'neighbours_distance', nID)

    def get_nucleus_direction(self, nID):
        """
        Get direction of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'direction', nID)

    def get_nucleus_apical_distance(self, nID):
        """
        Get apical distance of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'apical_dist', nID)

    def get_nucleus_nuclei_in_direction(self, nID):
        """
        Get nuclei in direction

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'nuclei_in_direction', nID)

    def get_nucleus_minor_axis(self, nID):
        """
        Get minor axis of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'minor_axis', nID)

    def get_nucleus_major_axis(self, nID):
        """
        Get major axis of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'major_axis', nID)

    def get_nucleus_mami_axis(self, nID):
        """
        Get major/minor axis of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'mami_axis', nID)

    def get_nucleus_minor_axis_orientation(self, nID):
        """
        Get minor axis orientation of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'minor_axis_orientation', nID)

    def get_nucleus_major_axis_orientation(self, nID):
        """
        Get major axis orientation of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'major_axis_orientation', nID)

    def get_nucleus_direction_orientation(self, nID):
        """
        Get direction orientation of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'direction_orientation', nID)

    def get_nucleus_lamin_int(self, nID):
        """
        Get lamin intensity of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'lamin_int', nID)

    def get_nucleus_dapi_int(self, nID):
        """
        Get dapi intensity of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'dapi_int', nID)

    def get_nucleus_membrane_int(self, nID):
        """
        Get membrane intensity of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'membrane_int', nID)

    def get_nucleus_centre(self, nID):
        """
        Get centre of nucleus

        :param nID:
        :return:
        """
        centre = self.get_param_from_nucleus('data_params', 'nuc_centre', nID)

        if type(centre) is str:
            # FIX: sometimes an array or a string is returned ... ?
            centre = np.fromstring(centre[1:-1], sep=' ')

        return centre

    def get_nucleus_centroids(self, nID, z=-1):
        """
        Get centroids of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_centroid', ['z', 'y', 'x'], nID, multi_rows=True, z=z)

    def get_nucleus_bboxes(self, nID, z=-1):
        """
        Get bounding boxes of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_bbox',
                                           ['z', 'min_row', 'min_col', 'max_row', 'max_col'],
                                           nID, multi_rows=True, z=z)

    def get_nucleus_areas(self, nID, z=-1):
        """
        Get areas of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_z_params', ['z', 'area'], nID, multi_rows=True, z=z)

    def get_nucleus_perimeters(self, nID, z=-1):
        """
        Get perimeters of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_z_params', ['z', 'perimeter'], nID, multi_rows=True, z=z)

    def get_nucleus_coords(self, nID, z=-1):
        """
        Get coords of nucleus

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_coords', ['z', 'y', 'x'], nID, multi_rows=True, z=z,
                                           join_params=['colour'])

    def get_classifier_edge_dist(self, nID):
        """
        Get edge distance from classifier parameters

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'nuc_edge_dist', nID)

    def get_classifier_topbot_ratio(self, nID):
        """
        Get top - bottom ratio from classifier parameters

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'area_topbot_ratio', nID)

    def get_classifier_depth_ratio(self, nID):
        """
        Get depth ratio from classifier parameters

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'area_depth_ratio', nID)

    def get_classifier_nuc_bbox(self, nID):
        """
        Get nucleus bounding box from classifier parameters

        :param nID:
        :return:
        """
        return self.get_param_from_nucleus('data_params', 'nuc_bbox', nID)

    def get_extra_param(self, extra_dict, nID, extra_param=None):
        """
        Get extra dictionary for nucleus

        :param extra_dict:
        :param extra_param:
        :param nID:
        :return:
        """
        extra_val = None

        # check if nucleus is in dataset
        if self.is_nucleus_in_nuclei(nID):
            if extra_dict in self.extra:
                if extra_param is not None and extra_param in self.extra[extra_dict]:
                    extra_val = self.extra[extra_dict][extra_param][nID]
                else:
                    # assemble all keys for dict
                    extra_val = dict()

                    for key, value in self.extra[extra_dict].items():
                        extra_val[key] = value[nID]

        return extra_val

    def get_img_boxes(self, nID, extra_param=None):
        """
        Get boxes for nucleus

        :param extra_param:
        :param nID:
        :return:
        """
        img_boxes = None

        # check if nucleus is in dataset
        if self.is_nucleus_in_nuclei(nID):
            img_box_key = list(self.img_boxes.keys())[0]
            if (nID in list(self.img_boxes[img_box_key].keys())) is False:
                self.build_nucleus_boxes(nID)

            # check if nucleus has a box
            if extra_param is not None and extra_param in self.img_boxes:
                img_boxes = self.img_boxes[extra_param][nID]
            else:
                # assemble all keys for dict
                img_boxes = dict()

                for key, value in self.img_boxes.items():
                    img_boxes[key] = value[nID]

        return img_boxes

    def get_extra_projections(self, nID, extra_param=None):
        """
        Get projections for nucleus

        :param extra_param:
        :param nID:
        :return:
        """
        return self.get_extra_param('img_projections', nID, extra_param)

    def get_extra_infos(self, nID, extra_param=None):
        """
        Get info for nucleus

        :param extra_param:
        :param nID:
        :return:
        """
        return self.get_extra_param('infos', nID, extra_param)

    def get_next_nID(self, nID, direction):
        """
        Get next nID

        :param nID:
        :return:
        """
        next_nID = -1
        asc = True

        # condition for next nID
        if direction > 0:
            next_cond = (self.data_frames['data_params'].data_frame.index > nID)
        elif direction < 0:
            next_cond = (self.data_frames['data_params'].data_frame.index < nID)
            asc = False
        else:
            next_cond = (self.data_frames['data_params'].data_frame.index == nID)

        next_nIDs = list(self.data_frames['data_params'].data_frame.sort_index(ascending=asc)[
                             (self.data_frames['data_params'].data_frame.rejected < 1)& next_cond
                         ].index.values)

        if len(next_nIDs) > 0:
            next_nID = next_nIDs[0]

        return next_nID

    """
    Setters
    """

    def set_param_for_nucleus(self, data_frame, param, val, nID):
        """
        Get param from nucleus

        :param data_frame:
        :param param:
        :param val:
        :param nID:
        :return:
        """
        val = None

        if self.is_nucleus_in_nuclei(nID):
            self.data_frames[data_frame].change_col_for_nID(param, nID, val)

        return val

    def set_nucleus_colour(self, val, nID):
        """
        Set colour of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('colour', nID, val)

    def reject_nucleus(self, nID):
        """
        Reject nucleus

        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('rejected', nID, 1)

    def accept_nucleus(self, nID):
        """
        Accept nucleus

        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('rejected', nID, 0)

    def set_nucleus_volume(self, val, nID):
        """
        Set volume of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('volume', nID, val)

    def set_nucleus_surface(self, val, nID):
        """
        Set surface of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('surface', nID, val)

    def set_nucleus_depth(self, val, nID):
        """
        Set depth of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('depth', nID, val)

    def set_nucleus_neighbours(self, val, nID):
        """
        Set neighbours of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('neighbours', nID, val)

    def set_nucleus_neighbours_distance(self, val, nID):
        """
        Set neighbours distance of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('neighbours_distance', nID, val)

    def set_nucleus_direction(self, val, nID):
        """
        Set direction of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('direction', nID, val, is_array=True)

    def set_nucleus_apical_distance(self, val, nID):
        """
        Set apical distance of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('apical_dist', nID, val)

    def set_nucleus_nuclei_in_direction(self, val, nID):
        """
        Set nuclei in direction of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('nuclei_in_direction', nID, val)

    def set_nucleus_minor_axis(self, val, nID):
        """
        Set minor of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('minor_axis', nID, val)

    def set_nucleus_major_axis(self, val, nID):
        """
        Set major of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('major_axis', nID, val)

    def set_nucleus_mami_axis(self, val, nID):
        """
        Set major/minor axis of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('mami_axis', nID, val)

    def set_nucleus_minor_axis_orientation(self, val, nID):
        """
        Set minor axis orientation of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('minor_axis_orientation', nID, val)

    def set_nucleus_major_axis_orientation(self, val, nID):
        """
        Set major axis orientation of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('major_axis_orientation', nID, val)

    def set_nucleus_direction_orientation(self, val, nID):
        """
        Set direction orientation of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('direction_orientation', nID, val)

    def set_nucleus_bbox(self, val, nID):
        """
        Set bbox of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('nuc_bbox', nID, val, is_array=True)

    def set_nucleus_volume_depth_ratio(self, val, nID):
        """
        Set volume depth ratio of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('volume_depth_ratio', nID, val)

    def set_nucleus_surface_volume_ratio(self, val, nID):
        """
        Set surface volume ratio of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('surface_volume_ratio', nID, val)

    def set_nucleus_lamin_int(self, val, nID, force_add=False):
        """
        Set lamin intensity of nucleus

        :param val:
        :param nID:
        :param force_add:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('lamin_int', nID, val, force_add=force_add)

    def set_nucleus_dapi_int(self, val, nID, force_add=False):
        """
        Set dapi intensity of nucleus

        :param val:
        :param nID:
        :param force_add:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('dapi_int', nID, val, force_add=force_add)

    def set_nucleus_membrane_int(self, val, nID, force_add=False):
        """
        Set membrane intensity of nucleus

        :param val:
        :param nID:
        :param force_add:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('membrane_int', nID, val, force_add=force_add)

    def set_nucleus_centre(self, val, nID):
        """
        Set centre of nucleus

        :param val:
        :param nID:
        :param force_add:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('nuc_centre', nID, val, is_array=True)

    def set_nucleus_centroids(self, val, nID):
        """
        Set centroid of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_centroid'].change_col_for_nID(['z', 'y', 'x'], nID, val)

    def set_nucleus_bboxes(self, val, nID):
        """
        Set bounding boxes of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_centre'].change_col_for_nID(
            ['z', 'min_row', 'min_col', 'max_row', 'max_col'], nID, val)

    def set_nucleus_areas(self, val, nID):
        """
        Set areas of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_z_params'].change_col_for_nID(['z', 'area'], nID, val)

    def set_nucleus_perimeters(self, val, nID):
        """
        Set perimeters of nucleus

        :param val:
        :param nID:
        :return:
        """
        self.data_frames['data_z_params'].change_col_for_nID(['z', 'perimeter'], nID, val)

    def set_nucleus_coords(self, val, nID):
        """
        Set coords of nucleus

        :param val:
        :param nID:
        :return:
        """
        # TODO how to set multiple values for coords
        # ValueError: cannot set using a slice indexer with a different length than the value
        self.data_frames['data_coords'].change_col_for_nID(['z', 'y', 'x'], nID, val)

    def set_classifier_edge_dist(self, val, nID, force_add=False):
        """
        Set edge dist of classifier

        :param val:
        :param nID:
        :param force_add:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('nuc_edge_dist', nID, val, force_add=force_add)

    def set_classifier_topbot_ratio(self, val, nID, force_add=False):
        """
        Set top - bottom ratio of classifier

        :param val:
        :param nID:
        :param force_add:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('area_topbot_ratio', nID, val, force_add=force_add)

    def set_classifier_depth_ratio(self, val, nID, force_add=False):
        """
        Set depth ratio of classifier

        :param val:
        :param nID:
        :param force_add:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('area_depth_ratio', nID, val, force_add=force_add)

    def set_classifier_nuc_bbox(self, val, nID, force_add=False):
        """
        Set nucleus bounding box of classifier

        :param val:
        :param nID:
        :param force_add:
        :return:
        """
        self.data_frames['data_params'].change_col_for_nID('nuc_bbox', nID, val, force_add=force_add)
