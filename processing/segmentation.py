"""
Segmentation algorithms. This includes the merging and correction of nuclei.
"""

import csv
import numpy as np
import random as rdm
import math
import pickle
import os
import re
import operator

from skimage.measure import regionprops
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import threading

import storage.config as cfg

from storage.images import Image
from storage.stacks import Stack
from processing.image import ImageProcessing
from storage.image import ImageHandler
from storage.struct import Struct
from processing.filter import Dilation, Equalise
from storage.lookup import Lookup

from frontend.figures.plot import Plot

class Segmentation:

    def __init__(self, image_info):
        print('=== Segmentation for %s ===' % image_info['ID'])

        load_non_nuclei = False

        self.image_info = image_info

        if ImageHandler.is_revision_by_ID(image_info['ID']) is True:
            # create a new segmentation
            load_non_nuclei = True

        # create new struct for stacks
        self.stacks = Stack()

        # load non-nuclei instead of lamin
        if load_non_nuclei is True:
            # load non-nuclei from parent experiment
            parent_info = ImageHandler.get_parent_info_by_id(image_info['ID'])

            # prepare non-nuclei path
            non_nuclei_path =\
                self.get_results_dir(exp_id=parent_info['ID']).stacks_corr + cfg.file_stack_non_nuclei

            # load image data
            self.image_stack = ImageHandler.load_image(parent_info,
                                                       non_nuclei_path=non_nuclei_path)

            self.non_nuclei_loaded = True
        else:
            # load image data
            self.image_stack = ImageHandler.load_image(self.image_info)

            self.non_nuclei_loaded = False

        # set stacks
        self.stacks.lamin = self.image_stack[ImageHandler.CHN_LAMIN]
        self.stacks.membrane = self.image_stack[ImageHandler.CHN_MEMBRANE]
        self.stacks.dapi = self.image_stack[ImageHandler.CHN_DAPI]

        # load nuclei criteria
        self.get_nuclei_criteria()

        # for raw reload
        self.raw_nuclei = None
        self.raw_lookup = None
        self.stacks.raw_labels_props = None
        self.nuc_corr_loaded = False
        self.labels_props_corr_loaded = False

        self.correction = None
        self.lookup = Lookup()

    def set_correction(self, correction):
        # set correction instance
        self.correction = correction

    def merge_parent_with_rev(self, rev_info):
        """
        Merge stacks and segmentation of the parent with revisions.
        This method should be called from the parent!

        :param rev_info:
        :return:
        """
        # get revision directories
        rev_dirs = self.get_results_dir(rev_info['ID'])

        # load nuclei from revision
        with open(rev_dirs.nuclei_corr, "rb") as fin:
            rev_nuclei = pickle.load(fin)

        # build a new list of nuclei
        nuclei_merge = list()

        # go through parent nuclei
        for nucleus in self.nuclei:
            # reset id
            nucleus['nID'] = len(nuclei_merge)
            nuclei_merge.append(nucleus)

        # go through revision nuclei
        for nucleus in rev_nuclei:
            # reset id
            nucleus['nID'] = len(nuclei_merge)
            nuclei_merge.append(nucleus)

        # save nuclei
        self.set_nuclei(nuclei_merge)

    def postprocess_for_nuclei(self):
        """
        Make final changes to nucleus after merging

        :param nuclei:
        :return:
        """

        for nucleus in self.nuclei:
            self.postprocess_for_nucleus(nucleus)

    def postprocess_for_nucleus(self, nucleus):
        """
        Apply postprocessing for nucleus

        :param nucleus:
        :return:
        """

        # dilate nuclei
        dilate = Dilation()

        # go through coordinates and add to frame
        for z, coords in enumerate(nucleus['coords']):
            # init frame
            frame = np.zeros_like(self.stacks.nuclei[0])

            for coord in coords[1]:
                frame[coord[0], coord[1]] = 1

            # dilate
            dilated = dilate.apply(frame, {'size': cfg.merge_post_dil, 'bin': None})

            # update coordinates
            nucleus['coords'][z] = (coords[0], np.argwhere(dilated))

    def save_merge_segmentation(self):
        """
        Save merge segmentation

        :param nuclei:
        :param stack_dir:
        :return:
        """
        # get directories
        result_dirs = self.get_results_dir()

        # add nuclei to stack
        nuclei_stack = Segmentation.add_nuclei_to_stack(
            self.nuclei, np.zeros_like(self.stacks.nuclei))

        # show lamin only where there is a nuclei - dilate first
        nuclei_lamin = self.stacks.lamin.copy()
        nuclei_lamin[nuclei_stack == 0] = 0

        # save stacks
        ImageHandler.save_stack_as_tiff(self.stacks.lamin, result_dirs.stacks_merge + cfg.file_stack_lamin)
        ImageHandler.save_stack_as_tiff(self.stacks.labels, result_dirs.stacks_merge + cfg.file_stack_labels)
        ImageHandler.save_stack_as_tiff(nuclei_stack, result_dirs.stacks_merge + cfg.file_stack_nuclei)
        ImageHandler.save_stack_as_tiff(nuclei_lamin, result_dirs.stacks_merge + cfg.file_stack_nuclam)

        # save nuclei
        with open(result_dirs.merge + cfg.file_nuclei, "wb") as fin:
            pickle.dump(self.nuclei, fin)

    def process(self):
        """
        Process stack

        :return:
        """
        # apply filters
        self.stacks.labels = self.apply_filters()

    def segment(self, process=True, merge=True, filter=True):
        """
        Segment stack

        :param lamin_stack:
        :return:
        """
        # reload criteria
        self.get_nuclei_criteria(force_reload=True)

        # process stack
        if process is True:
            print('Process image')
            self.process()

        # The image for segmentation is already the non-nuclei one
        #if self.non_nuclei_loaded:
        #    # remove nuclei that have already been segmented
        #    self.stacks.labels[self.stacks.nuclei > 0] = 0

        # segment labeled image
        if merge is True:
            print('Merge labels')
            self.merge_labels()

        # filter nuclei
        if filter is True:
            print('Filter nuclei')
            self.filter_nuclei()

        # add filtered nuclei to stack
        if hasattr(self, 'nuclei') and self.nuclei is not None:
            if hasattr(self.stacks, 'nuclei') and self.stacks.nuclei is not None:
                self.stacks.nuclei = Segmentation.add_nuclei_to_stack(self.nuclei,
                                                                      np.zeros_like(self.stacks.labels))

    def apply_filters(self):
        """
        Apply filters to planes

        :return:
        """

        return ImageProcessing.apply_filters_by_image_info(self.image_info, self.stacks.lamin)

    def get_validated_params_for_nucleus(self, nucleus):
        """
        Return validated params for nucleus

        :param nucleus:
        :return:
        """
        # get labels for nucleus
        nucleus_labels = self.build_label_from_nucleus(nucleus)

        # build dict for params
        validated_nucleus_params = dict()
        validated_labels_params = dict()

        # validate nuclei
        for nuclei_filter in cfg.filter_criteria_nuclei:
            cur_filter = nuclei_filter.lower()

            if self.is_param_in_nuclei(cur_filter) is True:
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

    def filter_planes(self):
        """
        Remove labels from stack that do not meet the merging criteria
        :return:
        """
        # load label properties
        self.get_label_props()

        # iterate through the planes and look at the individual labels
        for z, ids in enumerate(self.stacks.labels_props):
            tmp_del = 0

            for cur_id, cur_params in enumerate(ids):
                delete_label = False

                # take a label
                cur_label = self.stacks.labels_props[z][cur_id]

                # go through label criteria
                for labels_filter in cfg.filter_criteria_labels:
                    if self.is_param_in_criteria_range(cur_label, labels_filter) is False:
                        delete_label = True

                if delete_label:
                    self.stacks.labels_props[z][cur_id] = None
                    tmp_del += 1

            # how many labels did you delete?
            print('Labels deleted in plane %i: %i of %i' % (z, tmp_del, len(ids)))

            # redefine plane by removing None values
            self.stacks.labels_props[z] = [x for x in self.stacks.labels_props[z] if x is not None]

    def filter_nuclei(self):
        """
        Filter nuclei on specific requirements

        :return:
        """
        filtered_nuclei = list()
        removed_nuclei = list()

        for nID, params in enumerate(self.nuclei):
            criteria_met = True

            # go through nuclei criteria
            for label_filter in cfg.filter_criteria_nuclei:
               # test if param is in nuclei
               if self.is_param_in_nuclei(label_filter.lower()) is True:
                    if self.is_param_in_criteria_range(params, label_filter) is False:
                        criteria_met = False

            # if all criteria are met, add the nucleus to the filtered list
            if criteria_met is True:
                filtered_nuclei.append(params)
            else:
                removed_nuclei.append(params)

        # how nuclei survived the filtering?
        print('Nuclei accepted: %i; rejected: %i'
              % (len(filtered_nuclei), len(self.nuclei) - len(filtered_nuclei)))

        self.set_nuclei(filtered_nuclei)

        return removed_nuclei

    def is_param_in_criteria_range(self, cur_object, param):
        """
        Is the label in range of parameters?

        :param cur_object:
        :param param:
        :return:
        """
        in_range = True

        if cur_object is not None:
            # too high?
            if self.nuc_criteria[param]['MAX'] is not None:
                if self.nuc_criteria[param]['MAX'] > 0 \
                    and cur_object[param.lower()] > self.nuc_criteria[param]['MAX']:
                    in_range = False

            # too low?
            if self.nuc_criteria[param]['MIN'] is not None:
                if self.nuc_criteria[param]['MIN'] > 0 \
                    and cur_object[param.lower()] <= self.nuc_criteria[param]['MIN']:
                    in_range = False
        else:
            in_range = False

        return in_range

    def set_nuclei(self, nuclei):
        """
        Set nuclei and lookup tables

        :param nuclei:
        :return:
        """
        self.nuclei = nuclei

    def update_nucleus(self, new_nucleus):
        """
        Update nucleus in nuclei

        :param new_nucleus:
        :return:
        """
        self.set_nucleus_by_id(new_nucleus['nID'], new_nucleus)

    def remerge_nucleus(self, nID, corr_start, corr_stop,
                        merge_depth=False, force_raw_labels_props=False, take_nucleus=None):
        """
        Remerge nucleus while ignoring the merge depth parameter

        :return:
        """
        # get nucleus
        if take_nucleus is not None:
            nucleus = take_nucleus
        else:
            nucleus = self.get_nucleus_by_id(nID)

        # get first and last plane of the nucleus and coordinates
        nucleus_start = nucleus['coords'][0][0]
        nucleus_stop = nucleus['coords'][-1][0]

        print('nID-%i OLD: %i - %i NEW: %i - %i' % (nucleus['nID'], nucleus_start, nucleus_stop, corr_start, corr_stop))

        # correction needed?
        corr_needed = True

        if nucleus_start == corr_start and nucleus_stop == corr_stop:
            corr_needed = False

        if corr_needed:
            # correction top
            if nucleus_start > corr_start:
                # merge up
                print('merge up')
                nucleus = self.remerge_nucleus_part(nucleus, nucleus_start, corr_start,
                                                    merge_depth=merge_depth, force_raw_labels_props=force_raw_labels_props)
            elif nucleus_start < corr_start:
                # delete planes
                print('delete top')
                nucleus = self.delete_nucleus_part(nucleus, corr_start, nucleus_start)

            # correct bottom
            if nucleus_stop < corr_stop:
                # merge down
                print('merge down')
                nucleus = self.remerge_nucleus_part(nucleus, nucleus_stop, corr_stop,
                                                    merge_depth=merge_depth, force_raw_labels_props=force_raw_labels_props)
            elif nucleus_stop > corr_stop:
                # delete planes
                print('delete bottom')
                nucleus = self.delete_nucleus_part(nucleus, corr_stop, nucleus_stop)

            nucleus_start = nucleus['coords'][0][0]
            nucleus_stop = nucleus['coords'][-1][0]

            print('nID-%i RESULT: %i - %i' % (nucleus['nID'], nucleus_start, nucleus_stop))

            # update nucleus
            self.update_nucleus(nucleus)

            # postprocess
            self.postprocess_for_nucleus(nucleus)

            # recalculate params
            self.calc_nucleus_params(nucleus, self.stacks.nuclei)

        return nucleus

    def delete_nucleus_part(self, nucleus, new_start, new_stop):
        """
        Delete part of the nucleus

        :param nucleus:
        :param range_start:
        :param range_stop:
        :return:
        """
        # adjust merge range
        range_start = new_start
        range_stop = new_stop

        # set range for deleting
        if new_stop < new_start:
            merge_range = list(reversed(range(range_stop, range_start)))
        else:
            merge_range = list(reversed(range(range_start + 1, range_stop + 1)))

        # go up or down from the start of the nucleus
        for y in merge_range:
            nucleus = self.del_from_nucleus(y, nucleus)

        return nucleus

    def remerge_nucleus_part(self, nucleus, new_start, new_stop,
                             merge_depth=False, force_raw_labels_props=False):
        """
        Remerge nucleus part

        :param nucleus:
        :param nucleus_start:
        :param corr_plane:
        :param merge_depth:
        :return:
        """
        # adjust merge range
        # FIX: for nuclei whose ends are not merged properly - take the middle
        new_start = nucleus['coords'][int(len(nucleus['coords'])/2)][0]

        range_start = new_start
        range_stop = new_stop

        # set range for merging
        if new_stop < new_start:
            merge_range = reversed(list(range(range_stop, range_start)))
        else:
            merge_range = list(range(range_start + 1, range_stop + 1))

        # operate on a copy of the labeled stack to merge nuclei
        if force_raw_labels_props is True:
            merge_stack = self.get_raw_labels_props().copy()
        else:
            merge_stack = self.stacks.labels_props.copy()

        # merge down by starting with the last nucleus plane
        #print('range %i - %i' % (range_start, range_stop))
        cur_label = self.build_label_from_nucleus(nucleus, range_start - nucleus['coords'][0][0])[0]
        z = range_start

        # go through stack and merge labels
        last_y = 0
        for y in merge_range:
            try_to_merge = True

            # z has the value of the last successful merge
            # if y, the current plane, is bigger than z considering the merge depth then stop
            if merge_depth is True:
                if new_stop < new_start:
                    if y < (z - cfg.merge_depth):
                        try_to_merge = False
                else:
                    if y > (z + cfg.merge_depth):
                        try_to_merge = False

            if try_to_merge is True:
                # get a label in the next plane in the vicinity of the current label
                min_row, max_row, min_col, max_col = Segmentation.calc_vicinity_of_label(cur_label)

                #print('%i:%i:%i:%i' % (min_row, max_row, min_col, max_col))

                # set overlap for the next planes
                has_overlap = False

                # go through the labels of the next plane
                for next_id in range(0, len(merge_stack[y])):
                    # take a label
                    next_label = merge_stack[y][next_id]

                    # is the label in the vicinity of the current label?
                    if next_label is not None \
                            and (next_label['centroid'][0] > min_row \
                                 and next_label['centroid'][0] < max_row) \
                            and (next_label['centroid'][1] > min_col \
                                 and next_label['centroid'][1] < max_col):
                        # calculate the overlap of the current and the next label
                        overlap = Segmentation.calc_overlap(self.stacks.labels[0], cur_label, next_label)

                        # is the overlap sufficient for merging?
                        if overlap > cfg.merge_min_overlap:
                            # there is an overlap
                            has_overlap = True

                            # did you have to skip the last planes?
                            # Then add the AND_img in between
                            if y > (z + 1):
                                # Take the AND_img and calculate new props
                                AND_props =\
                                    regionprops(
                                        Segmentation.calc_AND_img(
                                            self.stacks.labels[0],
                                            cur_label,
                                            next_label).astype(np.int))[0]

                                AND_labels_props = Segmentation.create_labels_props(AND_props)

                                for i in range((z + 1), y):
                                    #print('add AND nucleus %i' % i)
                                    Segmentation.add_to_nucleus(i, AND_labels_props, nucleus)

                            #print('add nucleus %i' % y)
                            Segmentation.add_to_nucleus(y, next_label, nucleus)

                            # reset current parameters
                            z = y
                            cur_label = next_label

                last_y = y

        # if the last plane has no overlap add the AND_props
        if has_overlap == False and merge_depth is False:
            if (z + 1) > last_y:
                last_range = reversed(list(range(last_y, (z + 1))))
            else:
                last_range = list(range((z + 1), last_y + 1))

            for i in last_range:
                Segmentation.add_to_nucleus(i, cur_label, nucleus)

        return nucleus

    def merge_labels(self):
        """
        Merge potential nuclei based on their selection criteria
        :return:
        """
        # delete labels that do not fit the selection criteria
        self.filter_planes()

        # store nuclei in a central list
        nuclei = list()

        # operate on a copy of the labeled stack to merge nuclei
        merge_stack = self.stacks.labels_props.copy()

        # go through stack and merge labels
        # take one label at a time and merge down
        # as far you can and delete the labels that
        # you used on the way
        for raw_z, plane_labels in enumerate(self.stacks.labels_props):
            print('Merge %i labels in plane %i' % (len(plane_labels), raw_z))

            for raw_id, raw_label in enumerate(plane_labels):
                if raw_id % 100 == 0:
                    print('\t%i' % (raw_id))

                cur_label = None

                if raw_label is not None:
                    # take a label
                    cur_label = raw_label.copy()
                    cur_id = raw_id
                    z = raw_z

                # create a nucleus
                nucleus = None

                while cur_label is not None:
                    # compare to next plane labels and then merge down
                    # until no overlap is found anymore in range of the merge depth

                    # get a label in the next plane in the vicinity of the current label
                    min_row, max_row, min_col, max_col = Segmentation.calc_vicinity_of_label(cur_label)

                    # list of skipped ids that will be merged by approximating their coords with AND
                    merge_skipped = list()

                    # set overlap for the next planes
                    has_overlap = False

                    # get the next label within the merge depth
                    for y in range((z + 1), (z + cfg.merge_depth + 1)):
                        if y < len(merge_stack)\
                                and merge_stack[y] is not None\
                                and has_overlap is False:
                            # to store highest overlap for this plane
                            highest_overlap = -1
                            highest_overlap_ID = -1
                            highest_overlap_Z = -1

                            # go through the labels of the next plane
                            for next_id in range(0, len(merge_stack[y])):
                                # take a label
                                next_label = merge_stack[y][next_id]

                                # is the label in the vicinity of the current label?
                                if next_label is not None \
                                        and (next_label['centroid'][0] > min_row \
                                             and next_label['centroid'][0] < max_row) \
                                        and (next_label['centroid'][1] > min_col \
                                             and next_label['centroid'][1] < max_col):
                                    # calculate the overlap of the current and the next label
                                    overlap = Segmentation.calc_overlap(self.stacks.labels[0], cur_label, next_label)

                                    # was this the highest overlap so far in this plane?
                                    if overlap > highest_overlap:
                                        highest_overlap = overlap
                                        highest_overlap_ID = next_id
                                        highest_overlap_Z = y

                                    # is the overlap sufficient for merging?
                                    if overlap > cfg.merge_min_overlap:
                                        # there is an overlap
                                        has_overlap = True
                                        #print('overlap %i in %i:%i to %i:%i' % (overlap, z, cur_id, y, next_id))

                                        # create nucleus
                                        if nucleus is None:
                                            nucleus = Segmentation.create_nucleus(z, cur_label)
                                        else:
                                            # did you have to skip the last planes?
                                            # Then add the AND_img in between
                                            if y > (z + 1):
                                                # Take the AND_img and calculate new props
                                                AND_props =\
                                                    regionprops(
                                                        Segmentation.calc_AND_img(
                                                            self.stacks.labels[0],
                                                            cur_label,
                                                            next_label).astype(np.int))[0]

                                                AND_labels_props = Segmentation.create_labels_props(AND_props)

                                                for i in range((z + 1), y):
                                                    Segmentation.add_to_nucleus(i, AND_labels_props, nucleus)

                                                    if len(merge_skipped) > 0:
                                                        skipped_label = merge_skipped.pop(0)
                                                        merge_stack[skipped_label[0]][skipped_label[1]] = None

                                            Segmentation.add_to_nucleus(y, next_label, nucleus)

                                            # set new current label for next iteration and
                                            # delete the current one from the list
                                            merge_stack[z][cur_id] = None

                                            # reset current parameters
                                            cur_id = next_id
                                            z = y
                                            cur_label = next_label

                                            # stop to look for labels in this plane
                                            next_id = len(merge_stack[y])

                            # add the highest overlap after each iteration in the merge depth
                            if has_overlap == False:
                                # Add ID to the merge skip list if this had the highest overlap
                                # This will then be used to calculate the AND image to approximate
                                # missing planes in the nucleus

                                if highest_overlap_ID >= 0:
                                    merge_skipped.append([highest_overlap_Z, highest_overlap_ID])

                    # no overlapping label has been found in the merge depth
                    if has_overlap == False:
                        cur_label = None

                # add nucleus to the list if one could be found
                if nucleus is not None:
                    # set nucleus ID
                    nucleus = Segmentation.create_nID_by_nuclei(nucleus, nuclei)

                    # add nucleus to list
                    nuclei.append(nucleus)

        self.set_nuclei(nuclei)

        # update nuclei params
        self.update(save=False)

        # perform post processing
        self.postprocess_for_nuclei()

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

    @staticmethod
    def calc_overlap(tpl_img, cur_label, next_label):
        """
        Caluclate overlap between two labels

        :param tpl_img:
        :param cur_label:
        :param next_label:
        :return:
        """
        # get AND sum
        AND_sum = Segmentation.calc_AND_img(tpl_img, cur_label, next_label).sum()
        XOR_sum = Segmentation.calc_XOR_img(tpl_img, cur_label, next_label).sum()

        # Calculate overlap
        min_cur_next = min(cur_label['area'], next_label['area'])

        overlap = (AND_sum / min_cur_next) * 100
        #overlap = (AND_sum / (AND_sum + XOR_sum)) * 100

        return overlap

    @staticmethod
    def calc_AND_img(tpl_img, cur_label, next_label):
        """
        Calculate logical and between two labels

        :param tpl_img:
        :param cur_label:
        :param next_label:
        :return:
        """
        # get a blank image
        blank_img = np.zeros_like(tpl_img)

        # add nucleus to blank image
        cur_img = Segmentation.add_label_to_img(cur_label, blank_img, colour=1)
        next_img = Segmentation.add_label_to_img(next_label, blank_img, colour=1)

        # calculate AND
        AND_img = np.logical_and(cur_img, next_img)

        return AND_img

    @staticmethod
    def calc_XOR_img(tpl_img, cur_label, next_label):
        """
        Calculate logical xor between two labels

        :param tpl_img:
        :param cur_label:
        :param next_label:
        :return:
        """
        # get a blank image
        blank_img = np.zeros_like(tpl_img)

        # add nucleus to blank image
        cur_img = Segmentation.add_label_to_img(cur_label, blank_img, colour=1)
        next_img = Segmentation.add_label_to_img(next_label, blank_img, colour=1)

        # calculate AND
        XOR_img = np.logical_xor(cur_img, next_img)

        return XOR_img

    @staticmethod
    def calc_vicinity_of_label(label):
        """
        Calculate a box for a label

        :param label:
        :return:
        """
        # calc radius of nucleus
        cur_radius = math.sqrt(label['area']/math.pi)

        # radius of max_area
        min_row = label['centroid'][0] - cur_radius
        max_row = label['centroid'][0] + cur_radius

        min_col = label['centroid'][1] - cur_radius
        max_col = label['centroid'][1] + cur_radius

        return min_row, max_row, min_col, max_col

    @staticmethod
    def create_nucleus(z, label):
        """
        Create a new nucleus

        :param z:
        :param label:
        :return:
        """
        nucleus = dict()

        # go through label properties to get
        for prop in cfg.label_props_to_get:
            nucleus[prop] = list()
            nucleus[prop].append((z, label[prop]))

        nucleus['colour'] = label['colour']

        return nucleus

    def build_bw_nuclei(self):
        """
        Build black/white version of nuclei

        :param nuclei:
        :return:
        """
        bw_nuclei = np.zeros_like(self.stacks.nuclei)
        bw_nuclei[self.stacks.nuclei > 0] = 1

        return bw_nuclei

    def convert_nucleus_label_props_to_lists(self, nucleus):
        """
        Convert nucleus label props to lists for easier editing

        :param nucleus:
        :return:
        """
        for prop in cfg.label_props_to_get:
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
        props_to_convert = cfg.label_props_to_get
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
                                        data[-1].append(prop_val)
                        # single value
                        else:
                            for prop_z in nucleus[prop]:
                                data.append([prop_z[0], prop_z[1]])

                props_list[prop] = data

        return props_list

    def get_label_props_from_array_lists(self, nucleus, data_struct, data_cols):
        """
        Convert array lists to label props

        :param nucleus:
        :param data_struct:
        :param data_cols:
        :return:
        """

        # TODO go through data columns and map to nucleus
        for prop in data_cols:
            # get prop from struct
            data_frame = getattr(data_struct, prop)

            # slice nID from struct
            data_nID = data_frame.loc[nucleus['nID']]

            for i, data_item in data_nID.iteritems():
                # TODO - what is the datatype here?
                """
                volume ===
                1996.0
                surface ===
                795.742207411
                depth ===
                22.0
                z === <class 'pandas.core.series.Series'>
                0     0
                0     1
                """
                print(i, '===')
                print(type(data_item))
                print(data_item)

        return nucleus

    def build_nuclei_from_dataframe(self, data_struct, data_cols):
        """
        Build nuclei from dataframe

        :param data_struct:
        :param data_cols:
        :return:
        """
        nuclei = list()

        # get all IDs and build nuclei
        for nID in data_struct.data_params.index.values:
            print('load %i' % nID)
            nuclei.append(self.build_nucleus_from_dataframe(data_struct, data_cols, nID))

        return nuclei

    def build_nucleus_from_dataframe(self, data_struct, data_cols, nID):
        """
        Build a nucleus object from a given dataframe

        :param data_struct:
        :param data_cols:
        :param nID:
        :return:
        """
        nucleus = dict()

        nucleus['nID'] = nID

        # set params
        for kID, value in enumerate(data_struct.data_params.loc[nID]):
            key = data_struct.data_params.columns.values[kID]
            nucleus[key] = value

        # set intensities
        for kID, value in enumerate(data_struct.data_int.loc[nID]):
            key = data_struct.data_int.columns.values[kID]
            nucleus[key] = value

        # set label values
        nucleus = self.get_label_props_from_array_lists(nucleus, data_struct, data_cols)

        # set centre
        nucleus['centre'] = np.array(data_struct.data_centre.loc[nID])

        # set centroid
        nucleus['centroid'] = np.array(data_struct.data_centroid.loc[nID])

        # set coords
        nucleus['coords'] = np.array(data_struct.data_coords.loc[nID])

        # set area
        nucleus['area'] = np.array(data_struct.data_area.loc[nID])

        # convert to lists
        nucleus = self.convert_nucleus_label_props_to_lists(nucleus)

        # build image information

        # projections
        for key in data_struct.img_projection.columns.values:
            nucleus['projection_%s' % key] = data_struct.img_projection[key].loc[nID]

        # preview image
        nucleus['img'] = nucleus['projection_y']

        # create boxes for nucleus
        nucleus = self.build_nucleus_boxes(nucleus)

        return nucleus

    def calc_nuclei_params(self):
        """
        Calculate params for nuclei

        :return:
        """
        print('calc nucleus params for %i' % len(self.nuclei))

        # threading parameters
        nID = 0
        max_threads = cfg.merge_threads_nuc_params

        while nID < len(self.nuclei):
            print('\t%i' % nID)
            nuc_threads = []

            # start threads
            for i in range(0, max_threads):
                if nID < len(self.nuclei):
                    thread = threading.Thread(target=self.calc_nucleus_params, args=[self.nuclei[nID]])
                    thread.start()
                    nuc_threads.append(thread)

                    nID += 1

            # wait to finish
            for thread in nuc_threads:
                thread.join()

        # update lookup table
        self.lookup.rebuild_tables(self.nuclei)

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

        # take the first nucleus and try to look up param
        if param in self.nuclei[0]:
            is_in_nuclei = True

        return is_in_nuclei

    def calc_nucleus_params(self, nucleus):
        """
        Calculate parameters for nucleus

        :param nucleus:
        :return:
        """

        # volume
        nucleus['volume'] = 0
        for area in nucleus['area']:
            nucleus['volume'] += area[1]

        # surface
        nucleus['surface'] = 0

        # add first and last area slices to surface
        nucleus['surface'] += nucleus['area'][0][1]
        nucleus['surface'] += nucleus['area'][-1][1]

        for perimeter in nucleus['perimeter'][1:-2]:
            nucleus['surface'] += perimeter[1]

        # depth
        nucleus['depth'] = len(nucleus['coords'])

        # create box
        nucleus = self.build_nucleus_boxes(nucleus)

        # euqalise projections
        equalise_filter = Equalise()

        # create projections
        nucleus['projection_z'] = np.zeros_like(nucleus['box'][0, :, :])
        for z in range(0, nucleus['box'].shape[0]):
            nucleus['projection_z'] += nucleus['box'][z, :, :]

        nucleus['projection_z'] = equalise_filter.apply(nucleus['projection_z'])

        nucleus['projection_y'] = np.zeros_like(nucleus['box'][:, 0, :])
        for y in range(0, nucleus['box'].shape[1]):
            nucleus['projection_y'] += nucleus['box'][:, y, :]

        nucleus['projection_y'] = equalise_filter.apply(nucleus['projection_y'])

        nucleus['projection_x'] = np.zeros_like(nucleus['box'][:, :, 0])
        for x in range(0, nucleus['box'].shape[2]):
            nucleus['projection_x'] += nucleus['box'][:, :, x]

        nucleus['projection_x'] = equalise_filter.apply(nucleus['projection_x'])

        nucleus['img'] = nucleus['projection_y']

        # draw nucleus on stack
        nucleus_stack = self.add_nucleus_to_stack(nucleus, np.zeros_like(self.stacks.lamin), 1)

        # calculate DAPI and membrane intensity
        lamin_signal = self.stacks.lamin * nucleus_stack
        dapi_signal = self.stacks.dapi * nucleus_stack
        membrane_signal = self.stacks.membrane * nucleus_stack

        # get DAPI and membrane image
        #dapi_box = Plot.get_nucleus_box(nucleus, dapi_signal, cfg.criteria_select_nuc_box_offset)
        #membrane_box = Plot.get_nucleus_box(nucleus, membrane_signal, cfg.criteria_select_nuc_box_offset)

        lamin_mean = np.mean(lamin_signal.ravel()[np.flatnonzero(lamin_signal)])
        dapi_mean = np.mean(dapi_signal.ravel()[np.flatnonzero(dapi_signal)])
        membrane_mean = np.mean(membrane_signal.ravel()[np.flatnonzero(membrane_signal)])

        #nucleus['img'] = dapi_box[:, round(membrane_box.shape[1]/2), :]

        nucleus['lamin_int'] = lamin_mean
        nucleus['dapi_int'] = dapi_mean
        nucleus['membrane_int'] = membrane_mean

        # centre of nucleus by taking the average of centroids
        centroids_y = list()
        centroids_x = list()
        for centroid in nucleus['centroid']:
            centroids_y.append(centroid[1][0])
            centroids_x.append(centroid[1][1])

        # take average
        avg_centroid = (
            nucleus['centroid'][int(len(nucleus['centroid'])/2)][0],
            int(sum(centroids_y) / len(centroids_y)),
            int(sum(centroids_x) / len(centroids_x))
        )

        nucleus['centre'] = avg_centroid

        # calculate distance to image borders
        edge_distances = [
            avg_centroid[0], nucleus_stack.shape[0] - avg_centroid[0],
            avg_centroid[1], nucleus_stack.shape[1] - avg_centroid[1],
            avg_centroid[2], nucleus_stack.shape[2] - avg_centroid[2]
        ]

        nucleus['nuc_edge_dist'] = min(edge_distances)

        # take ratio of average area of top to bottom
        area_total = list()
        for area in nucleus['area']:
            area_total.append(area[1])

        area_top = area_total[0:int(len(area_total)/2)]
        area_bottom = area_total[int(len(area_total)/2):len(area_total)]
        area_ratio = float(sum(area_top)/sum(area_bottom))

        nucleus['area_topbot_ratio'] = area_ratio
        nucleus['area_depth_ratio'] = float(sum(area_total)/nucleus['depth'])

        # calculate donut for nucleus
        donut_props = self.calc_donuts_for_nucleus(nucleus, self.stacks)

        # set donut props for nucleus
        for key in donut_props.keys():
            nucleus[key] = donut_props[key]

        # calc neighbour box
        nuc_bbox = [
            nucleus['coords'][0][0], nucleus['centre'][1], nucleus['centre'][2],
            nucleus['coords'][-1][0], nucleus['centre'][1], nucleus['centre'][2]
        ]

        # go through all bbox
        for bbox in nucleus['bbox']:
            # min
            for i in range(0, 2):
                if bbox[1][i] < nuc_bbox[i + 1]:
                    nuc_bbox[i + 1] = bbox[1][i]

            # max
            for i in range(2, 4):
                if bbox[1][i] > nuc_bbox[i + 1]:
                    nuc_bbox[i + 1] = bbox[1][i]

        nucleus['nuc_bbox'] = nuc_bbox

        return nucleus

    def build_nucleus_boxes(self, nucleus):
        """
        Build nucleus boxes for display

        :param nucleus:
        :return:
        """
        bw_nuclei = self.build_bw_nuclei()

        bw_stack = Segmentation.add_nucleus_to_stack(nucleus, np.zeros_like(bw_nuclei), 1)
        nucleus['box'] = Plot.get_nucleus_box(nucleus,
                                              bw_stack,
                                              cfg.criteria_select_nuc_box_offset)
        nucleus['lamin_box'] = Plot.get_nucleus_box(nucleus,
                                                    self.stacks.lamin,
                                                    cfg.criteria_select_nuc_box_offset)
        nucleus['labels_box'] = Plot.get_nucleus_box(nucleus,
                                                    self.stacks.labels,
                                                    cfg.criteria_select_nuc_box_offset)
        nucleus['membrane_box'] = Plot.get_nucleus_box(nucleus,
                                                    self.stacks.membrane,
                                                    cfg.criteria_select_nuc_box_offset)
        nucleus['dapi_box'] = Plot.get_nucleus_box(nucleus,
                                                    self.stacks.dapi,
                                                    cfg.criteria_select_nuc_box_offset)

        # create cropped boxes for signals
        nucleus['cropped_lamin_box'] = nucleus['lamin_box'] * nucleus['box']
        nucleus['cropped_membrane_box'] = nucleus['membrane_box'] * nucleus['box']
        nucleus['cropped_dapi_box'] = nucleus['dapi_box'] * nucleus['box']

        # create RGB version
        nucleus['cropped_rgb_box'] = np.zeros(shape=(nucleus['cropped_lamin_box'].shape[0],
                                                     nucleus['cropped_lamin_box'].shape[1],
                                                     nucleus['cropped_lamin_box'].shape[2], 4))
        nucleus['cropped_rgb_box'][:, :, :, 0] = nucleus['cropped_membrane_box'].astype(float) / 255
        nucleus['cropped_rgb_box'][:, :, :, 1] = nucleus['cropped_lamin_box'].astype(float) / 255
        nucleus['cropped_rgb_box'][:, :, :, 2] = nucleus['cropped_dapi_box'].astype(float) / 255
        nucleus['cropped_rgb_box'][:, :, :, 3] = 0.0

        # set alpha '1' if one channel is over one
        for chn in range(0, nucleus['cropped_rgb_box'].shape[3]):
            nucleus['cropped_rgb_box'][:, :, :, 3][nucleus['cropped_rgb_box'][:, :, :, chn] > 0] = 1.0

        return nucleus

    def calc_donuts_for_nucleus(self, nucleus, stacks):
        """
        Calculate donuts for all labels of the nucleus
        and return the mean intensities

        :param nucleus:
        :param stacks:
        :return:
        """
        # to store mean intensities of donut
        donut_means = dict()

        # go through each label of the nucleus and draw a donut
        for z, coords in enumerate(nucleus['coords']):
            # build label
            label = self.build_label_from_nucleus(nucleus, z)[0]

            # get images
            imgs = Image()
            imgs.lamin = stacks.lamin[coords[0]]
            imgs.dapi = stacks.dapi[coords[0]]
            imgs.membrane = stacks.membrane[coords[0]]

            # calculate donut
            calc_props = Segmentation.calc_donut_for_label(label, imgs,
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

    @staticmethod
    def calc_donut_for_label(label_props, imgs, dilero_param=1):
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

    def get_listID(self, nID):
        """
        Return the position of the nucleus with nID

        :param nID:
        :return:
        """
        return Segmentation.get_listID_by_nID(self.nuclei, nID)

    @staticmethod
    def get_listID_by_nID(nuclei, nID):
        """
        Return the position of the nucleus with nID

        :param nuclei:
        :param nID:
        :return:
        """
        listID = 0

        # go through nuclei and look for nID
        for i, nucleus in enumerate(nuclei):
            if nucleus['nID'] == nID:
                listID = i

        return listID

    @staticmethod
    def get_slicebox_of_nucleus_img(nucleus, offset):
        """
        Calculate box of nucleus image

        :param nucleus:
        :param offset:
        :param outside:
        :return:
        """
        slice_y = Segmentation.get_slice_of_nucleus_img(nucleus, offset, 0)
        slice_box = Segmentation.get_slice_of_nucleus_img(nucleus, offset, 1, image=slice_y)

        return slice_box

    @staticmethod
    def get_slice_of_nucleus_img(nucleus, offset, slice_pos, outside=False, outside_left=True, image=None):
        """
        Calculate slice of nucleus image

        :param nucleus:
        :param offset:
        :param slice_pos:
        :param outside:
        :return:
        """
        if image is None:
            image = nucleus['img']

        # get middle
        mid_slice = round(image.shape[slice_pos]/2)

        # calculate offset from middle
        slice_min = mid_slice - offset
        slice_max = mid_slice + offset

        # adjust min/max for image boundaries
        if slice_min < 0:
            slice_min = 0

        if slice_max > image.shape[slice_pos]:
            slice_max = image.shape[slice_pos]

        # calculate slices
        if slice_pos == 0:
            if outside is True:
                if outside_left is True:
                    img_slice = image[0:slice_min, :]
                else:
                    img_slice = image[slice_max:, :]
            else:
                img_slice = image[slice_min:slice_max, :]
        else:
            if outside is True:
                if outside_left is True:
                    img_slice = image[:, 0:slice_min]
                else:
                    img_slice = image[:, slice_max:]
            else:
                img_slice = image[:, slice_min:slice_max]

        return img_slice

    def build_label_from_nucleus(self, nucleus, z=-1):
        """
        Build label from nucleus

        :param nucleus:
        :param z:
        :return:
        """
        labels_probs = list()

        if z >= 0:
            labels_probs.append(dict())

            # go through label properties to get
            for prop in cfg.label_props_to_get:
                labels_probs[-1][prop] = nucleus[prop][z][1]
        else:
            # go through all the planes of the nucleus
            for i, centroid in enumerate(nucleus['centroid']):
                labels_probs.append(dict())

                # go through label properties to get
                for prop in cfg.label_props_to_get:
                    labels_probs[-1][prop] = nucleus[prop][i][1]

                # calculate extra parameters
                imgs = Image()
                imgs.lamin = self.stacks.lamin[centroid[0]]
                imgs.dapi = self.stacks.dapi[centroid[0]]
                imgs.membrane = self.stacks.membrane[centroid[0]]
                labels_probs[-1] = Segmentation.calc_labels_props(labels_probs[-1], imgs)

        return labels_probs

    @staticmethod
    def add_to_nucleus(z, label, nucleus):
        """
        Map more data to a nucleus

        :param label:
        :return:
        """
        # see if you already have information in z
        prev_index = -1
        for index, val in enumerate(nucleus['centroid']):
            if val[0] == z:
                prev_index = index

        if prev_index < 0:
            # do you need to insert at the beginning?
            if z < nucleus['centroid'][0][0]:
                for prop in cfg.label_props_to_get:
                    nucleus[prop].insert(0, (z, label[prop]))
            else:
                for prop in cfg.label_props_to_get:
                    nucleus[prop].append((z, label[prop]))
        else:
            for prop in cfg.label_props_to_get:
                nucleus[prop][prev_index] = (z, label[prop])

        return nucleus

    @staticmethod
    def del_from_nucleus(z, nucleus):
        """
        Map more data to a nucleus

        :param label:
        :return:
        """
        # convert z to index
        index = z - nucleus['coords'][0][0]

        # delete variables
        for prop in cfg.label_props_to_get:
            nucleus[prop].pop(index)

        return nucleus

    def get_nuclei_criteria(self, force_reload=False):
        """
        Load criteria to accept labels for merging and later to
        accept the merged nuclei as potential nuclei
        :return:
        """
        # have you loaded the criteria already?
        if not hasattr(self, 'nuc_criteria') or force_reload is True:
            self.nuc_criteria = dict()

            # get init row
            init_row = Segmentation.get_nuclei_criteria_map()

            # init array with None
            for key in range(0, len(init_row)):
                self.nuc_criteria[init_row[key]] = dict()
                self.nuc_criteria[init_row[key]]['MIN'] = None
                self.nuc_criteria[init_row[key]]['MAX'] = None

            with open(cfg.file_nuc_criteria, 'r') as csvfile:
                criteria_reader = csv.reader(csvfile,
                                          delimiter=cfg.CSV_DEL,
                                          quotechar=cfg.CSV_QUOT)
                # go through mapping and create dictionary per keyword
                for row in criteria_reader:
                    if row[0] == ImageHandler.extract_expnum_from_ID(self.image_info['ID']):
                        # set values for criteria
                        for key in range(1, len(row)):

                            if row[key].isdigit():
                                val = int(row[key])
                            else:
                                try:
                                    val = float(row[key])
                                except ValueError:
                                    val = row[key]

                            # get min and max
                            min_val = None
                            max_val = None

                            # search for value
                            min_raw = re.search('[0-9]+.[0-9]+-', val)
                            max_raw = re.search('-[0-9]+.[0-9]+', val)

                            if min_raw is not None:
                                min_val = float(min_raw.group()[:-1])

                            if max_raw is not None:
                                max_val = float(max_raw.group()[1:])

                            self.nuc_criteria[init_row[key - 1]]['MIN'] = min_val
                            self.nuc_criteria[init_row[key - 1]]['MAX'] = max_val

                csvfile.close()

        return self.nuc_criteria

    def save_nuclei_criteria(self, new_criteria):
        """
        Save given criteria for nuclei selection

        :param new_criteria:
        :return:
        """

        # get map for criteria
        criteria_mapping = Segmentation.get_nuclei_criteria_map()

        # build a string from the criteria given
        criteria_rows = list()
        criteria_rows.append(self.image_info['ID'])

        # go through mapping
        for criteria in criteria_mapping:
            row = 'None'

            if criteria in new_criteria.keys():
                if new_criteria[criteria]['MIN'] is not None:
                    row = str(new_criteria[criteria]['MIN']) + '-'

                if new_criteria[criteria]['MIN'] is None and new_criteria[criteria]['MAX'] is not None:
                    row += '-'

                if new_criteria[criteria]['MAX'] is not None:
                    row += str(new_criteria[criteria]['MAX'])

            criteria_rows.append(row)

        # write to processing file
        ImageHandler.update_exp_csv(self.image_info['ID'],
                                    cfg.file_nuc_criteria,
                                    criteria_rows)

    @staticmethod
    def get_nuclei_criteria_map():
        """
        Return dictionary for nuclei criteria mapping

        :return:
        """
        init_row = str

        with open(cfg.file_nuc_criteria, 'r') as csvfile:
            criteria_reader = csv.reader(csvfile,
                                          delimiter=cfg.CSV_DEL,
                                          quotechar=cfg.CSV_QUOT)

            # go through mapping and create dictionary per keyword
            init_row = None

            for row in criteria_reader:
                # create dict for criteria based on the first row
                if row[0] == 'ID':
                    init_row = row[1:].copy()
                    break

            csvfile.close()

        return init_row

    def get_label_props(self):
        """
        Return properties of labels in a list in each plane of the stack
        :return:
        """
        # are the props already loaded?
        labels_props_loaded = False

        if hasattr(self.stacks, 'labels_props') and self.stacks.labels_props is not None:
            labels_props_loaded = True

        if labels_props_loaded is False:
            self.stacks.labels_props = list(range(0, len(self.stacks.labels)))

            # threading parameters
            z = 0
            max_threads = cfg.merge_threads_labels_props

            while z < len(self.stacks.labels):
                print('\t%i' % z)
                labels_threads = []

                # start threads
                for i in range(0, max_threads):
                    if z < len(self.stacks.labels):
                        thread = threading.Thread(target=self.get_labels_props_for_z, args=(z,))
                        thread.start()
                        labels_threads.append(thread)

                        z += 1

                # wait to finish
                for thread in labels_threads:
                    thread.join()

    def get_labels_props_for_z(self, z):
        """
        Get label props in a plane

        :param z:
        :return:
        """
        frame = self.stacks.labels[z]

        frame_prop = regionprops(frame.astype(np.int), cache=True)

        self.stacks.labels_props[z] = list()

        if cfg.general_verbose is True:
            print('Get %i props from plane %i' % (len(frame_prop), z))

        for object in frame_prop:
            imgs = Image()
            imgs.lamin = self.stacks.lamin[z]
            imgs.dapi = self.stacks.dapi[z]
            imgs.membrane = self.stacks.membrane[z]
            cur_obj = Segmentation.create_labels_props(object, imgs)

            self.stacks.labels_props[z].append(cur_obj.copy())

    @staticmethod
    def create_labels_props(object, imgs=None, create_lamin_donut=True):
        """
        Map props to label

        :return:
        """
        new_obj = dict()
        new_obj['label'] = object.label
        new_obj['coords'] = object.coords
        new_obj['img'] = object.filled_image
        new_obj['area'] = object.area
        new_obj['centroid'] = object.centroid
        new_obj['bbox'] = object.bbox
        new_obj['perimeter'] = object.perimeter
        new_obj['eccentricity'] = object.eccentricity

        new_obj = Segmentation.calc_labels_props(new_obj, imgs, create_lamin_donut)

        return new_obj

    @staticmethod
    def calc_labels_props(obj, imgs=None, create_lamin_donut=True):
        """
        Calculate extra labels properties

        :param object:
        :param imgs:
        :param create_lamin_donut:
        :return:
        """
        obj['circularity'] = obj['perimeter']/math.sqrt(obj['area'])

        obj['colour'] = rdm.randrange(1, 2**8)
        obj['nID'] = -1

        if imgs is not None:
            # get edge distance from bounding box
            edge_distances = [
                obj['bbox'][0], imgs.lamin.shape[0] - obj['bbox'][2],
                obj['bbox'][1], imgs.lamin.shape[1] - obj['bbox'][3]
            ]

            obj['edge_dist'] = min(edge_distances)

            # create lamin donut
            if create_lamin_donut is True:
                obj['lamin_donut_ratio'] = Segmentation.calc_lamin_donut(obj, imgs)

            # get lamin signal
            # ! very memory expensive !
            #lamin_signal = np.zeros_like(imgs.lamin)
            #for coord in new_obj['coords']:
            #    lamin_signal[coord[0], coord[1]] = imgs.lamin[coord[0], coord[1]]

            #new_obj['img'] = lamin_signal[new_obj['bbox'][0]:new_obj['bbox'][2],
            #                              new_obj['bbox'][1]:new_obj['bbox'][3]]

        return obj

    @staticmethod
    def calc_lamin_donut(label_props, imgs):
        """
        Calculate lamin donut based on lamin signal

        :param label_props:
        :param imgs:
        :return:
        """
        donut_props = Segmentation.calc_donut_for_label(label_props, imgs,
                                                 dilero_param=cfg.criteria_select_lamin_donut_ring)
        return donut_props['donut_ratio']

    def get_sorted_prop_list(self, param, is_nuclei=False):
        """
        Return properties list of a specific parameter

        :param param:
        :param is_nuclei:
        :return:
        """
        # sort values by param
        self.sort_prop_list(param, is_nuclei)

        # list for param
        param_list = list()

        # go through sorted label props
        for prop in self.sorted_probs:
            param_list.append(prop[param])

        return param_list

    def sort_prop_list(self, param, is_nuclei=False):
        """
        Sort properties list according to parameter

        :param param:
        :param is_nuclei:
        :return:
        """
        # flatten stack
        if is_nuclei is False:
            to_be_sorted = ImageProcessing.flatten_stack(self.stacks.labels_props)
        else:
            to_be_sorted = self.nuclei

        # go through labels props
        to_be_sorted.sort(key=operator.itemgetter(param))
        #to_be_sorted = sorted(to_be_sorted, key=lambda k: k[param])

        # set sorted labels
        self.sorted_probs = to_be_sorted

    @staticmethod
    def add_label_to_img(ind_label, blank_img, colour=None, max_area=-1):
        """
        Add label to image

        :param ind_label:
        :param img:
        :param colour:
        :param max_area:
        :return:
        """
        img = blank_img.copy()

        if ind_label != None and (max_area < 0 or ind_label['area'] < max_area):
            for coords in ind_label['coords']:
                if colour is None:
                    ind_colour = ind_label['colour']
                else:
                    ind_colour = colour
                img[coords[0], coords[1]] = ind_colour

        return img

    @staticmethod
    def add_labels_to_stack(labels, blank_stack, colour=None):
        """
        Add label to image

        :param labels:
        :param blank_stack:
        :param colour:
        :param max_area:
        :return:
        """
        stack = blank_stack.copy()

        # go through labels
        for z, ids in enumerate(labels):
            for cur_id, cur_params in enumerate(ids):
                # take a label
                label = labels[z][cur_id]

                if label is not None:
                    for coords in label['coords']:
                        if colour is None:
                            ind_colour = label['colour']
                        else:
                            ind_colour = colour

                        stack[z][coords[0], coords[1]] = ind_colour

        return stack

    @staticmethod
    def add_nuclei_to_stack(nuclei, stack, nucleus_value=None):
        """
        Add nuclei to a stack based on its coordinates

        :param nucleus:
        :param stack:
        :return:
        """
        # go through
        for nucleus in nuclei:
            stack = Segmentation.add_nucleus_to_stack(nucleus, stack, nucleus_value=nucleus_value)

        return stack

    @staticmethod
    def add_nucleus_to_stack(nucleus, stack, nucleus_value=None):
        """
        Add a single nucleus to a stack

        :param nucleus:
        :param stack:
        :return:
        """
        # colour choice
        if nucleus_value is None:
            nucleus_value = nucleus['colour']
        elif nucleus_value < 0:
            nucleus_value = rdm.randrange(0, 255)

        # iterate through the coordinates
        if type(nucleus['coords']) == np.ndarray:
            for coords in nucleus['coords']:
                stack[coords[0]][coords[1], coords[2]] = nucleus_value
        else:
            for index, coords in enumerate(nucleus['coords']):
                for i, frame in enumerate(coords[1]):
                    stack[coords[0]][frame[0], frame[1]] = nucleus_value

        return stack

    def get_results_dir(self, exp_id=None):
        """
        Return result dirs for root, stacks and nuclei file

        :param base:
        :return:
        """
        dirs = Struct()

        if exp_id is None:
            exp_id = self.image_info['ID']

        dirs.results = ImageHandler.create_dir(cfg.path_results + exp_id + cfg.OS_DEL)

        # create merge directory
        dirs.merge = ImageHandler.create_dir(dirs.results + cfg.path_merge)

        # create directories for stacks
        dirs.stacks_raw = ImageHandler.create_dir(dirs.results + cfg.path_results_stacks_raw)
        dirs.stacks_corr = ImageHandler.create_dir(dirs.results + cfg.path_results_stacks_corr)
        dirs.stacks_merge = ImageHandler.create_dir(dirs.results + cfg.path_merge_stacks)

        # path to nuclei file
        dirs.nuclei_raw = dirs.results + cfg.file_nuclei_raw
        dirs.nuclei_corr = dirs.results + cfg.file_nuclei_corr

        # path to nuclei data
        dirs.nuclei_data = ImageHandler.create_dir(dirs.results + cfg.path_nuclei_data)

        # path to lookup table
        dirs.lookup_raw = dirs.results + cfg.file_lookup_raw
        dirs.lookup_corr = dirs.results + cfg.file_lookup_corr

        # path to props
        dirs.labels_props_raw = dirs.results + cfg.file_labels_props_raw
        dirs.labels_props_corr = dirs.results + cfg.file_labels_props_corr

        # create directory for corrections
        dirs.corr = ImageHandler.create_dir(dirs.results + cfg.path_corrections)
        dirs.corr_stacks = ImageHandler.create_dir(dirs.results + cfg.path_correction_stacks)

        # create directory for classifier
        ImageHandler.create_dir(dirs.results + cfg.path_classifier)
        dirs.classifier = dirs.results + cfg.file_classifier

        # tmp dir
        dirs.tmp = ImageHandler.create_dir(dirs.results + cfg.path_tmp)

        return dirs

    def save(self, force_labels_props_raw=False, force_nuclei_raw=False,
             lamin_stack=None, labels_stack=None, nuclei_stack=None,
             force_nuclei_stack_rebuild=False):
        """
        Store stacks and nuclei

        :param force_labels_props_raw:
        :param force_nuclei_raw:
        :param lamin_stack:
        :param labels_stack:
        :param nuclei_stack:
        :return:
        """
        print('save segmentation')

        dirs = self.get_results_dir()

        # determine whether to save correction or raw
        save_labels_props_corr = False
        save_nuclei_corr = False

        if os.path.isfile(dirs.labels_props_raw) and force_labels_props_raw is False:
            save_labels_props_corr = True

        if os.path.isfile(dirs.nuclei_raw) and force_nuclei_raw is False:
            save_nuclei_corr = True

        # save correction for labels props?
        if save_labels_props_corr is True:
            print('save labels props correction')
            labels_props_file = dirs.labels_props_corr
        else:
            labels_props_file = dirs.labels_props_raw

        # corrections made for nucleus?
        if save_nuclei_corr is True:
            print('save nuclei correction')
            stacks_path = dirs.stacks_corr
            nuclei_file = dirs.nuclei_corr
            lookup_file = dirs.lookup_corr

            # create non-nuclei from nuclei
            non_nuclei = Segmentation.create_non_nuclei_stack(self.stacks)

            # save non_nuclei as tiff
            ImageHandler.save_stack_as_tiff(non_nuclei, dirs.stacks_corr + cfg.file_stack_non_nuclei)
        else:
            stacks_path = dirs.stacks_raw
            nuclei_file = dirs.nuclei_raw
            lookup_file = dirs.lookup_raw

        # set stacks new?
        if lamin_stack is not None:
            self.stacks.lamin = lamin_stack

        if labels_stack is not None:
            self.stacks.labels = labels_stack

        if nuclei_stack is not None:
            self.stacks.nuclei = nuclei_stack

        # save labels props
        if hasattr(self.stacks, 'labels_props') and self.stacks.labels_props is not None:
            # save object
            with open(labels_props_file, "wb") as fin:
                pickle.dump(self.stacks.labels_props, fin)

        # save nuclei
        if hasattr(self, 'nuclei') and self.nuclei is not None:
            # save object
            with open(nuclei_file, "w+b") as fin:
                pickle.dump(self.nuclei, fin)

            # save lookup
            with open(lookup_file, "wb") as fin:
                pickle.dump(self.lookup, fin)

            # create a stack for nuclei?
            if not hasattr(self.stacks, 'nuclei') or force_nuclei_stack_rebuild is True:
                self.stacks.nuclei = self.add_nuclei_to_stack(self.nuclei, np.zeros_like(self.stacks.lamin))

            # how many nuclei were identified?
            print('Identified nuclei (%s): %i' % (nuclei_file, len(self.nuclei)))

        if hasattr(self.stacks, 'lamin'):
            ImageHandler.save_stack_as_tiff(self.stacks.lamin, stacks_path + cfg.file_stack_lamin)
        if hasattr(self.stacks, 'labels'):
            ImageHandler.save_stack_as_tiff(self.stacks.labels, stacks_path + cfg.file_stack_labels)
        if hasattr(self.stacks, 'nuclei'):
            ImageHandler.save_stack_as_tiff(self.stacks.nuclei, stacks_path + cfg.file_stack_nuclei)

    def load(self, force_props_load=False, force_nuclei_load=False):
        """
        Reload stacks and properties of labels and nuclei

        :param force_props_load:
        :param force_nuclei_load:
        :return:
        """
        # load stacks
        dirs = self.get_results_dir()

        # load nuclei - get correction if present
        if os.path.isfile(dirs.nuclei_corr) and force_nuclei_load is False:
            print('load nuclei correction')
            self.nuc_corr_loaded = True
            stacks_path = dirs.stacks_corr
            nuclei_file = dirs.nuclei_corr
            lookup_file = dirs.lookup_corr
        else:
            self.nuc_corr_loaded = False
            stacks_path = dirs.stacks_raw
            nuclei_file = dirs.nuclei_raw
            lookup_file = dirs.lookup_raw

        if os.path.isfile(stacks_path + cfg.file_stack_labels):
            self.stacks.labels = ImageHandler.load_tiff_as_stack(stacks_path + cfg.file_stack_labels)
        if os.path.isfile(stacks_path + cfg.file_stack_nuclei):
            self.stacks.nuclei = ImageHandler.load_tiff_as_stack(stacks_path + cfg.file_stack_nuclei)

        # load label props
        labels_props_file = None
        if os.path.isfile(dirs.labels_props_corr) and force_props_load is False:
            print('load labels props correction')
            self.labels_props_corr_loaded = True
            labels_props_file = dirs.labels_props_corr
        elif os.path.isfile(dirs.labels_props_raw) and force_props_load is False:
            self.labels_props_corr_loaded = False
            labels_props_file = dirs.labels_props_raw

        if labels_props_file is not None:
            with open(labels_props_file, "rb") as fin:
                self.stacks.labels_props = pickle.load(fin)
        else:
            self.get_label_props()

        if os.path.isfile(nuclei_file):
            with open(nuclei_file, "rb") as fin:
                self.nuclei = pickle.load(fin)
            with open(lookup_file, "rb") as fin:
                self.lookup = pickle.load(fin)

        self.get_raw_lookup()

    @staticmethod
    def create_non_nuclei_stack(stacks):
        """
        Create non-nuclei stack

        :return:
        """
        # substract nuclei from lamin
        non_nuclei = stacks.lamin.copy()

        non_nuclei[stacks.nuclei > 0] = 0

        return non_nuclei

    def get_nucleus_by_id(self, nID):
        """
        Cycle through nuclei and return the nucleus with a specific ID

        :param nID:
        :return:
        """
        nucleus = None

        for cur_nucleus in self.nuclei:
            if cur_nucleus['nID'] == nID:
                nucleus = cur_nucleus
                break

        return nucleus

    def get_raw_nucleus_by_id(self, nID):
        """
        Get raw nucleus, reload if necessary

        :param nID:
        :return:
        """
        # reload if correction loaded
        raw_nuclei = self.get_raw_nuclei()

        nucleus = None

        for cur_nucleus in raw_nuclei:
            if cur_nucleus['nID'] == nID:
                nucleus = cur_nucleus
                break

        return nucleus

    def get_raw_labels_props(self):
        """
        Reload raw labels props if necessary

        :return:
        """
        if self.labels_props_corr_loaded is True:
            if self.stacks.raw_labels_props is None:
                # load raw labels props
                with open(self.get_results_dir().labels_props_raw, "rb") as fin:
                    self.stacks.raw_labels_props = pickle.load(fin)
        else:
            self.stacks.raw_labels_props = self.stacks.labels_props

        return self.stacks.raw_labels_props

    def get_raw_nuclei(self):
        """
        Reload raw nuclei if necessary

        :return:
        """
        if self.nuc_corr_loaded is True:
            if self.raw_nuclei is None:
                # load raw nuclei
                with open(self.get_results_dir().nuclei_raw, "rb") as fin:
                    self.raw_nuclei = pickle.load(fin)
        else:
            self.raw_nuclei = self.nuclei

        return self.raw_nuclei

    def get_raw_lookup(self):
        """
        Reload raw lookup if necessary

        :return:
        """
        if self.nuc_corr_loaded is True:
            if self.raw_lookup is None:
                with open(self.get_results_dir().lookup_raw, "rb") as fin:
                    self.raw_lookup = pickle.load(fin)
        else:
            self.raw_lookup = self.lookup

        return self.raw_lookup

    def get_nucleus_by_pos(self, pos):
        """
        Return nucleus which have the position

        :param pos:
        :return:
        """
        return self.get_nucleus_by_pos_in_nuclei(pos, self.lookup)

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

    def get_expanded_bbox_for_nucleus(self, nucleus):
        """
        Return expanded bbox for nucleus

        :param nucleus:
        :return:
        """
        nuc_bbox = nucleus['nuc_bbox'].copy()

        # add offset
        offset = cfg.nuclei_bbox_range

        for i in range(0, 3):
            nuc_bbox[i] -= offset

            if nuc_bbox[i] < 0:
                nuc_bbox[i] = 0

        for i in range(3, 6):
            nuc_bbox[i] += offset

            if nuc_bbox[i] >= self.stacks.nuclei.shape[i - 3]:
                nuc_bbox[i] = self.stacks.nuclei.shape[i - 3]

        return nuc_bbox

    def get_nuclei_by_pos_range(self, pos):
        """
        Return nuclei which are in the position range

        :param pos:
        :return:
        """
        return self.get_nuclei_by_pos_range_in_nuclei(pos, self.lookup)

    def get_raw_label_by_pos(self, pos):
        """
        Return label which has the position from the raw file

        :param pos:
        :return:
        """
        return Segmentation.get_label_by_pos_in_props(pos, self.get_raw_labels_props())

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

    @staticmethod
    def get_label_by_pos_in_props(pos, props):
        """
        Get label by position in props list

        :param pos:
        :param pops:
        :return:
        """
        ret_label = None

        # go through label and check if the position
        # is part of the coordinates
        if pos[0] in range(0, len(props)):
            # get z
            props_z = props[pos[0]]

            # cycle through all props in z
            for cur_props in props_z:
                # go through coords
                for cur_pos in cur_props['coords']:
                    # do X and Y match?
                    if cur_pos[0] == pos[1] and cur_pos[1] == pos[2]:
                        ret_label = cur_props
                        break

        return ret_label

    def get_nucleus_by_pos_in_nuclei(self, pos, lookup):
        """
        Get nucleus by position in nuclei list

        :param pos:
        :param lookup:
        :return:
        """
        nucleus = None

        # lookup nucleus
        nID = lookup.get_nID_by_coords_pos(pos)
        nucleus = self.get_nucleus_by_id(nID)

        # go through nuclei and check if the position
        # is part of the coordinates
        #for cur_nucleus in nuclei:
        #    # go through coords
        #    for cur_coord in cur_nucleus['coords']:
        #        # does Z match?
        #        if cur_coord[0] == pos[0]:
        #            for cur_pos in cur_coord[1]:
        #                # do X and Y match?
        #                if cur_pos[0] == pos[1] and cur_pos[1] == pos[2]:
        #                    nucleus = cur_nucleus
        #                    break

        return nucleus

    def get_nuclei_by_pos_range_in_nuclei(self, pos_range, lookup):
        """
        Return nuclei in range of position

        :param pos_range:
        :param nuclei:
        :return:
        """
        nuclei = list()

        nIDs = lookup.get_nIDs_by_coords_pos_range(pos_range)

        for nID in nIDs:
            nuclei.append(self.get_nucleus_by_id(nID))

        # go through nuclei and check if the position
        # is part of the coordinates
        #for cur_nucleus in nuclei_list:
        #    pos_in_nucleus = False

        #    # go through coords
        #    for cur_coord in cur_nucleus['coords']:
        #        # does Z match?
        #        if cur_coord[0] in range(pos_range[0], pos_range[3] + 1):
        #            for cur_pos in cur_coord[1]:
        #                # do X and Y match?
        #                if cur_pos[0] in range(pos_range[1], pos_range[4] + 1):
        #                    if cur_pos[1] in range(pos_range[2], pos_range[5] + 1):
        #                        pos_in_nucleus = True
        #                        break
        #
        #    if pos_in_nucleus is True:
        #        nuclei.append(cur_nucleus)

        return nuclei

    def is_nucleus_in_nuclei(self, nID):
        """
        Look if the nucleus is in the nuclei list

        :param nucleus:
        :return:
        """
        in_list = False

        if self.get_nucleus_by_id(nID) is not None:
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

    def update(self, save=True, calc_nuclei_params=True,
               force_labels_props_raw=False, force_nuclei_raw=False):
        """
        Update segmentation, for example when corrections have been applied

        :param force_labels_props_raw:
        :param force_nuclei_raw:
        :return:
        """
        # update nucleus stack
        self.stacks.nuclei = Segmentation.add_nuclei_to_stack(
            self.nuclei, np.zeros_like(self.stacks.lamin))

        # update nuclei params
        if calc_nuclei_params is True:
            self.calc_nuclei_params()

        # save updates
        if save is True:
            self.save(force_labels_props_raw=force_labels_props_raw, force_nuclei_raw=force_nuclei_raw)

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

    def set_stacks_labels(self, labels_stack):
        """
        Set labels stack

        :param labels_stack:
        :return:
        """
        self.stacks.labels = labels_stack
