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

from ast import literal_eval

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
from storage.nuclei import Nuclei

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
        self.stacks.raw_labels_props = None
        self.nuc_corr_loaded = False
        self.labels_props_corr_loaded = False

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
                self.stacks.nuclei = self.nuclei.add_nuclei_to_stack(
                    np.zeros_like(self.stacks.labels), only_accepted=True)

    def apply_filters(self):
        """
        Apply filters to planes

        :return:
        """

        return ImageProcessing.apply_filters_by_image_info(self.image_info, self.stacks.lamin)

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

    def is_param_in_criteria_range(self, cur_object, param):
        """
        Is the label in range of parameters?

        :param cur_object:
        :param param:
        :return:
        """
        in_range = True

        if cur_object is not None:
            if param in cur_object:
                # too high?
                if self.nuc_criteria[param]['MAX'] is not None:
                    if self.nuc_criteria[param]['MAX'] > 0 \
                        and cur_object[param] > self.nuc_criteria[param]['MAX']:
                        in_range = False

                # too low?
                if self.nuc_criteria[param]['MIN'] is not None:
                    if self.nuc_criteria[param]['MIN'] > 0 \
                        and cur_object[param] <= self.nuc_criteria[param]['MIN']:
                        in_range = False
        else:
            in_range = False

        return in_range

    def merge_labels(self):
        """
        Merge potential nuclei based on their selection criteria
        :return:
        """
        # delete labels that do not fit the selection criteria
        self.filter_planes()

        # store nuclei in a central list
        nuclei = Nuclei(self)

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

                # create a negative nucleus ID
                nID = -1

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
                                        if nID < 0:
                                            nID = nuclei.create_nucleus(z, cur_label)
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
                                                    nuclei.add_to_nucleus(i, AND_labels_props, nID)

                                                    if len(merge_skipped) > 0:
                                                        skipped_label = merge_skipped.pop(0)
                                                        merge_stack[skipped_label[0]][skipped_label[1]] = None

                                            nuclei.add_to_nucleus(y, next_label, nID)

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

        self.set_nuclei(nuclei)

        # update nuclei params
        self.update(save=False)

        # perform post processing
        # Do not perform post-process - rather do this within the actual processing filters
        #self.nuclei.postprocess_for_nuclei(self.stacks.nuclei[0])

    def remerge_nucleus_part(self, nID, new_start, new_stop,
                             merge_depth=False, force_raw_labels_props=False):
        """
        Remerge nucleus part

        :param nID:
        :param nucleus_start:
        :param corr_plane:
        :param merge_depth:
        :return:
        """
        # adjust merge range
        # FIX: for nuclei whose ends are not merged properly - take the middle
        nucleus_coords = self.nuclei.get_nucleus_coords(nID)
        new_start = nucleus_coords[int(len(list(np.unique(nucleus_coords[:, 0]))) / 2), 0]

        range_start = int(new_start)
        range_stop = int(new_stop)

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
        cur_label = self.build_label_from_nucleus(nID, range_start)[0]
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
                                    self.nuclei.add_to_nucleus(i, AND_labels_props, nID,
                                                               remove_before=True)

                            #print('add nucleus %i' % y)
                            self.nuclei.add_to_nucleus(y, next_label, nID,
                                                       remove_before=True)

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
                self.nuclei.add_to_nucleus(i, cur_label, nID,
                                           remove_before=True)

        return nID

    def delete_nucleus_part(self, nID, new_start, new_stop):
        """
        Delete part of the nucleus

        :param nID:
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
            self.nuclei.del_from_nucleus(y, nID)

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

    def build_label_from_nucleus(self, nID, z=-1):
        """
        Build label from nucleus

        :param nID:
        :param z:
        :return:
        """
        labels_probs = list()

        # get all properties for a label
        nucleus_centroids = self.nuclei.get_nucleus_centroids(nID, z=z)
        nucleus_areas = self.nuclei.get_nucleus_areas(nID, z=z)
        nucleus_coords = self.nuclei.get_nucleus_coords(nID, z=z)
        nucleus_perimeters = self.nuclei.get_nucleus_perimeters(nID, z=z)
        nucleus_bboxes = self.nuclei.get_nucleus_bboxes(nID, z=z)

        # go through the properties and add as dictionary
        if z >= 0:
            z_planes = (z, )
        else:
            z_planes = list(nucleus_centroids[:, 0])

        # go through all the planes of the nucleus
        for i, z in enumerate(z_planes):
            labels_probs.append(dict())

            # get current props
            if len(nucleus_centroids.shape) > 1:
                curr_centroids = nucleus_centroids[i]
                curr_areas = nucleus_areas[i]
                curr_perimeters = nucleus_perimeters[i]
                curr_bboxes = nucleus_bboxes[i]
                curr_coords = nucleus_coords[nucleus_coords[:, 0] == z]
            else:
                curr_centroids = nucleus_centroids
                curr_areas = nucleus_areas
                curr_perimeters = nucleus_perimeters
                curr_bboxes = nucleus_bboxes
                curr_coords = nucleus_coords

            # set the properties
            labels_probs[-1]['centroid'] = (curr_centroids[1], curr_centroids[2])
            labels_probs[-1]['area'] = curr_areas[1]
            labels_probs[-1]['perimeter'] = curr_perimeters[1]
            labels_probs[-1]['bbox'] = (curr_bboxes[1], curr_bboxes[2],
                                        curr_bboxes[3], curr_bboxes[4])

            # build coords array
            coord_list = list()
            for coord in curr_coords:
                coord_list.append((coord[1], coord[2]))
            coord_array = np.array(coord_list)

            labels_probs[-1]['coords'] = coord_array

            # calculate extra parameters
            imgs = Image()
            imgs.lamin = self.stacks.lamin[z]
            imgs.dapi = self.stacks.dapi[z]
            imgs.membrane = self.stacks.membrane[z]
            labels_probs[-1] = Segmentation.calc_labels_props(labels_probs[-1], imgs)

        return labels_probs

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

        if labels_props_loaded is False and hasattr(self.stacks, 'labels'):
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

            # TODO create lamin donut
            # if create_lamin_donut is True:
            #     obj['lamin_donut_ratio'] = Segmentation.calc_lamin_donut(obj, imgs)

            # get lamin signal
            # ! very memory expensive !
            #lamin_signal = np.zeros_like(imgs.lamin)
            #for coord in new_obj['coords']:
            #    lamin_signal[coord[0], coord[1]] = imgs.lamin[coord[0], coord[1]]

            #new_obj['img'] = lamin_signal[new_obj['bbox'][0]:new_obj['bbox'][2],
            #                              new_obj['bbox'][1]:new_obj['bbox'][3]]

        return obj

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

    def get_raw_label_by_pos(self, pos):
        """
        Return label which has the position from the raw file

        :param pos:
        :return:
        """
        return Segmentation.get_label_by_pos_in_props(pos, self.get_raw_labels_props())

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

    def get_sorted_prop_list(self, param):
        """
        Return properties list of a specific parameter

        :param param:
        :return:
        """
        # sort values by param
        self.sort_prop_list(param)

        # list for param
        param_list = list()

        # go through sorted label props
        for prop in self.sorted_probs:
            param_list.append(prop[param])

        return param_list

    def sort_prop_list(self, param):
        """
        Sort properties list according to parameter

        :param param:
        :return:
        """
        # sort labels
        to_be_sorted = ImageProcessing.flatten_stack(self.stacks.labels_props)

        # sort labels props
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

        # create directories for nuclei params
        dirs.nuclei_params_raw = ImageHandler.create_dir(dirs.results + cfg.path_results_nucleus_params_raw)
        dirs.nuclei_params_corr = ImageHandler.create_dir(dirs.results + cfg.path_results_nucleus_params_corr)

        # path to nuclei file
        dirs.nuclei_raw = dirs.results + cfg.file_nuclei_raw
        dirs.nuclei_corr = dirs.results + cfg.file_nuclei_corr

        # path to nuclei data
        dirs.nuclei_data = ImageHandler.create_dir(dirs.results + cfg.path_nuclei_data)

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

        if os.path.isfile(dirs.stacks_raw + cfg.file_stack_nuclei) and force_nuclei_raw is False:
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
            nuclei_params_path = dirs.nuclei_params_corr

            # create non-nuclei from nuclei
            non_nuclei = Segmentation.create_non_nuclei_stack(self.stacks)

            # save non_nuclei as tiff
            ImageHandler.save_stack_as_tiff(non_nuclei, dirs.stacks_corr + cfg.file_stack_non_nuclei)
        else:
            stacks_path = dirs.stacks_raw
            nuclei_params_path = dirs.nuclei_params_raw

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
            # save nuclei data
            self.nuclei.save(nuclei_params_path)

            # create a stack for nuclei?
            if not hasattr(self.stacks, 'nuclei') or force_nuclei_stack_rebuild is True:
                self.stacks.nuclei = self.nuclei.add_nuclei_to_stack(
                    np.zeros_like(self.stacks.lamin), only_accepted=True)

            # how many nuclei were identified?
            print('Identified nuclei (%s): %i' % (nuclei_params_path, len(self.nuclei.get_nIDs(only_accepted=True))))

        if hasattr(self.stacks, 'lamin'):
            ImageHandler.save_stack_as_tiff(self.stacks.lamin, stacks_path + cfg.file_stack_lamin)
        if hasattr(self.stacks, 'labels'):
            ImageHandler.save_stack_as_tiff(self.stacks.labels, stacks_path + cfg.file_stack_labels)
        if hasattr(self.stacks, 'nuclei'):
            ImageHandler.save_stack_as_tiff(self.stacks.nuclei, stacks_path + cfg.file_stack_nuclei)
        if hasattr(self.stacks, 'membin'):
            ImageHandler.save_stack_as_tiff(self.stacks.membin, stacks_path + cfg.file_stack_membin)

    def load(self, force_props_load=False, force_nuclei_load=False, force_extras_recalc=False):
        """
        Reload stacks and properties of labels and nuclei

        :param force_props_load:
        :param force_nuclei_load:
        :return:
        """
        # load stacks
        dirs = self.get_results_dir()

        # load nuclei - get correction if present
        if os.path.isfile(dirs.stacks_corr + cfg.file_stack_nuclei) and force_nuclei_load is False:
            print('load nuclei correction')
            self.nuc_corr_loaded = True
            stacks_path = dirs.stacks_corr
            nuclei_params_path = dirs.nuclei_params_corr
        else:
            self.nuc_corr_loaded = False
            stacks_path = dirs.stacks_raw
            nuclei_params_path = dirs.nuclei_params_raw

        if os.path.isfile(stacks_path + cfg.file_stack_labels):
            self.stacks.labels = ImageHandler.load_tiff_as_stack(stacks_path + cfg.file_stack_labels)
        if os.path.isfile(stacks_path + cfg.file_stack_nuclei):
            self.stacks.nuclei = ImageHandler.load_tiff_as_stack(stacks_path + cfg.file_stack_nuclei)
        if os.path.isfile(stacks_path + cfg.file_stack_membin):
            self.stacks.membin = ImageHandler.load_tiff_as_stack(stacks_path + cfg.file_stack_membin)

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

        # load nuclei data
        if os.path.isdir(nuclei_params_path):
            self.set_nuclei(Nuclei(self))
            self.nuclei.load(nuclei_params_path, force_extra_recalc=force_extras_recalc)

    def update(self, save=True, calc_nuclei_params=True,
               force_labels_props_raw=False, force_nuclei_raw=False):
        """
        Update segmentation, for example when corrections have been applied

        :param force_labels_props_raw:
        :param force_nuclei_raw:
        :return:
        """
        # update nucleus stack
        self.stacks.nuclei = self.nuclei.add_nuclei_to_stack(np.zeros_like(self.stacks.lamin), only_accepted=True)

        # update nuclei params
        if calc_nuclei_params is True:
            self.nuclei.calc_nuclei_params()

        # save updates
        if save is True:
            self.save(force_labels_props_raw=force_labels_props_raw, force_nuclei_raw=force_nuclei_raw)

    def set_stacks_labels(self, labels_stack):
        """
        Set labels stack

        :param labels_stack:
        :return:
        """
        self.stacks.labels = labels_stack

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

    def set_nuclei(self, nuclei):
        """
        Set nuclei tables

        :param nuclei:
        :return:
        """
        self.nuclei = nuclei

    def has_membin(self):
        """
        Check if binary membrane stack has been set

        :return:
        """
        membin = False

        if hasattr(self.stacks, 'membin') and self.stacks.membin is not None:
            membin = True

        return membin

    def create_membin(self):
        """
        Create binary version of the membrane channel

        :return:
        """
        # define processing steps to create binary membrane mask
        processing_steps = [
            ['EQU'],
            ['THR', 'OTSU', 100, '3D'],
            ['FILL'],
            ['OPN', 'bin', 3],
            ['CONV_BIT', 16, '3D']
        ]

        # apply processing steps
        self.stacks.membin = ImageProcessing.apply_filters(
            processing_steps, self.stacks.membrane, verbose=cfg.general_verbose)

    def rot_vector(self, to_rotate, degree=0, axis=0):
        """
        Rotate a vector around an axis
        axis: 0=z; 1=y; 2=x

        :param to_rotate:
        :param degree:
        :param axis:
        :return:
        """

        # create roation matrices
        # !! order for vectors in Z, Y, X !!
        # hence all matrices are transposed

        rot_z = np.transpose(
            np.array([
                [math.cos(degree), -math.sin(degree), 0],
                [math.sin(degree), math.cos(degree), 0],
                [0, 0, 1]
            ])
        )

        rot_y = np.transpose(
            np.array([
                [math.cos(degree), 0, math.sin(degree)],
                [0, 1, 0],
                [-math.sin(degree), 0, math.cos(degree)]
            ])
        )

        rot_x = np.transpose(
            np.array([
                [1, 0, 0],
                [0, math.cos(degree), -math.sin(degree)],
                [0, math.sin(degree), math.cos(degree)]
            ])
        )

        rotated = None

        # transpose array to vector
        to_rotate = np.transpose([to_rotate])

        # apply rotation
        if axis == 0:
            rotated = np.dot(rot_z, to_rotate)
        elif axis == 1:
            rotated = np.dot(rot_y, to_rotate)
        elif axis == 2:
            rotated = np.dot(rot_x, to_rotate)

        # transpose result back to array
        rotated = rotated.flatten()

        return rotated
