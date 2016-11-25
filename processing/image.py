"""
Processing steps for image
"""
import csv
import re

import storage.config as cfg
from frontend.figures.plot import Plot
from collections import OrderedDict

from storage.image import ImageHandler

# import all filter classes
from processing.filter import *

class ImageProcessing:

    @staticmethod
    def apply_filters_by_image_info(image_info, image_data):
        """
        Apply defined filters for a given image

        :param image_info:
        :param image_data:
        :return:
        """
        # retrieve filters from configuration
        processing_steps = ImageProcessing.get_filters(image_info)

        # apply filters
        filtered_image = ImageProcessing.apply_filters(processing_steps, image_data, verbose=cfg.general_verbose)

        return filtered_image

    @staticmethod
    def apply_filters(processing_steps, image_data, verbose=False):
        """
        Apply processing steps to image

        :param processing_steps:
        :param image_data:
        :return:
        """
        # read filter mapping
        filter_mapping = ImageProcessing.get_filter_mapping()

        # filtered image
        filtered_image = image_data.copy()

        # adjust if only one frame
        is_two_dim = False

        if len(filtered_image.shape) < 3:
            filtered_image = [filtered_image]
            is_two_dim = True

        # show intermediate results for each processing step?
        show_results = False

        # go through filters and apply
        if verbose is True:
            print('Processing steps: %i' % len(processing_steps))
        for step in processing_steps:
            # ignore specific keywords
            if step[0] == 'SHOW':
                show_results = True
                show_range = range(step[2], step[3], step[4])
                results = list()

            # get mapping
            filter_map = filter_mapping[step[0]]

            # initialise class
            filter_class = eval(filter_map['class'])
            filter = filter_class()

            # build parameter list
            params = dict()

            if verbose is True:
                print('Apply %s:' % filter_map['class'])
            for param in range(1, len(step)):
                if verbose is True:
                    print('\tparam %s:%s' % (filter_map['params'][param - 1], step[param]))
                params[filter_map['params'][param - 1]] = step[param]

            # is 3D required for filter?
            if '3D' in params.keys() and params['3D'] == '3D':
                filtered_image = filter.apply(filtered_image, params)
            else:
                # go through all slices individually
                for z, frame in enumerate(filtered_image):
                    filtered_image[z] = filter.apply(frame, params)

            if show_results:
                # prepare result for plotting
                results = Plot.prepare_output(
                    results, filtered_image.copy(),
                    '%s: %s' % (step[0], params['size'] if 'size' in params.keys() else 'NaN'),
                    'hot')

        if show_results:
            # show results of processing steps
            Plot.view_stacks(results[0], 1, show_range, results[1], results[2], scale=2)

        # prepare 2D for return
        if is_two_dim is True:
            filtered_image = filtered_image[0]

        return filtered_image

    @staticmethod
    def get_filters(image):
        """
        Retrieve filters for image from configuration

        :param image:
        :return:
        """
        # processing steps for image
        processing_steps = list()

        with open(cfg.file_processing, 'r') as csvfile:
            image_reader = csv.reader(csvfile,
                                      delimiter=cfg.CSV_DEL,
                                      quotechar=cfg.CSV_QUOT)
            # find filters for image
            for row in image_reader:
                if row[0] == ImageHandler.extract_expnum_from_ID(image['ID']):
                    for filter in range(1, len(row)):
                        # get filter and parameters
                        split_filter = row[filter].split(cfg.PROC_PARAMS)
                        # build the filter and parameter list
                        steps = list()
                        steps.append(split_filter[0])

                        # Does the filter have parameters?
                        if len(split_filter) > 1:
                            # split parameters
                            split_params = split_filter[1].split(cfg.PROC_PARAMS_DEL)

                            for param in split_params:
                                # is parameter a digit?
                                if param.isdigit():
                                    param = int(param)

                                steps.append(param)

                        processing_steps.append(steps)

                    break

            csvfile.close()

        return processing_steps

    @staticmethod
    def get_filter_mapping():
        """
        Retrieve filter mapping

        :return:
        """
        # filter mapping
        filter_mapping = OrderedDict()

        with open(cfg.file_filter_mapping, 'r') as csvfile:
            filter_reader = csv.reader(csvfile,
                                      delimiter=cfg.CSV_DEL,
                                      quotechar=cfg.CSV_QUOT)
            # go through mapping and create dictionary per keyword
            for row in filter_reader:
                if not re.match('^#.*', row[0]):
                    # go through parameters and add them to the list
                    params = list()

                    for param in range(2, len(row)):
                        params.append(row[param])

                    filter_mapping[row[0]] = {
                        'class': row[1],
                        'params': params
                    }

            csvfile.close()

        return filter_mapping

    @staticmethod
    def get_index_of_filter(mapping, filter_to_find):
        """
        Return index of filter in ordered dictionary

        :param mapping:
        :param filter_to_find:
        :return:
        """
        index = 0

        counter = 0
        for filter in mapping:
            if filter == filter_to_find:
                index = counter
                break

            counter += 1

        return index

    @staticmethod
    def is_param_range(param):
        """
        Is the parameter a range?

        :param param:
        :return:
        """
        is_range = False

        if isinstance(param, str) and re.match('[0-9]*-[0-9]*', param):
            is_range = True

        return is_range

    @staticmethod
    def get_param_from_range(param_range):
        """
        Is the parameter a range?

        :param param_range:
        :return:
        """
        param = str(param_range[0]) + '-' + str(param_range[-1])

        return param

    @staticmethod
    def get_param_range(param):
        """
        If the param is a range return the range

        :param param:
        :return:
        """
        param_range = None

        if ImageProcessing.is_param_range(param):
            range_limits = re.findall('[0-9]+', param)
            param_range = range(int(range_limits[0]), int(range_limits[1]) + 1)

        return param_range

    @staticmethod
    def flatten_stack(stack):
        """
        Flatten stack

        :param stack:
        :return:
        """
        stack_list = list()

        # go through list and append to new one
        for frame in stack:
            for x in frame:
                stack_list.append(x)

        return stack_list
