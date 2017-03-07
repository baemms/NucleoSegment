"""
Image loading and assignment of channels to types of stacks
"""
import csv
import re
import datetime
import os
import numpy as np

import storage.config as cfg
import skimage.io as io

class ImageHandler:

    """
    Types of channels
    """
    CHN_DAPI = 'DAPI'
    CHN_LAMIN = 'LAMIN'
    CHN_MEMBRANE = 'MEMBRANE'

    @staticmethod
    def save_image_info(image_info, del_ext=False):
        """
        Save image info

        :return:
        """
        # load csv
        csv_rows = list()
        with open(cfg.file_input, 'r') as csvfile:
            image_reader = csv.reader(csvfile,
                                      delimiter=cfg.CSV_DEL,
                                      quotechar=cfg.CSV_QUOT)
            for row in image_reader:
                csv_rows.append(row)

        # save to csv
        with open(cfg.file_input, 'w') as csvfile:
            image_writer = csv.writer(csvfile,
                                delimiter=cfg.CSV_DEL,
                                quotechar=cfg.CSV_QUOT)

            # write rows from before
            for row in csv_rows:
                write_row = False

                if row[0] != image_info['ID']:
                    write_row = True

                # do not write if there is an extension
                if del_ext is True:
                    if ImageHandler.extract_exp_from_ID(row[0]) == ImageHandler.extract_exp_from_ID(image_info['ID'])\
                            and ImageHandler.extract_num_from_ID(row[0]) == ImageHandler.extract_num_from_ID(image_info['ID'])\
                            and ImageHandler.extract_ext_from_ID(row[0]) >= 0:
                        write_row = False

                if write_row is True:
                    image_writer.writerow(row)

            image_writer.writerow(ImageHandler.prep_image_info_for_csv(image_info))

        csvfile.close()

    @staticmethod
    def prep_image_info_for_csv(image_info):
        """
        Transform image info into iterable object that can be written

        :param image_info:
        :return:
        """
        iter_info = list()

        iter_info.append(image_info['ID'])
        iter_info.append(image_info['exp'])
        iter_info.append(image_info['date'])
        iter_info.append(image_info['file'])
        iter_info.append(image_info['bound']['Z-min'])
        iter_info.append(image_info['bound']['Z-max'])
        iter_info.append(image_info['bound']['Y-min'])
        iter_info.append(image_info['bound']['Y-max'])
        iter_info.append(image_info['bound']['X-min'])
        iter_info.append(image_info['bound']['X-max'])

        for channel in image_info['channels']:
            iter_info.append(channel)

        return iter(iter_info)

    @staticmethod
    def update_exp_csv(exp_id, file_path, new_row):
        """
        Update line in csv file for an experiment

        :param exp_id:
        :param file_path:
        :param new_row:
        :return:
        """
        # load csv
        csv_rows = list()
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file,
                                      delimiter=cfg.CSV_DEL,
                                      quotechar=cfg.CSV_QUOT)
            for row in csv_reader:
                csv_rows.append(row)

        # save to csv
        with open(file_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file,
                                delimiter=cfg.CSV_DEL,
                                quotechar=cfg.CSV_QUOT)

            # write rows from before
            row_to_write = None
            new_written = False

            for row in csv_rows:
                if row[0] != exp_id:
                    row_to_write = row
                else:
                    row_to_write = new_row
                    new_written = True

                csv_writer.writerow(row_to_write)

            # add line if it was not already in the file
            if new_written is False:
                csv_writer.writerow(new_row)

        csv_file.close()

    @staticmethod
    def get_exp_csv(exp_id, file_path):
        """
        Get experiment information from file

        :param exp_id:
        :param file_path:
        :return:
        """
        vals = None

        # go through rows and find experiment
        if os.path.exists(file_path):
            with open(file_path, 'r') as csvfile:
                csv_reader = csv.reader(csvfile,
                                delimiter=cfg.CSV_DEL,
                                quotechar=cfg.CSV_QUOT)

                for row in csv_reader:
                    if row[0] == exp_id:
                        vals = row[1:]

        return vals

    @staticmethod
    def load_image_infos():
        """
        Load images from config/input.csv and assign to specific channels

        :return:
        """
        # images to be processed
        images = list()

        # read csv file in config
        with open(cfg.file_input, 'r') as csvfile:
            image_reader = csv.reader(csvfile,
                                      delimiter=cfg.CSV_DEL,
                                      quotechar=cfg.CSV_QUOT)

            for row in image_reader:
                if not re.match('^#.*', row[0]):
                    # get image boundaries
                    img_bound = {
                        'Z-min': int(row[4]), 'Z-max': int(row[5]),
                        'Y-min': int(row[6]), 'Y-max': int(row[7]),
                        'X-min': int(row[8]), 'X-max': int(row[9]),
                     }

                    # get channels
                    img_channels = list()
                    for channel in range(10, len(row)):
                        img_channels.append(row[channel])

                    # add to image list for processing
                    images.append({
                        'ID': row[0],
                        'exp': row[1],
                        'date': row[2],
                        'file': row[3],
                        'path': cfg.path_raw
                                + row[1] + cfg.OS_DEL
                                + row[2] + cfg.OS_DEL
                                + row[3],
                        'bound': img_bound,
                        'channels': img_channels
                        })

            csvfile.close()

        return images

    @staticmethod
    def num_images_from_exp(image_infos, id):
        """
        How many images are already in the input?

        :param image_infos:
        :param id:
        :return:
        """
        # go through list and remember how many images
        # are already given as input from this experiment
        num_imgs = 0

        for info in image_infos:
            # skip if the ID is revision
            if not ImageHandler.is_revision_by_ID(info['ID']):
                if ImageHandler.extract_exp_from_ID(info['ID']) == id:
                    num_imgs += 1

        return num_imgs

    @staticmethod
    def create_new_id_for_exp(image_infos, id):
        """
        Get

        :param image_infos:
        :param id:
        :return:
        """
        # go through image info and extract the highest ID
        highest_id = -1

        for info in image_infos:
            # skip if the ID is revision
            if not ImageHandler.is_revision_by_ID(info['ID']):
                if ImageHandler.extract_exp_from_ID(info['ID']) == id:
                    cur_id = ImageHandler.extract_num_from_ID(info['ID'])
                    if cur_id > highest_id:
                        highest_id = cur_id

        # increase id and return
        return (highest_id + 1)

    @staticmethod
    def is_revision_by_ID(id):
        """
        Is the current ID a revision?

        :param id:
        :return:
        """
        is_rev = False

        if re.match('^.*-r[0-9]*$', id) is not None:
            is_rev = True

        return is_rev

    @staticmethod
    def extract_parent_from_ID(id):
        """
        Returns the parent from a revision segmentation

        :param ID:
        :return:
        """
        parent_id = -1

        if ImageHandler.is_revision_by_ID(id):
            parent_id = re.match(r'^.*-|r[0-9]*$', id).group(0)[0:-1]

        return parent_id

    @staticmethod
    def extract_exp_from_ID(id):
        """
        Returns the experiment from segmentation

        :param ID:
        :return:
        """
        return ImageHandler.extract_from_ID(id, 0)

    @staticmethod
    def extract_num_from_ID(id):
        """
        Returns the experiment from segmentation

        :param ID:
        :return:
        """
        return ImageHandler.extract_from_ID(id, 1)

    @staticmethod
    def extract_ext_from_ID(id):
        """
        Returns extension from segmentation

        :param ID:
        :return:
        """
        return ImageHandler.extract_from_ID(id, 2)

    @staticmethod
    def extract_rev_from_ID(id):
        """
        Returns extension from segmentation

        :param ID:
        :return:
        """
        return ImageHandler.extract_from_ID(id, 3)

    @staticmethod
    def extract_expnum_from_ID(id):
        """
        Returns the experiment from segmentation

        :param ID:
        :return:
        """
        extracted_id = ImageHandler.extract_exp_from_ID(id)
        extracted_num = ImageHandler.extract_num_from_ID(id)

        expnum = extracted_id + '-' + str(extracted_num)

        return expnum

    @staticmethod
    def extract_from_ID(id, pos):
        """
        Extract number from id at position
        0 - Experiment
        1 - number
        2 - ext number
        3 - revision number

        :param id:
        :param pos:
        :return:
        """
        req_id = -1

        if pos == 0:
            # experiment number
            search = re.search('N[0-9]+-[0-9]+', id)
            if search is not None:
                req_id = search.group()
        elif pos > 0 and pos < 3:
            counter = 0

            # go through until position is found
            for num in re.finditer('-[0-9]+', id):
                if counter == pos:
                    req_id = int(num.group()[1:])
                    break

                counter += 1
        elif pos == 3:
            # revision number
            search = re.search('^.*-r[0-9]*$', id)
            if search is not None:
                parent_id = re.match(r'^.*-r|[0-9]*$', id).group(0)
                req_id = int(id[len(parent_id):])

        return req_id

    @staticmethod
    def load_image_by_path(path, channel=None):
        """
        Load tiff file by path given

        :param path:
        :return:
        """
        image = io.imread(path)

        if channel is not None:
            image = image[:, :, :, channel]

        # determine channels in image
        if len(image.shape) > 3:
            channels = image.shape[3]
        else:
            channels = 1

        return image, channels

    @staticmethod
    def get_image_channel(image, channel=None):
        """
        Return channel of image

        :param path:
        :return:
        """
        img_channel = image[:, :, :, channel]

        return img_channel

    @staticmethod
    def load_image(image_info, non_nuclei_path=None):
        """
        Load an image in the path

        :param image_info:
        :param non_nuclei_path:
        :return:
        """
        # get image properties
        path = image_info['path']
        img_bound = image_info['bound']
        img_channels = image_info['channels']

        print('Load image: %s' % path)

        raw_stack = io.imread(path)

        # stack for resizing
        res_stack = raw_stack.copy()

        # define image stacks for more intuitive handling
        channel_stack = dict()

        # scale image
        if img_bound['Z-min'] is not None\
                and img_bound['Z-max'] is not None:
            res_stack = res_stack[img_bound['Z-min']:img_bound['Z-max'], :, :, :];

        if img_bound['Y-min'] is not None\
                and img_bound['Y-max'] is not None:
            res_stack = res_stack[:, img_bound['Y-min']:img_bound['Y-max'], :, :];

        if img_bound['X-min'] is not None\
                and img_bound['X-max'] is not None:
            res_stack = res_stack[:, :, img_bound['X-min']:img_bound['X-max'], :];

        # convert to 8bit from 16bit
        if res_stack[0].dtype == 'uint16':
            for key, value in enumerate(res_stack):
                res_stack[key] = (value/(2**4)).astype('uint8')

        # reorder stacks to channels
        for cID, channel in enumerate(img_channels):
            print('Channel %i:%s' % (cID, channel))
            channel_stack[channel] = res_stack[:, :, :, cID].copy();

        # delete resized stack
        del res_stack

        # load non-nuclei instead of lamin
        if non_nuclei_path is not None:
            channel_stack[ImageHandler.CHN_LAMIN], channels = ImageHandler.load_image_by_path(
                non_nuclei_path
            )

        return channel_stack

    @staticmethod
    def create_tmp_dir():
        """
        Create a temporary directory with date and a serial number

        :return:
        """
        # get date
        time = datetime.datetime.now()
        timestamp = "{:%Y%m%d}".format(time)

        # get revision number
        revision = 0
        tmp_path = "tmp/%s-r%i/" % (timestamp, revision)

        # create folder
        while os.path.exists(tmp_path):
            revision += 1
            tmp_path = "tmp/%s-r%i/" % (timestamp, revision)

        os.makedirs(tmp_path)

        return tmp_path

    @staticmethod
    def save_stack_as_tiff(img_stack, path):
        """
        Save stack as tiff image

        :param img_stack:
        :param path:
        :return:
        """
        # convert to 8bit
        #conv_stack = img_stack.astype(img_type)

        io.imsave(path, img_stack, plugin='tifffile')

    @staticmethod
    def load_tiff_as_stack(path):
        """
        Load tiff as stack

        :param path:
        :return:
        """
        stack = io.imread(path)

        return stack

    @staticmethod
    def create_dir(path):
        """
        Create directory

        :param path:
        :return:
        """
        # create directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    @staticmethod
    def transform_rgb_img(rgb_img):
        """
        Transform RGB image

        :param rgb_stack:
        :return:
        """
        # create shape
        trans_img = np.zeros(shape=(rgb_img.shape[1], rgb_img.shape[0], rgb_img.shape[2]))

        # transform for each channel
        for chnl in range(0, rgb_img.shape[2]):
            trans_img[:, :, chnl] = rgb_img[:, :, chnl].T

        return trans_img

    @staticmethod
    def add_stack_to_centre(add_stack, tpl_stack):
        """
        Add a stack to a template stack in the centre

        :param add_stack:
        :param tpl_stack:
        :return:
        """
        # sliced stack
        centred_stack = np.zeros_like(tpl_stack)

        # slices
        slices = list()

        # calculate positions
        for i, tpl_dim in enumerate(tpl_stack.shape[0:3]):
            # substract and take the middle
            add_dim = add_stack.shape[i]

            diff = tpl_dim - add_dim
            dist = diff/2

            # add to slices
            slices.append(list())
            slices[-1].append(dist)
            slices[-1].append(add_dim + dist)

        # add image to slice
        #print('%i:%i, %i:%i, %i:%i' % (slices[0][0], slices[0][1],
        #                               slices[1][0], slices[1][1],
        #                               slices[2][0], slices[2][1]))
        #print('add ', add_stack.shape)

        centred_stack[slices[0][0]:slices[0][1],
                      slices[1][0]:slices[1][1],
                      slices[2][0]:slices[2][1]] = add_stack

        return centred_stack

    @staticmethod
    def get_ext_infos_by_expnum(expnum):
        """
        Get infos for extensions from expnum

        :param expnum:
        :return:
        """
        # get image infos
        image_infos = ImageHandler.load_image_infos()
        ext_infos = list()

        # cycle through image infos and return IDs that
        # match the expnum and are extensions
        for info in image_infos:
            if ImageHandler.extract_expnum_from_ID(info['ID']) == expnum:
                if ImageHandler.extract_ext_from_ID(info['ID']) >= 0:
                    ext_infos.append(info)

        return ext_infos

    @staticmethod
    def get_revs_by_expnum(expnum):
        """
        Get revisions for experiment

        :param expnum:
        :return:
        """
        # get image infos
        image_infos = ImageHandler.load_image_infos()
        rev_infos = list()

        # cycle through image infos and return IDs that
        # match the expnum and are revisions
        for info in image_infos:
            if ImageHandler.extract_expnum_from_ID(info['ID']) == expnum:
                if ImageHandler.extract_rev_from_ID(info['ID']) >= 0:
                    rev_infos.append(info)

        return rev_infos

    @staticmethod
    def get_parent_info_by_id(id):
        """
        Get parent image info by id

        :param id:
        :return:
        """
        # get image infos
        image_infos = ImageHandler.load_image_infos()
        parent_info = None

        # cycle through image infos and return IDs that
        # match the parent
        for info in image_infos:
            if info['ID'] == ImageHandler.extract_parent_from_ID(id):
                parent_info = info
                break

        return parent_info

    @staticmethod
    def create_new_rev_for_expnum(expnum):
        """
        Create a new revision number for experiment

        :param expnum:
        :return:
        """
        # get revision count
        revs = ImageHandler.get_revs_by_expnum(expnum)

        # count and return +1
        new_rev_id = '%s-r%i' % (expnum, len(revs))

        return new_rev_id
