"""
Presentation of images, preprocessing and segmentation results. These should be generic methods that you
can reuse for the presentation of the different analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# threading for nucleus planes
import threading

from mpl_toolkits.axes_grid1 import make_axes_locatable

from storage.image import ImageHandler
from storage.stacks import Stack
import storage.config as cfg

import frontend.figures.thread as plot_thread


class Plot:

    @staticmethod
    def view_stack(image_stack, num_cols=10, view_range=range(0, 0),
                   colour_map='gray', zoom=1, scale=1.0):
        """
        Show planes of one stack

        :param image_stack:
        :param num_cols:
        :param view_range:
        :param colour_map:
        :param zoom:
        :param scale:
        :return:
        """
        # get z depth
        stack_z = image_stack.shape[0]

        # define viewing range if nothin given as the whole stack
        if len(view_range) < 1:
            view_range = range(0, stack_z)

        # How many rows do I need for this amount of columns?
        num_rows = np.int(np.ceil(len(view_range) / num_cols))

        # prepare subplot
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(scale*num_cols, scale*num_rows))

        # go through stack
        counter = 0
        for n in view_range:
            i = counter // num_cols
            j = counter % num_cols

            # show image
            axes[i, j].imshow(image_stack[n, ...], interpolation='nearest', cmap=colour_map)

            # remove ticks
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            counter += 1

        # Remove empty plots
        for ax in axes.ravel():
            if not(len(ax.images)):
                fig.delaxes(ax)

        plt.tight_layout()

    def show_stacks(fig, image_stacks, view_range=range(0, 0), img_title=None,
                colour_map=None, zoom=1, scale=2.0):
        """
        Show stacks from image side by side

        :param num_cols:
        :param view_range:
        :param img_title:
        :param colour_map:
        :param zoom:
        :param scale:
        :return:
        """
        # get z depth
        stack_depth = image_stacks[0].shape[0]

        # how many stacks?
        num_stacks = len(image_stacks)

        # is a viewing range defined? If not, show the whole stack
        if len(view_range) < 1:
            view_range = range(0, stack_depth)

        # init colour_map
        if colour_map is None:
            colour_map = list(range(0, num_stacks))

            for i in range(0,len(colour_map)):
                colour_map[i] = 'gray'

        # create figure
        fig_cols = num_stacks
        fig_rows = len(list(view_range))

        # go through stacks
        counter = 0
        for n in view_range:
            # go through the individual stack
            for x in range(0, num_stacks):
                image = image_stacks[x]

                # calculate zoom
                if zoom > 1:
                    image = Plot.get_zoom_box(image_stacks[x], zoom)

                # position in the figure
                fig_pos = (counter * num_stacks) + (x + 1)

                # show the plane
                axes = fig.add_subplot(fig_rows, fig_cols, fig_pos)
                axes.imshow(image[n, :, :], interpolation='nearest', cmap=colour_map[x])

                # show title
                if img_title is not None:
                    axes.set_title('%s (%i)' % (img_title[x], n), cfg.fontsdict_plot)

                # remove ticks
                axes.set_xticks([])
                axes.set_yticks([])

            counter += 1

        plt.tight_layout()

    @staticmethod
    def get_zoom_box(image_stack, zoom):
        """
        Get zoom box for image

        :param image_stack:
        :param zoom:
        :return:
        """
        row_len = image_stack.shape[1]
        col_len = image_stack.shape[2]

        min_row = int((row_len/2) - ((row_len/2) / zoom))
        max_row = int((row_len/2) + ((row_len/2) / zoom))

        min_col = int((col_len/2) - ((col_len/2) / zoom))
        max_col = int((col_len/2) + ((col_len/2) / zoom))

        return image_stack[:, min_row:max_row, min_col:max_col]

    @staticmethod
    def show_nucleus_planes(fig, stacks, nucleus_centroids, view_range=range(0, 0), zoom=1):
        """
        Show multiple stacks. This method assumes that all stacks have the same shape
        to show them side by side.

        :param fig:
        :param image_stacks:
        :param nucleus_centroids:
        :param view_range:
        :param colour_map:
        :param zoom:
        :return:
        """
        # prepare output
        results = list()
        results = Plot.prepare_output(results, stacks.lamin, 'Lamin', 'gray')
        results = Plot.prepare_output(results, stacks.labels, 'Labels', 'gray')
        results = Plot.prepare_output(results, stacks.nuclei, 'Nuclei', 'spectral')

        # set image stacks
        image_stacks = results[0]

        # get z depth
        stack_depth = image_stacks[0].shape[0]

        # how many stacks?
        num_stacks = len(image_stacks)

        # colours
        colour_map = results[2]

        # is a viewing range defined? If not, show the whole stack
        if len(view_range) < 1:
            view_range = range(0, stack_depth)

        # init colour_map
        if colour_map is None:
            colour_map = list(range(0, num_stacks))

            for i in range(0,len(colour_map)):
                colour_map[i] = 'gray'

        # set dpi
        #fig.set_dpi(100)

        # calculate nucleus first and last
        nucleus_fila = (
            nucleus_centroids[0, 0],
            nucleus_centroids[-1, 0]
        )

        # calculate start of z
        if nucleus_fila[0] < cfg.nucleus_box_offset:
            z_start = nucleus_fila[0]
        else:
            z_start = nucleus_fila[0] - cfg.nucleus_box_offset

        # calculate zoom
        row_len = len(image_stacks[0][0])
        col_len = len(image_stacks[0][0][0])

        min_row = int((row_len/2) - ((row_len/2) / zoom))
        max_row = int((row_len/2) + ((row_len/2) / zoom))

        min_col = int((col_len/2) - ((col_len/2) / zoom))
        max_col = int((col_len/2) + ((col_len/2) / zoom))

        for x in range(0, num_stacks):
            image_stacks[x] = image_stacks[x][:, min_row:max_row, min_col:max_col]

        # go through stacks
        for n in view_range:
            # calculate current plane
            cur_plane = z_start + n

            # go through the individual stack
            for x in range(0, num_stacks):
                # show the plane
                threading.Thread(target=Plot.show_nucleus_planes_axes,
                                 args= (
                                     image_stacks, num_stacks, stack_depth,
                                     n, x, cur_plane, fig, nucleus_fila,
                                     colour_map, zoom
                                 )
                                 ).start()
                if plot_thread.thread is False:
                    break

            if plot_thread.thread is False:
                break

        if plot_thread.thread:
            plt.tight_layout()

    @staticmethod
    def show_nucleus_planes_axes(image_stacks, num_stacks, stack_depth, n, x, cur_plane,
                                 fig, nucleus_fila, colour_map, zoom):
        # position in the figure
        fig_pos = (n * num_stacks) + (x + 1)

        axes = fig.add_subplot(stack_depth, num_stacks, fig_pos)
        axes.imshow(image_stacks[x][n],
                    interpolation='nearest', cmap=colour_map[x])

        # show start and end of nucleus
        if x == 0:
            if nucleus_fila[0] == cur_plane:
                axes.set_title('--- START ---', cfg.fontsdict_plot)
            if nucleus_fila[1] == cur_plane:
                axes.set_title('--- END ---', cfg.fontsdict_plot)

        # show z on last stack
        if x == num_stacks -1:
            axes.set_title('Plane: %i' % cur_plane, cfg.fontsdict_plot)

        # remove ticks
        axes.set_xticks([])
        axes.set_yticks([])

    @staticmethod
    def prepare_output(output, image, title=None, cmap='gray', text_pos=None, zoom=1):
        """
        Prepare output to plot

        :param image:
        :param title:
        :param cmap:
        :return:
        """
        # create output if none is given to be modified
        if len(output) == 0:
            output = list(range(0, 5))

            for key, value in enumerate(output):
                output[key] = list()

        # add information to output
        output[0].append(image)
        output[1].append(title)
        output[2].append(cmap)

        # calculate zoom
        row_len = image.shape[0]
        col_len = image.shape[1]

        if text_pos is not None and zoom > 1:
            min_row = int(text_pos[1] - ((row_len/2) / zoom))
            max_row = int(text_pos[1] + ((row_len/2) / zoom))

            min_col = int(text_pos[0] - ((col_len/2) / zoom))
            max_col = int(text_pos[0] + ((col_len/2) / zoom))

            # adjust image slices
            if min_row < 0:
                min_row = 0
            if min_col < 0:
                min_col = 0
            if max_row > row_len:
                max_row = row_len
            if max_col > col_len:
                max_col = col_len

            # adjust text position
            text_pos = (text_pos[0] - min_col, text_pos[1] - min_row)
        else:
            min_row = 0
            max_row = row_len

            min_col = 0
            max_col = col_len

        # add text position
        output[3].append(text_pos)

        # add image zoom
        output[4].append([min_row, min_col, max_row, max_col])

        return output

    @staticmethod
    def view_images(img_list, are_objects=False, cmap='gray', display=True, save=False):
        """
        View individual images

        :param are_objects:
        :param cmap:
        :param display:
        :param save:
        :return:
        """
        if save:
            # create temporary directory to save the images
            tmp_dir = ImageHandler.create_tmp_dir()

        if display:
            # create a plot to show images
            fig, axes = plt.subplots(ncols=len(img_list), figsize=(2*len(img_list), 2),
                                     sharex=True, sharey=True)

        for i, img in enumerate(img_list):
            if save:
                # save images in temporary directory
                plt.imsave((tmp_dir + '/Image-%i') % i, img, cmap=cmap)

            if display:
                # displat the objects
                if are_objects:
                    axes[i].imshow(img['img'], cmap='Spectral')
                    axes[i].set_title('Obj: %i' % img['label'])
                else:
                    axes[i].imshow(img, cmap=cmap)
                    axes[i].set_title('Image: %i' % i)

                axes[i].axis('off')
                axes[i].set_adjustable('box-forced')

    @staticmethod
    def view_histogram_of_stack(image_stack):
        """
        Show histogram of image stack

        :param image_stack:
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(image_stack.flatten(), log=True, bins=2**8, range=(0, 2**8))

        _ = ax.set_title(
            'Min value: %i \n'
            'Max value: %i \n'
            'Image shape: %s \n'
            % (
                image_stack.min(),
                image_stack.max(),
                image_stack.shape
            )
        )

    @staticmethod
    def view_histogram_of_value_list(fig, value_list, bins=10):
        """
        Show histogram of values in a list

        :param image_stack:
        :return:
        """
        ax = fig.add_subplot(111)

        # get minimum and maximum
        min_value = min(value_list)
        max_value = max(value_list)

        # create histogram
        hist = ax.hist(value_list, log=True, bins=bins, range=(0, max_value))

        fig.tight_layout()

    @staticmethod
    def show_nucleus_box(fig, nucleus_boxes, nucleus_centroids, nucleus_areas, stacks, reload=False):
        """
        Show a nucleus in a box

        :param fig:
        :param nID:
        :param nucleus_boxes:
        :param stacks:
        :return:
        """
        stack_boxes = Plot.get_nucleus_boxes(
            nucleus_boxes, nucleus_centroids, nucleus_areas, stacks, reload=reload)

        # calculate overlays
        overlay_xy = nucleus_boxes['crop_rgb'][round(nucleus_boxes['crop_rgb'].shape[0] / 2), :, :].copy()
        overlay_zx = nucleus_boxes['crop_rgb'][:, round(nucleus_boxes['crop_rgb'].shape[1] / 2), :].copy()
        overlay_zy = ImageHandler.transform_rgb_img(
            nucleus_boxes['crop_rgb'][:, :, round(nucleus_boxes['crop_rgb'].shape[2] / 2)].copy())

        # xy image
        ax_xy = fig.add_subplot(111)
        ax_xy.imshow(stack_boxes.lamin[round(stack_boxes.lamin.shape[0] / 2)], cmap='gray')
        ax_xy.imshow(overlay_xy, cmap=cfg.nucleus_select_corr_nuclei_overlay_colour,
                     alpha=cfg.nucleus_select_corr_nuclei_overlay_alpha)
        divider = make_axes_locatable(ax_xy)

        axes_size = 2.5

        # zx projection
        ax_zx = divider.append_axes("bottom", axes_size, pad=0.2, sharex=ax_xy)
        ax_zx.imshow(stack_boxes.lamin[:, round(stack_boxes.lamin.shape[1] / 2), :], cmap='gray')
        ax_zx.imshow(overlay_zx, cmap=cfg.nucleus_select_corr_nuclei_overlay_colour,
                     alpha=cfg.nucleus_select_corr_nuclei_overlay_alpha)
        ax_zx.set_ylim([stack_boxes.lamin.shape[0], 0])

        # zy projection
        ax_zy = divider.append_axes("right", axes_size, pad=0.2, sharey=ax_xy)
        ax_zy.imshow(stack_boxes.lamin[:, :, round(stack_boxes.lamin.shape[2] / 2)].T, cmap='gray')
        ax_zy.imshow(overlay_zy, cmap=cfg.nucleus_select_corr_nuclei_overlay_colour,
                     alpha=cfg.nucleus_select_corr_nuclei_overlay_alpha)
        ax_zy.set_xlim([0, stack_boxes.lamin.shape[0]])

        return stack_boxes

    @staticmethod
    def get_nucleus_box(nucleus_centroids, nucleus_areas, img_stack, offset, img_type='uint8'):
        """
        Calculate box around the nucleus

        :param nucleus_centroids:
        :param nucleus_areas:
        :param img_stack:
        :param offset:
        :return:
        """
        # get the middle centroid
        nucleus_middle = round((nucleus_centroids[-1][0] - nucleus_centroids[0][0])/2)

        # crop the lamin image around the centroid depending on the area plus a certain value
        radius = round(math.sqrt(nucleus_areas[int(nucleus_middle)][1]/math.pi) + offset)

        # calculate box parameters
        z = (nucleus_centroids[0][0] - offset,
             nucleus_centroids[-1][0] + offset)
        row = (round(nucleus_centroids[int(nucleus_middle), 1])
               - radius, round(nucleus_centroids[int(nucleus_middle), 1]) + radius)
        col = (round(nucleus_centroids[int(nucleus_middle), 2])
               - radius, round(nucleus_centroids[int(nucleus_middle), 2]) + radius)

        # adjust for negative values
        z = [0 if i < 0 else int(i) for i in z]
        row = [0 if i < 0 else int(i) for i in row]
        col = [0 if i < 0 else int(i) for i in col]

        # crop stack
        box_stack = img_stack[z[0]:z[1], row[0]:row[1], col[0]:col[1]]

        return box_stack.astype(img_type)

    @staticmethod
    def get_nucleus_boxes(nucleus_boxes, nucleus_centroids, nucleus_areas, stacks, reload=False):
        """
        Calculate boxes for nucleus

        :param nucleus_boxes
        :param stacks:
        :return:
        """
        stack_boxes = Stack()

        # get boxes for the stacks
        if reload is True:
            stack_boxes.lamin = Plot.get_nucleus_box(
                nucleus_centroids, nucleus_areas, stacks.lamin, cfg.nucleus_box_offset)
            stack_boxes.labels = Plot.get_nucleus_box(
                nucleus_centroids, nucleus_areas, stacks.labels, cfg.nucleus_box_offset)
            stack_boxes.nuclei = Plot.get_nucleus_box(
                nucleus_centroids, nucleus_areas, stacks.nuclei, cfg.nucleus_box_offset)
        else:
            stack_boxes.lamin = nucleus_boxes['lamin']
            stack_boxes.labels = nucleus_boxes['labels']
            stack_boxes.nuclei = nucleus_boxes['bw']

        # make all nuclei highest value
        #stack_boxes.nuclei[stack_boxes.nuclei > 0] = 255

        return stack_boxes

    @staticmethod
    def show_image_preview(fig, image, z=-1):
        """
        Show image in xyz

        :param fig:
        :param image:
        :param z:
        :return:
        """
        if z < 0:
            z = round(image.shape[0] / 2)

        if z >= image.shape[0]:
            z = image.shape[0] - 1

        # xy image
        ax_xy = fig.add_subplot(111)
        ax_xy.imshow(image[z], cmap='gray')
        ax_xy.set_xticks([], [])
        ax_xy.set_yticks([], [])

        return ax_xy

    @staticmethod
    def show_image_xyz(fig, image, z=-1):
        """
        Show image in xyz

        :param fig:
        :param image:
        :param z:
        :return:
        """
        if z < 0:
            z = int(image.shape[0] / 2)

        if z >= image.shape[0]:
            z = image.shape[0] - 1

        # xy image
        ax_xy = fig.add_subplot(111)
        ax_xy.imshow(image[z], cmap='gray')
        divider = make_axes_locatable(ax_xy)

        # zx projection
        ax_zx = divider.append_axes("bottom", 1, pad=0.2, sharex=ax_xy)
        ax_zx.imshow(image[:, round(image.shape[1] / 2), :], cmap='gray')
        ax_zx.set_ylim([image.shape[0], 0])
        ax_zx.set_xticks([], [])
        ax_zx.set_yticks([], [])

        # zy projection
        ax_zy = divider.append_axes("right", 1, pad=0.2, sharey=ax_xy)
        ax_zy.imshow(image[:, :, round(image.shape[2] / 2)].T, cmap='gray')
        ax_zy.set_xlim([0, image.shape[0]])
        ax_zy.set_xticks([], [])
        ax_zy.set_yticks([], [])

        return ax_xy, ax_zx, ax_zy

    @staticmethod
    def show_images(fig, results, cols=1, scale_x=-1, scale_y=-1, overlay=None):
        """
        Show images

        :param fig:
        :param images:
        :return:
        """
        # set image properties
        images = results[0]
        num_images = len(images)
        titles = results[1]
        colour_map = results[2]
        title_pos = results[3]
        zoom = results[4]

        # set overlay
        if overlay is not None and len(overlay) > 0:
            overlay_images = overlay[0]
            overlay_colour_map = overlay[2]

        # init colour_map
        if colour_map is None:
            colour_map = list(range(0, num_images))

            for i in range(0,len(colour_map)):
                colour_map[i] = 'gray'

        # calculate rows
        rows = math.ceil(num_images / cols)

        # store axes
        axes = list()

        # go through the individual images
        for x in range(0, num_images):
            # position in the figure
            fig_pos = (x + 1)

            # show image
            axes.append(fig.add_subplot(rows, cols, fig_pos))
            axes[-1].imshow(images[x][zoom[x][0]:zoom[x][2], zoom[x][1]:zoom[x][3]].astype(int),
                        interpolation='nearest', cmap=colour_map[x])

            if overlay is not None and len(overlay) > 0:
                overlay_image = np.ma.masked_where(overlay_images[x] < 1, overlay_images[x])
                axes[-1].imshow(overlay_image, interpolation='none', cmap=overlay_colour_map[x])

            # remove ticks
            axes[-1].set_xticks([])
            axes[-1].set_yticks([])

            # set scales
            if scale_x >= 0:
                axes[-1].set_autoscalex_on(False)
                axes[-1].set_xlim([0, scale_x])

            if scale_y >= 0:
                axes[-1].set_autoscaley_on(False)
                axes[-1].set_ylim([0, scale_x])

            # set title
            if title_pos[x] is None:
                if titles[x] is not None:
                    axes[-1].set_title(titles[x])
            else:
                axes[-1].annotate(titles[x], title_pos[x], fontsize=20, color='red', weight='bold')

        fig.tight_layout()

        return axes
