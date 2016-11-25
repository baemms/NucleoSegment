"""
Choose processing steps for nuclei segmentation
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import time

# matplot for Qt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# Qt libraries
from PyQt4 import QtGui, QtCore

import frontend.gui.labels as gui_labels
from storage.image import ImageHandler
from storage.stacks import Stack
import storage.config as cfg
from processing.image import ImageProcessing
from frontend.figures.plot import Plot
from frontend.gui.layout import Layout
from processing.segmentation import Segmentation

class NucleoProcess(QtGui.QDialog):

    def __init__(self, image_info, silent_processing=False, parent=None):
        super(NucleoProcess, self).__init__(parent)

        # load image infos
        self.image_info = image_info

        # load image
        image_stack = ImageHandler.load_image(image_info)

        # focus on lamin signal
        self.stacks = Stack()
        self.stacks.lamin = image_stack[ImageHandler.CHN_LAMIN]

        # init processing steps
        self.processing_steps = list()
        self.reset_processing_steps()

        # build filtered list
        self.stacks.filtered = list()
        # apply filters
        self.stacks.filtered.append(
            [ImageProcessing.apply_filters(self.processing_steps, self.stacks.lamin)])

        # edit boxes for range
        self.edt_image_range_start = QtGui.QLineEdit()
        self.edt_image_range_stop = QtGui.QLineEdit()
        self.edt_image_range_int = QtGui.QLineEdit()
        self.edt_image_zoom = QtGui.QLineEdit()

        # calculate educated guesses for range take the middle
        range_start = round(self.stacks.lamin.shape[0]/2) - cfg.image_processing_default_range_offset
        range_stop = round(self.stacks.lamin.shape[0]/2) + cfg.image_processing_default_range_offset

        if range_start < 0:
            range_start = 0

        if range_stop > self.stacks.lamin.shape[0]:
            range_stop = self.stacks.lamin.shape[0]

        self.image_range = range(range_start, range_stop,
                            cfg.image_processing_default_range_int)
        self.image_int = cfg.image_processing_default_range_int
        self.image_zoom = cfg.image_processing_default_zoom

        # set layout for processing results
        self.ctn_image_process_results = QtGui.QGridLayout()

        # set combolist for filtering methods and the methods chosen
        self.sel_image_process_steps = list()

        self.filter_mapping = ImageProcessing.get_filter_mapping()

        # buttons to submit processing steps
        self.btn_image_process_reset = QtGui.QPushButton(gui_labels.btn_reset)
        self.btn_image_process_load = QtGui.QPushButton(gui_labels.btn_load)
        self.btn_image_process_save = QtGui.QPushButton(gui_labels.btn_save)
        self.btn_image_process = QtGui.QPushButton(gui_labels.btn_process)
        self.btn_close = QtGui.QPushButton(gui_labels.btn_close)

        # set main layout
        self.setLayout(self.prep_ctn_image_process())

        # silent processing
        if silent_processing is True:
            # load processing steps
            self.load_image_process_steps(False)

            # process image
            self.process_image()
        else:
            # load settings and apply
            self.load_image_process_steps()

    def prep_ctn_image_process(self):
        """
        Container to process image

        :return:
        """
        container = QtGui.QGridLayout()

        # range
        container.addLayout(self.prep_ctn_image_range(), 0, 0)

        # steps
        container.addLayout(self.prep_ctn_image_process_steps(), 1, 0)

        # storage and processing
        container.addLayout(self.prep_ctn_image_submit(), 2, 0)

        return container

    def prep_ctn_image_range(self):
        """
        Prepare container for image range and buttons for navigating
        the processing steps

        :return:
        """
        container = QtGui.QGridLayout()

        # add labels
        container.addWidget(QtGui.QLabel(gui_labels.proc_img_range_start), 0, 0)
        container.addWidget(QtGui.QLabel(gui_labels.proc_img_range_stop), 0, 2)
        container.addWidget(QtGui.QLabel(gui_labels.proc_img_range_int), 1, 0)
        container.addWidget(QtGui.QLabel(gui_labels.proc_img_zoom), 1, 2)

        # create buttons for navigation
        btn_next = QtGui.QPushButton(gui_labels.btn_add_new)
        btn_apply = QtGui.QPushButton(gui_labels.btn_apply)

        # connect listener
        btn_next.clicked.connect(self.add_image_process_results)
        btn_apply.clicked.connect(self.apply_image_process_steps)

        # add to container
        container.addWidget(self.edt_image_range_start, 0, 1)
        container.addWidget(self.edt_image_range_stop, 0, 3)
        container.addWidget(self.edt_image_range_int, 1, 1)
        container.addWidget(self.edt_image_zoom, 1, 3)
        container.addWidget(btn_next, 0, 4)
        container.addWidget(btn_apply, 1, 4)

        # set text defaults
        self.edt_image_range_start.setText(str(self.image_range[0]))
        self.edt_image_range_stop.setText(str(self.image_range[-1]))
        self.edt_image_range_int.setText(str(self.image_int))
        self.edt_image_zoom.setText(str(self.image_zoom))

        return container

    def prep_ctn_image_process_steps(self):
        """
        Prepare container for image processing steps

        :return:
        """
        # prepare original image
        self.prep_ctn_image_process_results()

        # prepare first filter
        self.add_image_process_results()

        # stretch results part
        self.ctn_image_process_results.setRowStretch(2, 1)

        return self.ctn_image_process_results

    def prep_ctn_image_submit(self):
        """
        Prepare container to submit the processing pipeline

        :return:
        """
        container = QtGui.QGridLayout()

        # add buttons to layout
        container.setColumnStretch(0, 1)
        container.addWidget(self.btn_image_process_reset, 0, 2)
        container.addWidget(self.btn_image_process_load, 0, 3)
        container.addWidget(self.btn_image_process_save, 0, 4)
        container.addWidget(self.btn_image_process, 0, 5)
        container.addWidget(self.btn_close, 0, 6)

        # add listener
        self.btn_image_process_reset.clicked.connect(self.reset_processing_selection)
        self.btn_image_process_load.clicked.connect(self.load_image_process_steps)
        self.btn_image_process_save.clicked.connect(self.save_image_process_steps)
        self.btn_image_process.clicked.connect(self.process_image)
        self.btn_close.clicked.connect(self.close)

        return container

    def close(self):
        """
        Close processing and show the main window again

        :return:
        """
        # show parent
        self.parent().showNormal()

        # close processing
        super(NucleoProcess, self).close()

    def process_image(self):
        """
        Segment image with the saved parameters

        :return:
        """
        non_nuclei = False

        # is current image a revision?
        if Segmentation.is_revision_by_ID(self.image_info['ID']):
            # create a new segmentation
            non_nuclei = True

        seg = Segmentation(self.image_info, non_nuclei)

        # process stack
        seg.segment(process=True, merge=False, filter=False)

        # save results
        seg.save(force_labels_props_raw=True)

    def save_image_process_steps(self):
        """
        Save configures processing steps to csv

        :return:
        """
        # prepare processing steps for writing to file
        proc_steps = list()

        # add id to the start
        proc_steps.append(self.image_info['ID'])

        for step in self.processing_steps[1:]:
            # filter name
            proc_steps.append(step[0])

            # filter params
            if len(step) > 1:
                proc_steps[-1] += '='

                # add parameters
                for id, param in enumerate(step[1:]):
                    if id > 0:
                        proc_steps[-1] += ','

                    # is param a range? then convert
                    if isinstance(param, range):
                        param = ImageProcessing.get_param_from_range(param)

                    proc_steps[-1] += str(param)

        # write to processing file
        ImageHandler.update_exp_csv(self.image_info['ID'],
                                    cfg.file_processing,
                                    proc_steps)

    def load_image_process_steps(self, process_images=True):
        """
        Load image processing steps by experiment id

        :return:
        """
        # clear processing steps from layout and reset selection
        self.reset_processing_selection()

        # get filters for image
        self.processing_steps = ImageProcessing.get_filters(self.image_info)

        # add new columns for steps
        if len(self.processing_steps) > 1:
            for num, step in enumerate(self.processing_steps):
                # add new column
                self.add_image_process_results(update_proc_steps=False)

                # update combobox
                self.sel_image_process_steps[num].setCurrentIndex(
                    ImageProcessing.get_index_of_filter(self.filter_mapping, step[0])
                )
            # set parameters
            self.add_image_process_params(update_fields=True)

            if process_images is True:
                # apply filters
                self.apply_image_process_steps()
        else:
            # add new default column
            self.add_image_process_results()

    def prep_ctn_image_process_results(self):
        """
        Prepare container for image processing results

        :return:
        """
        # show original
        self.ctn_image_process_results.addWidget(QtGui.QLabel(gui_labels.proc_step_ori), 0, 0)

        # FIX: that the size of the first column is similar to the processing steps
        self.ctn_image_process_results.setColumnMinimumWidth(0, cfg.image_processing_image_size * 150)

        # prepare figure and canvas for original
        self.fig_processing_steps = [plt.figure(
            figsize=(len(self.stacks.filtered[0]) * cfg.image_processing_image_size,
                     len(list(self.image_range)) * cfg.image_processing_image_size)
        )]
        self.cnv_processing_steps = [FigureCanvas(self.fig_processing_steps[-1])]

        self.ctn_image_process_results.addWidget(self.cnv_processing_steps[-1], 2, 0)

        Plot.show_stacks(self.fig_processing_steps[-1], self.stacks.filtered[-1],
                         self.image_range, ['ORI'], ['hot'], self.image_zoom)

    def add_image_process_results(self, update_proc_steps=True):
        """
        Add a new column to the processing results

        :return:
        """
        # update processing steps
        if update_proc_steps is True:
            self.update_image_process_steps()

        # dropdown menu for filters
        self.sel_image_process_steps.append(QtGui.QComboBox())

        # cycle through filtering mapping
        for filter in self.filter_mapping.keys():
            self.sel_image_process_steps[-1].addItem(filter)

        # add listener
        self.sel_image_process_steps[-1].activated[str].connect(self.add_image_process_params)
        self.ctn_image_process_results.addWidget(self.sel_image_process_steps[-1],
                                                 0, len(self.sel_image_process_steps))

    def add_image_process_params(self, update_fields=False):
        """
        Go through the image processing steps and add parameter for filter

        :return:
        """
        # go through comboboxes
        for i, step in enumerate(self.sel_image_process_steps):
            cur_proc_pos = i + 1

            # get filter
            sel_filter = step.currentText()
            first = False

            if cur_proc_pos >= len(self.processing_steps):
                # add to processing steps
                self.processing_steps.append([sel_filter])
                first = True

            # match with processing steps
            if first or self.processing_steps[cur_proc_pos][0] != sel_filter or update_fields is True:
                # delete previous params
                Layout.remove_layout(self.ctn_image_process_results.itemAtPosition(1, cur_proc_pos))

                # add parameters to layout
                ctn_params = self.get_image_process_params_ctn(sel_filter)
                self.ctn_image_process_results.addLayout(ctn_params, 1, cur_proc_pos)

                # add parameters if fields are to be updated
                if update_fields is True:
                    # go through params and add values to edit boxes
                    for param_id, param in enumerate(self.processing_steps[i][1:]):
                        param_edit = ctn_params.itemAtPosition(param_id, 1).widget()

                        # set param value
                        param_edit.setText(str(param))

                        param_id += 1
                else:
                    # change processing step entry
                    self.processing_steps[cur_proc_pos] = [sel_filter]

    def get_image_process_params_ctn(self, filter):
        """
        Return parameters for filter

        :param filter:
        :return:
        """
        container = QtGui.QGridLayout()

        # get parameters for filter
        last_row = 0
        for i, param in enumerate(self.filter_mapping[filter]['params']):
            # label
            container.addWidget(QtGui.QLabel(param), i, 0)

            # edit box
            container.addWidget(QtGui.QLineEdit(), i, 1)

            last_row = i

        container.setRowStretch(last_row + 1, 1)

        return container

    def apply_image_process_steps(self):
        """
        Go through the select image processing steps and apply and show results

        :return:
        """
        # update processing steps
        self.update_image_process_steps()

        # update range
        self.update_image_range()

        # rebuild filtered stack
        self.stacks.filtered = list()

        # reset figures and canvas
        self.fig_processing_steps = list()
        self.cnv_processing_steps = list()

        # get through filters and apply
        for step, filter in enumerate(self.processing_steps):
            if len(self.stacks.filtered) > 0:
                image = self.stacks.filtered[-1][0]
            else:
                image = self.stacks.lamin

                # get zoom box for image
                image = Plot.get_zoom_box(image.copy(), self.image_zoom)

            # build up filter list
            filters = list()
            titles = list()

            # is the current filter parameter a range?
            for param_id, param in enumerate(filter[1:]):
                if isinstance(param, range):
                    for param_value in param:
                        filter_range = filter.copy()
                        filter_range[param_id + 1] = param_value

                        filters.append(filter_range.copy())
                        titles.append('%s %i' % (filter[0], param_value))

            if len(filters) == 0:
                filters.append(filter)
                titles.append(filter[0])

            # apply filters
            filtered_images = list()
            colour_map = list()

            for x in filters:
                filtered_images.append(ImageProcessing.apply_filters([x], image))
                colour_map.append('hot')

            self.stacks.filtered.append(filtered_images)

            # prepare figure and canvas
            self.fig_processing_steps.append(plt.figure(
                figsize=(len(filtered_images) * cfg.image_processing_image_size,
                         len(list(self.image_range)) * cfg.image_processing_image_size)
            ))
            self.cnv_processing_steps.append(FigureCanvas(self.fig_processing_steps[-1]))

            self.ctn_image_process_results.addWidget(self.cnv_processing_steps[-1], 2, step)

            # adjust width of column and stretch
            self.ctn_image_process_results.setColumnMinimumWidth(step,
                                                                 len(filtered_images) * cfg.image_processing_image_size * 150)
            self.ctn_image_process_results.setColumnStretch(step, 0)

            # update canvas
            Plot.show_stacks(self.fig_processing_steps[step], filtered_images,
                         self.image_range, titles, colour_map, 1)

            self.cnv_processing_steps[step].draw()

        self.ctn_image_process_results.setColumnStretch(len(self.processing_steps), 1)

    def update_image_range(self):
        """
        Get limits for image range and update for display

        :return:
        """
        # get values from line edits
        range_start = int(self.edt_image_range_start.text())
        range_stop = int(self.edt_image_range_stop.text())
        range_int = int(self.edt_image_range_int.text())
        zoom = int(self.edt_image_zoom.text())

        # set variables
        self.image_range = range(range_start, range_stop, range_int)
        self.image_int = range_int
        self.image_zoom = zoom

    def update_image_process_steps(self):
        """
        Go through comboboxes and update processing steps

        :return:
        """
        # reset processing steps
        self.reset_processing_steps()

        for i, step in enumerate(self.sel_image_process_steps):
            # get filter
            sel_filter = step.currentText()

            # get layout for filter
            param_layout = self.ctn_image_process_results.itemAtPosition(1, i + 1)
            param_id = 0
            param_values = list()

            # get edit boxes
            if param_layout is not None and isinstance(param_layout, QtGui.QGridLayout):
                while param_layout.itemAtPosition(param_id, 1) is not None:
                    param_edit = param_layout.itemAtPosition(param_id, 1).widget()

                    # get param value
                    param = param_edit.text()

                    # convert to digit if possible
                    if param.isdigit():
                        param = int(param)

                    # is the parameter a range?
                    if ImageProcessing.is_param_range(param):
                        param = ImageProcessing.get_param_range(param)

                    param_values.append(param)

                    param_id += 1

            # update params
            self.processing_steps.append([sel_filter] + param_values)

    def reset_processing_steps(self):
        """
        Reset processing steps to default

        :return:
        """
        # empty list
        self.processing_steps = list()

        # add to show original as default
        self.processing_steps.append(['ORI'])

    def reset_processing_selection(self):
        """
        Reset processing selection

        :return:
        """
        # delete widgets
        for num, step in enumerate(self.sel_image_process_steps):
            # remove combobox
            Layout.remove_widget_from_grid(self.ctn_image_process_results, 0, num + 1)

            # remove param layout
            Layout.remove_layout_from_grid(self.ctn_image_process_results, 1, num + 1)

            # remove canvas
            Layout.remove_widget_from_grid(self.ctn_image_process_results, 2, num + 1)

        # empty lists
        self.sel_image_process_steps = list()
        self.processing_steps = list()
        self.reset_processing_steps()
