"""
Define the input image and define processing steps
"""

import re
import numpy as np
import random
import os

# Qt libraries
from PyQt4 import QtGui, QtCore
import matplotlib.pyplot as plt

# matplot for Qt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle, Ellipse, Circle

import frontend.gui.labels as gui_labels
from frontend.figures.plot import Plot
from processing.segmentation import Segmentation
from storage.image import ImageHandler
from storage.stacks import Stack
import storage.config as cfg

from frontend.gui.nuc_process import NucleoProcess
from frontend.gui.nuc_select import NucleoSelect
from frontend.gui.merge_criteria import MergeCriteria
from frontend.gui.nuc_criteria import NucleiCriteria
from frontend.gui.layout import Layout

class NucleoSegment(QtGui.QMainWindow):

    def __init__(self, image_path=None, exp_id=None, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.setWindowTitle('NucleoSegment')

        # load image infos
        self.image_infos = ImageHandler.load_image_infos()

        # set central widget
        central_widget = QtGui.QWidget()
        self.setCentralWidget(central_widget)

        # set main layout
        central_widget.setLayout(self.prep_ctn_image_config())

        # set size
        self.setFixedSize(1300, 800)

        # load experiment
        if exp_id is not None:
            self.load_exp(exp_id)
        else:
            # set the input image if given
            if image_path is not None:
                self.image_path = image_path

                self.select_input_image()

    def prep_ctn_image_config(self):
        """
        Create layout choose the input parameters

        :return:
        """
        container = QtGui.QGridLayout()

        # choose experiments
        container.addLayout(self.prep_ctn_exp(), 0, 0)

        # image information
        container.addLayout(self.prep_ctn_image_choice(), 1, 0)

        # channel definitions
        container.addLayout(self.prep_ctn_image_channels(), 2, 0)

        # cropping of image
        container.addLayout(self.prep_ctn_image_box(), 3, 0)

        # submit and process
        container.addLayout(self.prep_ctn_subproc(), 4, 0)

        # ROI and training selection
        container.addLayout(self.prep_ctn_image_regions(), 0, 1, 6, 1)

        container.setRowStretch(5, 1)

        return container

    def prep_ctn_image_regions(self):
        """
        Prepare container to select image ROI and training regions

        :return:
        """

        container = QtGui.QGridLayout()

        # prepare tabs for selection
        self.tb_image_roi = QtGui.QWidget()
        self.tb_image_train = QtGui.QWidget()

        self.tb_image_roi.setLayout(self.prep_ctn_image_preview())
        self.tb_image_train.setLayout(self.prep_ctn_image_train())

        # add tabs
        self.tbs_image_regions = QtGui.QTabWidget()

        self.tbs_image_regions.addTab(self.tb_image_roi, gui_labels.tb_image_regions_roi)
        self.tbs_image_regions.addTab(self.tb_image_train, gui_labels.tb_image_regions_train)

        #self.tbs_image_regions.currentChanged.connect()

        container.addWidget(self.tbs_image_regions, 0, 0)

        return container

    def prep_ctn_subproc(self):
        """
        Container to show submit and processing options

        :return:
        """
        container = QtGui.QGridLayout()

        # submit and process
        self.btn_image_save_settings = QtGui.QPushButton(gui_labels.btn_save_set)
        self.btn_image_process = QtGui.QPushButton(gui_labels.btn_image_process)
        self.btn_merge_criteria = QtGui.QPushButton(gui_labels.btn_merge_criteria)
        self.btn_nuclei_criteria = QtGui.QPushButton(gui_labels.btn_nuclei_criteria)
        self.btn_nuclei_select = QtGui.QPushButton(gui_labels.btn_nuclei_select)
        self.btn_create_revision = QtGui.QPushButton(gui_labels.btn_create_revision)

        container.addWidget(self.btn_image_save_settings, 0, 0)
        container.addWidget(self.btn_image_process, 0, 1)
        container.addWidget(self.btn_merge_criteria, 0, 2)
        container.addWidget(self.btn_nuclei_criteria, 1, 0)
        container.addWidget(self.btn_nuclei_select, 1, 1)
        container.addWidget(self.btn_create_revision, 1, 2)

        self.btn_image_save_settings.clicked.connect(self.image_save_settings)
        self.btn_image_process.clicked.connect(self.image_process)
        self.btn_merge_criteria.clicked.connect(self.merge_criteria)
        self.btn_nuclei_criteria.clicked.connect(self.nuclei_criteria)
        self.btn_nuclei_select.clicked.connect(self.nuclei_select)
        self.btn_create_revision.clicked.connect(self.create_revision)

        return container

    def merge_criteria(self):
        """
        Select criteria for label merging

        :return:
        """
        # hide yourself
        self.showMinimized()

        # open window for criteria
        nuc_proc = MergeCriteria(self.image_info, parent=self)
        nuc_proc.show()
        nuc_proc.raise_()
        nuc_proc.activateWindow()

    def nuclei_criteria(self):
        """
        Select criteria for nuclei selection

        :return:
        """
        # hide yourself
        self.showMinimized()

        # open window for criteria
        nuc_proc = NucleiCriteria(self.image_info, parent=self)
        nuc_proc.show()
        nuc_proc.raise_()
        nuc_proc.activateWindow()

    def image_process(self):
        """
        Hand over to image processing

        :return:
        """
        # hide yourself
        self.showMinimized()

        # open window for processing
        nuc_proc = NucleoProcess(self.image_info, parent=self)
        nuc_proc.show()
        nuc_proc.raise_()
        nuc_proc.activateWindow()

    def nuclei_select(self):
        """
        Hand over to nuclei selection

        :return:
        """
        # hide yourself
        self.showMinimized()

        # open window for processing
        nuc_proc = NucleoSelect(self.image_info, parent=self)
        nuc_proc.show()
        nuc_proc.raise_()
        nuc_proc.activateWindow()

    def prep_ctn_exp(self):
        """
        Prepare container to select from previous experiments

        :return:
        """
        container = QtGui.QGridLayout()
        container.addWidget(QtGui.QLabel(gui_labels.sel_load_exp), 0, 0)

        # create combobox to select previous experiments
        self.sel_load_exp = QtGui.QComboBox()

        # cycle through image infos and build selection
        for image_info in self.image_infos:
            self.sel_load_exp.addItem(image_info['ID'])

        # add listener
        self.sel_load_exp.activated[str].connect(self.load_exp)

        # add to container
        container.addWidget(self.sel_load_exp, 0, 1)

        return container

    def load_exp(self, ID):
        """
        Load experiment

        :param ID:
        :return:
        """
        self.image_info = None

        # get image info
        for info in self.image_infos:
            if info['ID'] == ID:
                self.image_info = info
                break

        # set parameters
        self.lbl_image_exp.setText(self.image_info['exp'])
        self.lbl_image_num.setText(str(ImageHandler.extract_num_from_ID(self.image_info['ID'])))
        self.lbl_image_date.setText(self.image_info['date'])
        self.lbl_image_file.setText(self.image_info['file'])
        self.edt_image_voxel_size.setText(str(self.image_info['voxel_size']))

        # channels
        self.clean_ctn_image_channels()
        for channel in self.image_info['channels']:
            self.prep_ctn_add_image_channels(channel)

        # crop
        self.edt_image_crop_x_min.setText(str(self.image_info['bound']['X-min']))
        self.edt_image_crop_x_max.setText(str(self.image_info['bound']['X-max']))
        self.edt_image_crop_y_min.setText(str(self.image_info['bound']['Y-min']))
        self.edt_image_crop_y_max.setText(str(self.image_info['bound']['Y-max']))
        self.edt_image_crop_z_min.setText(str(self.image_info['bound']['Z-min']))
        self.edt_image_crop_z_max.setText(str(self.image_info['bound']['Z-max']))

        # set image path
        self.image_path = self.image_info['path']

        self.load_image()

        # load segmentation if possible
        self.segmentation = Segmentation(self.image_info)
        self.segmentation.load()

    def create_revision(self):
        """
        Create revision if segmentation is loaded

        :return:
        """
        # is segmentation loaded?
        if hasattr(self, 'segmentation') and self.segmentation is not None:
            revision_id = ImageHandler.create_new_rev_for_expnum(
                ImageHandler.extract_expnum_from_ID(self.segmentation.image_info['ID'])
            )

            # save image settings
            self.image_save_settings(save_revision=revision_id)

    def image_save_settings(self, save_rand_patches=False, save_rand_nuclei=False,
                            save_revision=None):
        """
        Save settings for image processing

        :return:
        """
        # build image info dictionary
        # image boundaries
        img_bounds = list()

        if save_rand_patches is True:
            for patch_coords in self.rand_patch_coords:
                img_bounds.append({
                    'Z-min': int(self.edt_image_crop_z_min.text()), 'Z-max': int(self.edt_image_crop_z_max.text()),
                    'Y-min': patch_coords[1], 'Y-max': patch_coords[3],
                    'X-min': patch_coords[0], 'X-max': patch_coords[2],
                })
        else:
            img_bounds.append({
                'Z-min': int(self.edt_image_crop_z_min.text()), 'Z-max': int(self.edt_image_crop_z_max.text()),
                'Y-min': int(self.edt_image_crop_y_min.text()), 'Y-max': int(self.edt_image_crop_y_max.text()),
                'X-min': int(self.edt_image_crop_x_min.text()), 'X-max': int(self.edt_image_crop_x_max.text()),
            })

        # get channels
        img_channels = list()
        for channel in self.edt_image_channels:
            img_channels.append(channel.text())

        # label
        if int(self.lbl_image_num.text()) > 0:
            label = self.lbl_image_exp.text() + '-' + self.lbl_image_num.text()
        else:
            label = self.lbl_image_exp.text()

        # create list
        labels = list()
        if save_rand_patches is True:
            for i, patch_coords in enumerate(self.rand_patch_coords):
                labels.append(label + '-' + str(i))
        elif save_rand_nuclei is True:
            labels.append(label + '-' + str(0))
        elif save_revision is not None:
            labels.append(save_revision)
        else:
            labels.append(label)

        # add to image list for processing
        for i, label in enumerate(labels):
            del_ext = False

            if save_rand_patches is True and i == 0:
                del_ext = True

            image_info = {
                'ID': label,
                'exp': self.lbl_image_exp.text(),
                'date': self.lbl_image_date.text(),
                'file': self.lbl_image_file.text(),
                'voxel_size': self.edt_image_voxel_size.text(),
                'path': os.path.join(
                            cfg.path_raw,
                            self.lbl_image_exp.text(),
                            self.lbl_image_date.text(),
                            self.lbl_image_file.text()),
                'bound': img_bounds[i],
                'channels': img_channels
            }

            ImageHandler.save_image_info(image_info, del_ext=del_ext)

        # copy segmentation if random nuclei have to be saved
        if save_rand_nuclei is True:
            # create a copy from segmentation
            seg = Segmentation(self.image_info)
            seg.load()

            # set new ID
            seg.image_info['ID'] = labels[-1]

            # set nuclei
            seg.nuclei = self.train_rand_nuclei

            # save segmentation
            seg.save(force_nuclei_stack_rebuild=True)

    def prep_ctn_image_choice(self):
        """
        Prepare container to choose an image

        :return:
        """
        # image choice
        self.edt_input_image = QtGui.QLineEdit()
        self.btn_input_image = QtGui.QPushButton(gui_labels.btn_browse)

        # information about the image
        self.lbl_image_exp = QtGui.QLabel(gui_labels.img_exp)
        self.lbl_image_num = QtGui.QLabel(gui_labels.img_num)
        self.lbl_image_date = QtGui.QLabel(gui_labels.img_date)
        self.lbl_image_file = QtGui.QLabel(gui_labels.img_file)
        self.edt_image_voxel_size = QtGui.QLineEdit()

        # add event handler
        self.btn_input_image.clicked.connect(self.select_input_image)

        # build container
        container = QtGui.QGridLayout()

        container.addWidget(QtGui.QLabel(gui_labels.img_select), 0, 0)
        container.addWidget(self.edt_input_image, 0, 1)
        container.addWidget(self.btn_input_image, 0, 2)

        container.addWidget(QtGui.QLabel(gui_labels.img_file), 1, 0)
        container.addWidget(QtGui.QLabel(gui_labels.img_voxel_size), 2, 0)
        container.addWidget(QtGui.QLabel(gui_labels.img_exp), 3, 0)
        container.addWidget(QtGui.QLabel(gui_labels.img_num), 4, 0)
        container.addWidget(QtGui.QLabel(gui_labels.img_date), 5, 0)

        container.addWidget(self.lbl_image_file, 1, 1)
        container.addWidget(self.edt_image_voxel_size, 2, 1)
        container.addWidget(self.lbl_image_exp, 3, 1)
        container.addWidget(self.lbl_image_num, 4, 1)
        container.addWidget(self.lbl_image_date, 5, 1)

        return container

    def prep_ctn_image_preview(self):
        """
        Prepare container for image preview

        :return:
        """
        # show image XYZ
        self.fig_image_preview = plt.figure(figsize=(7, 7))
        self.cnv_image_preview = FigureCanvas(self.fig_image_preview)

        # slider for Z
        self.sld_image_preview = QtGui.QSlider(QtCore.Qt.Vertical, self)

        container = QtGui.QGridLayout()
        container.addWidget(self.cnv_image_preview, 0, 1)
        container.addWidget(self.sld_image_preview, 0, 2)

        container.addLayout(self.prep_ctn_cropped_image_preview(), 1, 1, 1, 2)

        container.setColumnStretch(0, 1)
        container.setColumnStretch(3, 1)

        return container

    def prep_ctn_cropped_image_preview(self):
        """
        Prepare container for cropped image preview

        :return:
        """
        # create container
        container = QtGui.QGridLayout()

        # create figure with slider
        self.fig_cropped_image_preview = plt.figure(figsize=(7,7))
        self.cnv_cropped_image_preview = FigureCanvas(self.fig_cropped_image_preview)
        self.sld_cropped_image_preview = QtGui.QSlider(QtCore.Qt.Vertical, self)

        # add to container
        container.addWidget(self.cnv_cropped_image_preview, 0, 0)
        container.addWidget(self.sld_cropped_image_preview, 0, 1)

        container.setColumnStretch(0, 1)

        return container

    def update_cropped_image_preview(self, z=-1):
        """
        Update cropped image preview based on selection

        :return:
        """
        # clear image
        self.fig_cropped_image_preview.clear()

        # set default coords
        # Z
        self.set_edt_image_cropped(self.edt_image_crop_z_min,
                                   self.edt_image_crop_z_max, 0, self.image.shape[0] - 1)

        # Y
        self.set_edt_image_cropped(self.edt_image_crop_y_min,
                                   self.edt_image_crop_y_max, 0, self.image.shape[1] - 1)

        # X
        self.set_edt_image_cropped(self.edt_image_crop_x_min,
                                   self.edt_image_crop_x_max, 0, self.image.shape[2] - 1)

        # calculate image preview
        self.cropped_image_preview = self.image_preview[
            int(self.edt_image_crop_z_min.text()):int(self.edt_image_crop_z_max.text()),
            int(self.edt_image_crop_y_min.text()):int(self.edt_image_crop_y_max.text()),
            int(self.edt_image_crop_x_min.text()):int(self.edt_image_crop_x_max.text())
        ]

        # set slider
        if self.sld_cropped_image_preview.maximum() != int(self.edt_image_crop_z_max.text()) \
            or self.sld_cropped_image_preview.minimum() != int(self.edt_image_crop_z_min.text()):
            # preview
            self.sld_cropped_image_preview.valueChanged[int].connect(self.change_z_for_cropped_preview)
            self.sld_cropped_image_preview.setMinimum(int(self.edt_image_crop_z_min.text()))
            self.sld_cropped_image_preview.setMaximum(int(self.edt_image_crop_z_max.text()))

        if z < 0:
            z = self.cropped_image_preview.shape[0] / 2

        if z >= self.cropped_image_preview.shape[0]:
            z = self.cropped_image_preview.shape[0] - 1

        self.sld_cropped_image_preview.setValue(z)

        # show image preview
        Plot.show_image_xyz(self.fig_cropped_image_preview, self.cropped_image_preview, z)

        # draw canvas
        self.cnv_cropped_image_preview.draw()

        # update training preview
        self.update_cropped_train_preview(z=z)

    def set_edt_image_cropped(self, edt_el_min, edt_el_max, value_min, value_max):
        """
        Set edit for image crop region

        :param edt_el:
        :param max:
        :return:
        """
        # min
        if edt_el_min.text() is None or edt_el_min.text() is '':
            edt_el_min.setText(str(value_min))

        # max
        if edt_el_max.text() is None or edt_el_max.text() is '':
            edt_el_max.setText(str(value_max))

    def image_preview_on_press(self, event):
        """
        Image preview pressed

        :return:
        """
        # set box boundaries
        self.edt_image_crop_x_min.setText(str(int(event.xdata)))
        self.edt_image_crop_y_min.setText(str(int(event.ydata)))

        # set pressed
        self.image_preview_pressed = True

    def image_preview_on_release(self, event):
        """
        Image preview pressed

        :return:
        """

        # set pressed
        self.image_preview_pressed = False

        # final update
        self.image_preview_update_patch(event)

        # update cropped image
        self.update_cropped_image_preview()

    def prep_ctn_image_channels(self):
        """
        Prepare container to choose an image

        :return:
        """
        containter = QtGui.QGridLayout()
        containter.addWidget(QtGui.QLabel(gui_labels.img_chnl), 0, 0)

        # have a dummy container to add channel information
        self.ctn_image_channels = QtGui.QVBoxLayout()

        containter.addLayout(self.ctn_image_channels, 1, 0)

        # make a number group for channel radio buttons
        self.nbg_image_channels = QtGui.QButtonGroup(self.ctn_image_channels)

        return containter

    def clean_ctn_image_channels(self):
        """
        Clear channels from container

        :return:
        """
        # clear channel list
        self.edt_image_channels = None
        self.rdo_image_channels = None

        # remove edit boxes
        while self.ctn_image_channels.count() > 0:
            layout = self.ctn_image_channels.takeAt(0)

            while layout.count() > 0:
                layout.takeAt(0).widget().deleteLater()

            layout.layout().deleteLater()

    def prep_ctn_add_image_channels(self, name=None):
        """
        Prepare container to choose an image

        :return:
        """
        row = QtGui.QHBoxLayout()

        # have a dummy container to add channel information
        if not hasattr(self, 'edt_image_channels') or self.edt_image_channels is None:
            self.edt_image_channels = list()
            self.rdo_image_channels = list()

        # create edit box
        self.edt_image_channels.append(QtGui.QLineEdit())

        if name is not None:
            self.edt_image_channels[-1].setText(name)
        else:
            self.edt_image_channels[-1].setText(gui_labels.img_chnl + (' %i' % len(self.ctn_image_channels)))

        # create selection button
        self.rdo_image_channels.append(QtGui.QRadioButton(str(len(self.ctn_image_channels))))
        self.rdo_image_channels[-1].clicked.connect(self.change_channel)

        # select if channel 0
        if len(self.ctn_image_channels) == 0:
            self.rdo_image_channels[-1].setChecked(True)

        # add to row
        row.addWidget(self.edt_image_channels[-1])
        row.addWidget(self.rdo_image_channels[-1])

        # add to container
        self.ctn_image_channels.addLayout(row)

    def change_channel(self):
        """
        Change channel of preview image

        :return:
        """
        # go through channels and see which one is ticked
        for i, channel in enumerate(self.rdo_image_channels):
            if channel.isChecked():
                self.load_image(False, i)

    def prep_ctn_image_box(self):
        """
        Prepare container to choose the image boundaries

        :return:
        """
        container = QtGui.QGridLayout()
        container.addWidget(QtGui.QLabel(gui_labels.img_crop), 0, 0)

        # crop image edit lines
        self.edt_image_crop_x_min = QtGui.QLineEdit()
        self.edt_image_crop_x_max = QtGui.QLineEdit()

        self.edt_image_crop_y_min = QtGui.QLineEdit()
        self.edt_image_crop_y_max = QtGui.QLineEdit()

        self.edt_image_crop_z_min = QtGui.QLineEdit()
        self.edt_image_crop_z_max = QtGui.QLineEdit()

        # labels
        container.addWidget(QtGui.QLabel(gui_labels.img_crop_x), 1, 0)
        container.addWidget(QtGui.QLabel(gui_labels.img_crop_y), 2, 0)
        container.addWidget(QtGui.QLabel(gui_labels.img_crop_z), 3, 0)

        #listeners
        self.edt_image_crop_x_min.editingFinished.connect(self.update_cropped_image_preview)
        self.edt_image_crop_x_max.editingFinished.connect(self.update_cropped_image_preview)
        self.edt_image_crop_y_min.editingFinished.connect(self.update_cropped_image_preview)
        self.edt_image_crop_y_max.editingFinished.connect(self.update_cropped_image_preview)
        self.edt_image_crop_z_min.editingFinished.connect(self.update_cropped_image_preview)
        self.edt_image_crop_z_max.editingFinished.connect(self.update_cropped_image_preview)

        # edit lines
        container.addWidget(self.edt_image_crop_x_min, 1, 1)
        container.addWidget(self.edt_image_crop_x_max, 1, 2)

        container.addWidget(self.edt_image_crop_y_min, 2, 1)
        container.addWidget(self.edt_image_crop_y_max, 2, 2)

        container.addWidget(self.edt_image_crop_z_min, 3, 1)
        container.addWidget(self.edt_image_crop_z_max, 3, 2)

        return container

    def select_input_image(self):
        """
        Set input image

        :return:
        """
        #if not hasattr(self, 'image_path') or self.image_path is None:
        # ask the user which image to use
        self.edt_input_image.setText(QtGui.QFileDialog.getOpenFileName())

        # extract parameters from path
        self.image_path = self.edt_input_image.text()
        #else:
        #    self.edt_input_image.setText(self.image_path)

        img_exp = re.search('N[0-9]*-[0-9]*', self.image_path).group()
        img_date = re.search('[0-9]{8}', self.image_path)
        img_file = self.image_path[img_date.span()[1]+1:]

        img_date = img_date.group()

        # lookup how many images are already in input from this exp
        # and add one
        img_num = ImageHandler.create_new_id_for_exp(self.image_infos, img_exp)

        # set parameters
        self.lbl_image_exp.setText(img_exp)
        self.lbl_image_num.setText(str(img_num))
        self.lbl_image_date.setText(img_date)
        self.lbl_image_file.setText(img_file)
        self.edt_image_voxel_size.setText('1')

        self.load_image()

        # add channel information
        self.clean_ctn_image_channels()

        print('channels %i' % self.image_channels)

        for channel in range(0, self.image_channels):
            print('add %i' % channel)
            self.prep_ctn_add_image_channels()

    def change_z_for_preview(self, value):
        """
        Change z for preview

        :param value:
        :return:
        """
        self.load_image(load=False, channel=-1, z=value)

    def change_z_for_cropped_preview(self, value):
        """
        Change z for cropped preview

        :param value:
        :return:
        """
        self.update_cropped_image_preview(z=value)

    def load_image(self, load=True, channel=0, z=-1):
        """
        Load image into preview

        :return:
        """
        # load image
        if load is True:
            self.image, self.image_channels = ImageHandler.load_image_by_path(self.image_path)

        # load channel
        if channel >= 0:
            self.image_preview = ImageHandler.get_image_channel(self.image, channel)

            # set z from slider
            z = self.sld_image_preview.value()

        # set slider
        if load is True:
            self.sld_image_preview.valueChanged[int].connect(self.change_z_for_preview)
            self.sld_image_preview.setMinimum(0)
            self.sld_image_preview.setMaximum(self.image.shape[0])
            self.sld_image_preview.setValue(round(self.image.shape[0] / 2))

        # clear image
        self.fig_image_preview.clear()

        # show image
        self.image_preview_xy, self.image_preview_xz, self.image_preview_yz = \
            Plot.show_image_xyz(self.fig_image_preview, self.image_preview, z)

        # draw canvas
        self.cnv_image_preview.draw()

        # create objects for patches
        self.train_preview_patch_xy = Rectangle((0, 0), 1, 1,
                                                color=cfg.nuclei_segment_overlay_colour,
                                                alpha=cfg.nuclei_segment_overlay_alpha)
        self.image_preview_patch_xz = Rectangle((0, 0), 1, 1,
                                                color=cfg.nuclei_segment_overlay_colour,
                                                alpha=cfg.nuclei_segment_overlay_alpha)
        self.image_preview_patch_yz = Rectangle((0, 0), 1, 1,
                                                color=cfg.nuclei_segment_overlay_colour,
                                                alpha=cfg.nuclei_segment_overlay_alpha)
        self.image_preview_patch_pouch = Ellipse((0, 0), 1, 1,
                                                 color=cfg.nuclei_segment_pouch_colour,
                                                 alpha=cfg.nuclei_segment_pouch_alpha)

        # add to patches
        self.image_preview_xy.add_patch(self.train_preview_patch_xy)
        self.image_preview_xz.add_patch(self.image_preview_patch_xz)
        self.image_preview_yz.add_patch(self.image_preview_patch_yz)

        # add listeners
        self.image_preview_pressed = False
        self.image_preview_xy.figure.canvas.mpl_connect('button_press_event',
                                                        self.image_preview_on_press)
        self.image_preview_xy.figure.canvas.mpl_connect('button_release_event',
                                                        self.image_preview_on_release)
        self.image_preview_xy.figure.canvas.mpl_connect('motion_notify_event',
                                                        self.image_preview_on_motion)

        # update cropped preview
        if load is True or channel >= 0:
            self.update_cropped_image_preview()

    def image_preview_on_motion(self, event):
        """
        Update the rectangle while moving

        :param event:
        :return:
        """

        if self.image_preview_pressed is True:
            self.image_preview_update_patch(event)

            # update image
            self.image_preview_xy.figure.canvas.draw()

    def image_preview_update_patch(self, event):
        """
        Update patch

        :param event:
        :return:
        """
        # set patch on xy
        xy_width = int(event.xdata) - int(self.edt_image_crop_x_min.text())
        xy_height = int(event.ydata) - int(self.edt_image_crop_x_min.text())
        xy_coords = (int(self.edt_image_crop_x_min.text()), int(self.edt_image_crop_y_min.text()))

        self.train_preview_patch_xy.set_width(xy_width)
        self.train_preview_patch_xy.set_height(xy_height)
        self.train_preview_patch_xy.set_xy(xy_coords)

        # set patch on xz
        xz_coords = (xy_coords[0], 0)
        self.image_preview_patch_xz.set_width(xy_width)
        self.image_preview_patch_xz.set_height(self.image_preview.shape[0])
        self.image_preview_patch_xz.set_xy(xz_coords)

        # set patch on yz
        yz_coords = (0, xy_coords[1])
        self.image_preview_patch_yz.set_width(self.image_preview.shape[0])
        self.image_preview_patch_yz.set_height(xy_height)
        self.image_preview_patch_yz.set_xy(yz_coords)

        # add ellipse to rectangle
        self.image_preview_patch_pouch.set_visible(False)

        pouch_xy = (
            xy_coords[0] + xy_width/2,
            xy_coords[1] + xy_height/2
        )
        self.image_preview_patch_pouch = Ellipse(pouch_xy, xy_width, xy_height,
                                                 color=cfg.nuclei_segment_pouch_colour,
                                                 alpha=cfg.nuclei_segment_pouch_alpha)
        self.image_preview_xy.add_patch(self.image_preview_patch_pouch)

        # set box boundaries
        self.edt_image_crop_x_max.setText(str(int(event.xdata)))
        self.edt_image_crop_y_max.setText(str(int(event.ydata)))

    def prep_ctn_image_train(self):
        """
        Prepare container to select regions for training

        :return:
        """
        container = QtGui.QGridLayout()

        # prepare tabs for selection
        self.tb_train_rand_patches = QtGui.QWidget()
        self.tb_train_rand_nuclei = QtGui.QWidget()

        # prepare containers
        self.ctn_cropped_train_preview = self.prep_ctn_cropped_train_preview()
        self.ctn_train_rand_patches = self.prep_ctn_train_rand_patches()
        self.ctn_train_rand_nuclei = self.prep_ctn_train_rand_nuclei()

        # add containers
        self.tb_train_rand_patches.setLayout(self.ctn_train_rand_patches)
        self.tb_train_rand_nuclei.setLayout(self.ctn_train_rand_nuclei)

        # add tabs
        self.tbs_train_regions = QtGui.QTabWidget()

        self.tbs_train_regions.addTab(self.tb_train_rand_patches, gui_labels.tb_train_rand_patches)
        self.tbs_train_regions.addTab(self.tb_train_rand_nuclei, gui_labels.tb_train_rand_nuclei)

        self.cur_train_tab = 0

        self.tbs_train_regions.currentChanged.connect(self.change_cur_train_tab)

        container.addWidget(self.tbs_train_regions, 0, 0)

        return container

    def change_cur_train_tab(self, tab_index):
        """
        Change training tab

        :return:
        """
        self.cur_train_tab = tab_index

        coords = [0, 0, 1, 4]

        # change preview
        if self.cur_train_tab == 0:
            self.swap_ctn_cropped_train_preview(self.ctn_train_rand_nuclei,
                                                self.ctn_train_rand_patches,
                                                coords)
        elif self.cur_train_tab == 1:
            self.swap_ctn_cropped_train_preview(self.ctn_train_rand_patches,
                                                self.ctn_train_rand_nuclei,
                                                coords)

        # update image
        self.update_cropped_image_preview()

    def prep_ctn_train_rand_nuclei(self):
        """
        Prepare container to select random patches

        :return:
        """

        container = QtGui.QGridLayout()

        # prepare buttons
        self.btn_add_rand_nuclei = QtGui.QPushButton(gui_labels.btn_add_rand_nuclei)
        self.btn_save_rand_nuclei = QtGui.QPushButton(gui_labels.btn_save_rand_nuclei)

        self.btn_add_rand_nuclei.clicked.connect(self.train_add_rand_nuclei)
        self.btn_save_rand_nuclei.clicked.connect(self.train_save_rand_nuclei)

        # add cropped preview
        #container.addLayout(self.prep_ctn_cropped_train_preview(), 0, 0, 1, 4)

        # add buttons
        container.addWidget(self.btn_add_rand_nuclei, 1, 1)
        container.addWidget(self.btn_save_rand_nuclei, 1, 2)

        container.setRowStretch(0, 1)
        container.setRowStretch(3, 1)
        container.setColumnStretch(0, 1)
        container.setColumnStretch(3, 1)

        return container

    def prep_ctn_train_rand_patches(self):
        """
        Prepare container to select random patches

        :return:
        """

        container = QtGui.QGridLayout()

        # prepare buttons
        self.btn_add_rand_patches = QtGui.QPushButton(gui_labels.btn_add_rand_patches)
        self.btn_save_rand_patches = QtGui.QPushButton(gui_labels.btn_save_rand_patches)

        self.btn_add_rand_patches.clicked.connect(self.train_add_rand_patches)
        self.btn_save_rand_patches.clicked.connect(self.train_save_rand_patches)

        # add cropped preview
        container.addLayout(self.ctn_cropped_train_preview, 0, 0, 1, 4)

        # add buttons
        container.addWidget(self.btn_add_rand_patches, 1, 1)
        container.addWidget(self.btn_save_rand_patches, 1, 2)

        # add patches
        container.addLayout(self.prep_ctn_train_rand_patches_previews(), 3, 0, 1, 4)

        container.setRowStretch(0, 1)
        container.setRowStretch(3, 1)
        container.setColumnStretch(0, 1)
        container.setColumnStretch(3, 1)

        return container

    def swap_ctn_cropped_train_preview(self, old_ctn, new_ctn, coords):
        """
        Swap preview to another container

        :param old_ctn:
        :param new_ctn:
        :param coords:
        :return:
        """
        # remove parent from layout
        Layout.remove_parent_from_grid(old_ctn, coords[0], coords[1])

        # add preview to another container
        new_ctn.addLayout(self.ctn_cropped_train_preview, coords[0], coords[1], coords[2], coords[3])

    def prep_ctn_cropped_train_preview(self):
        """
        Prepare container for cropped train preview

        :return:
        """
        # create container
        container = QtGui.QGridLayout()

        # create figure with slider
        self.fig_cropped_train_preview = plt.figure(figsize=(7,7))
        self.cnv_cropped_train_preview = FigureCanvas(self.fig_cropped_train_preview)

        # add to container
        container.addWidget(self.cnv_cropped_train_preview, 0, 0)

        container.setRowStretch(0, 1)
        container.setColumnStretch(0, 1)

        return container

    def prep_ctn_train_rand_patches_previews(self):
        """
        Prepare container for cropped image preview

        :return:
        """
        # create images for patches
        self.fig_cropped_image_previews = list()
        self.cnv_cropped_image_previews = list()

        container = QtGui.QGridLayout()

        # go through and add figures for the patches
        for patch in range(0, (2 * cfg.nuclei_segment_rand_patches), 2):
            self.fig_cropped_image_previews.append(plt.figure(figsize=(5,5)))
            self.cnv_cropped_image_previews.append(FigureCanvas(self.fig_cropped_image_previews[-1]))

            # add to container
            container.addWidget(self.cnv_cropped_image_previews[-1], 0, patch)

        return container

    def update_cropped_train_preview(self, z=-1):
        """
        Update cropped training preview

        :param z:
        :return:
        """
        # clear image
        self.fig_cropped_train_preview.clear()

        # show image preview for training
        self.cropped_train_preview_xy = Plot.show_image_preview(self.fig_cropped_train_preview, self.cropped_image_preview, z)

        # draw selected patches or nuclei
        if self.cur_train_tab == 0:
            self.image_add_patches()
        elif self.cur_train_tab == 1:
            self.image_add_ellipses()

        # draw canvas
        self.cnv_cropped_train_preview.draw()

    def train_add_rand_nuclei(self):
        """
        Add random nuclei to the image for machine learning

        :return:
        """
        # go through nuclei and get random ones
        # ! given that there is already a segmentation for this experiment !
        if hasattr(self.segmentation, 'nuclei') and self.segmentation.nuclei is not None:
            # get random nuclei from the segmentation
            self.train_rand_nuclei = random.sample(self.segmentation.nuclei, cfg.nuclei_segment_rand_nuclei)

            # prepare raster for preview image
            self.train_add_rand_nuclei_raster()

            # update image
            self.update_cropped_train_preview()

    def train_add_rand_nuclei_raster(self):
        """
        Add a raster for the cropped traininig preview

        :return:
        """
        # get dimensions of cropped image preview
        y_size = self.cropped_image_preview.shape[1]
        x_size = self.cropped_image_preview.shape[2]

        y_raster = int(y_size / cfg.nuclei_segment_rand_nuclei_raster)
        x_raster = int(x_size / cfg.nuclei_segment_rand_nuclei_raster)

        # generate rastered array
        self.rand_nuclei_raster = np.zeros((cfg.nuclei_segment_rand_nuclei_raster,
                                            cfg.nuclei_segment_rand_nuclei_raster))

        # go through random nuclei
        for nucleus in self.train_rand_nuclei:
            # get coords and see in which raster category it is
            y_raster_index = int(nucleus['centre'][1] / y_raster)
            x_raster_index = int(nucleus['centre'][2] / x_raster)

            # increase count for raster position
            self.rand_nuclei_raster[x_raster_index, y_raster_index] += 1

        print('raster ', self.rand_nuclei_raster)

    def image_add_ellipses(self):
        """
        Add ellipses corresponding to the number of random
        nuclei within the raster

        :return:
        """
        if hasattr(self, 'rand_nuclei_raster') is True and self.rand_nuclei_raster is not None:
            # highest count in raster
            max_count = np.max(self.rand_nuclei_raster)

            # get dimensions of cropped image preview
            y_raster_size = int(self.cropped_image_preview.shape[1] / cfg.nuclei_segment_rand_nuclei_raster)
            x_raster_size = int(self.cropped_image_preview.shape[2] / cfg.nuclei_segment_rand_nuclei_raster)

            # add ellipses to preview image
            for x, x_coords in enumerate(self.rand_nuclei_raster):
                for y, count in enumerate(x_coords):
                    # calculate position for ellipses
                    coords = (
                        x * x_raster_size + (x_raster_size / 2),
                        y * y_raster_size + (y_raster_size / 2),
                    )

                    # calculate radius for circle depending on raster count
                    if count > 0:
                        width = (count / max_count) * x_raster_size
                        height = (count / max_count) * y_raster_size
                    else:
                        width = 0
                        height = 0

                    # create objects for patches
                    self.train_preview_patch_xy = Ellipse(coords,
                                                          width,
                                                          height,
                                                          color=cfg.nuclei_segment_overlay_colour,
                                                          alpha=cfg.nuclei_segment_overlay_alpha)

                    # add patch to preview
                    self.cropped_train_preview_xy.add_patch(self.train_preview_patch_xy)

            # update image
            self.cropped_train_preview_xy.figure.canvas.draw()

    def train_add_rand_patches(self):
        """
        Add random patches to the image for machine learning

        :return:
        """
        self.rand_patch_coords = list()
        self.rand_patch_imgs = list()

        # generate random patches coordinates
        for i in range(0, cfg.nuclei_segment_rand_patches):
            overlap = True

            while overlap is True:
                overlap = False

                rand_x = random.randint(0, (self.cropped_image_preview.shape[2]
                                            - cfg.nuclei_segment_rand_patches_size) - 1)
                rand_y = random.randint(0, (self.cropped_image_preview.shape[1]
                                            - cfg.nuclei_segment_rand_patches_size) - 1)

                max_rand_x = rand_x + cfg.nuclei_segment_rand_patches_size
                max_rand_y = rand_y + cfg.nuclei_segment_rand_patches_size

                # do they overlap with another patch?
                for patch in self.rand_patch_coords:
                    min_patch_x = patch[0]
                    max_patch_x = patch[0] + cfg.nuclei_segment_rand_patches_size + 1
                    min_patch_y = patch[1]
                    max_patch_y = patch[1] + cfg.nuclei_segment_rand_patches_size + 1

                    if rand_x in range(min_patch_x, max_patch_x) and rand_y in range(min_patch_y, max_patch_y):
                        overlap = True

                    if max_rand_x in range(min_patch_x, max_patch_x) and rand_y in range(min_patch_y, max_patch_y):
                        overlap = True

                    if rand_x in range(min_patch_x, max_patch_x) and max_rand_y in range(min_patch_y, max_patch_y):
                        overlap = True

                    if max_rand_x in range(min_patch_x, max_patch_x) and max_rand_y in range(min_patch_y, max_patch_y):
                        overlap = True

            # add to patches
            self.rand_patch_coords.append((rand_x, rand_y, max_rand_x, max_rand_y))

        # update image
        self.update_cropped_train_preview()

    def image_add_patches(self):
        """
        Add random patches to image

        :return:
        """
        if hasattr(self, 'rand_patch_coords') is True and self.rand_patch_coords is not None:
            # add patches to preview image
            for i, patch in enumerate(self.rand_patch_coords):
                self.fig_cropped_image_previews[i].clear()

                # create objects for patches
                self.train_preview_patch_xy = Rectangle(patch,
                                                        cfg.nuclei_segment_rand_patches_size,
                                                        cfg.nuclei_segment_rand_patches_size,
                                                        color=cfg.nuclei_segment_overlay_colour,
                                                        alpha=cfg.nuclei_segment_overlay_alpha)

                # add patch to preview
                self.cropped_train_preview_xy.add_patch(self.train_preview_patch_xy)

                # create image for patch
                self.rand_patch_imgs.append(self.cropped_image_preview[:,
                                            patch[1]:patch[1] + cfg.nuclei_segment_rand_patches_size,
                                            patch[0]:patch[0] + cfg.nuclei_segment_rand_patches_size])

                # draw image of patch
                Plot.show_image_preview(self.fig_cropped_image_previews[i],
                                        self.rand_patch_imgs[-1])
                self.cnv_cropped_image_previews[i].draw()

            # update image
            self.cropped_train_preview_xy.figure.canvas.draw()

    def train_save_rand_patches(self):
        """
        Save random patches as new input for segmentation

        :return:
        """
        self.image_save_settings(save_rand_patches=True)

    def train_save_rand_nuclei(self):
        """
        Save random nuclei as new input for segmentation

        :return:
        """
        self.image_save_settings(save_rand_nuclei=True)
