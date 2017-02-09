"""
Run through nuclei and correct their segmentation if necessary
"""

# Qt libraries
from PyQt4 import QtGui, QtCore

# threading for nucleus planes
import threading

import numpy as np

# matplot for Qt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# classic matplot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# nuclei libraries
from frontend.figures.plot import Plot
import frontend.gui.labels as gui_labels
from processing.segmentation import Segmentation
from processing.correction import Correction

import storage.config as cfg
import frontend.figures.thread as plot_thread


class NucleoSelect(QtGui.QDialog):

    TB_DRAW_NUCLEI = 0
    TB_SEL_NON_NUCLEI = 1
    TB_SEL_CORR_NUCLEI = 2
    TB_CORR_NUCLEI = 3

    RBG_MODE = ['SEL', 'ADD', 'DEL']
    RBG_TOOL = ['POINT', 'BRUSH']
    RBG_LAYER = ['NUCLEI', 'NONUC', 'FILA', 'FILTERED', 'ADDED', 'LABELS', 'OVERLAP', 'REMERGED']

    def __init__(self, image_info, parent=None):
        super(NucleoSelect, self).__init__(parent)

        # load image infos
        self.image_info = image_info

        # init segmentation
        self.segmentation = Segmentation(self.image_info)

        # load segmentation
        self.segmentation.load()

        # set correction
        self.correction = Correction(self.segmentation)

        # get nuclei
        self.nuclei = self.segmentation.nuclei
        
        # set variables
        self.cur_nID = -1
        self.prev_nID = -2
        self.pouch_pos_zoom = cfg.nucleus_select_non_nuclei_zoom
        self.draw_nuclei_zoom = cfg.nucleus_select_draw_nuclei_zoom
        self.draw_nuclei_z = int(self.segmentation.stacks.lamin.shape[0] / 2)
        self.draw_nuclei_mode = 0
        self.draw_nuclei_tool = 0
        self.draw_nuclei_layer = 0
        self.draw_nuclei_cur_layer = self.segmentation.stacks.nuclei

        # set specific mode parameters
        self.draw_nuclei_del_size = 20
        self.draw_nuclei_is_pressed = False

        # set draw nuclei params
        self.draw_nuclei_coords = [self.draw_nuclei_z, 0, 0]

        self.prev_nucleus = None

        self.nuclei_details = dict()

        self.fig_nucleus_planes = None
        self.cnv_nucleus_planes = None
        self.thr_nucleus_planes = None

        self.set_cur_nID(self.nuclei.get_nID_from_sID(0, only_accepted=True))

        # set main layout
        self.setLayout(self.prep_ctn_nucleus_edt())

        # set size
        self.setFixedSize(900, 700)

        # select first nucleus and details
        self.sort_nuclei_examples(cfg.filter_criteria_nuclei[0])

        # show first nucleus
        self.change_nuclei_edt(0)

        # prepare keyboard shortcuts
        self.set_key_shortcuts()

        # show short stats
        self.show_stats_summary()

    def show_stats_summary(self):
        """
        Show a brief summary of the stats

        :return:
        """
        print('Nuclei: %i' % len(self.segmentation.nuclei.get_nIDs(only_accepted=True)))
        print('Raw: %i' % len(self.segmentation.nuclei.get_nIDs(only_accepted=False)))
        print('Filtered: %i' % (len(self.correction.corr_filtered) if (self.correction.corr_filtered is not None) else 0))
        print('Removed: %i' % (len(self.correction.corr_nonuc) if (self.correction.corr_nonuc is not None) else 0))
        print('Corrected: %i' % (len(self.correction.corr_fila) if (self.correction.corr_fila is not None) else 0))
        print('Added: %i' % (len(self.correction.corr_remerge) if (self.correction.corr_remerge is not None) else 0))

    def set_key_shortcuts(self):
        """
        Set keyboard shortcuts

        :return:
        """
        # nucleus navigation
        self.set_key_shortcut(QtCore.Qt.Key_M, self.move_criteria_examples_down)
        self.set_key_shortcut(QtCore.Qt.Key_Period, self.show_nucleus_next)
        self.set_key_shortcut(QtCore.Qt.Key_Comma, self.show_nucleus_prev)
        self.set_key_shortcut(QtCore.Qt.Key_Slash, self.move_criteria_examples_up)

        # tab navigation
        self.set_key_shortcut(QtCore.Qt.Key_J, self.make_change_tab(self.TB_DRAW_NUCLEI))
        self.set_key_shortcut(QtCore.Qt.Key_K, self.make_change_tab(self.TB_SEL_NON_NUCLEI))
        self.set_key_shortcut(QtCore.Qt.Key_L, self.make_change_tab(self.TB_SEL_CORR_NUCLEI))
        self.set_key_shortcut(QtCore.Qt.Key_Semicolon, self.make_change_tab(self.TB_CORR_NUCLEI))

        # select non-nuclei
        self.set_key_shortcut(QtCore.Qt.Key_X, self.change_chk_is_nucleus)
        self.set_key_shortcut(QtCore.Qt.Key_Apostrophe, self.make_change_pouch_pos_zoom(-1))
        self.set_key_shortcut(QtCore.Qt.Key_Backslash, self.make_change_pouch_pos_zoom(1))

        # select nuclei for correction
        btns_select_for_corr_keys = [
            QtCore.Qt.Key_Q,
            QtCore.Qt.Key_W,
            QtCore.Qt.Key_E,
            QtCore.Qt.Key_R,
            QtCore.Qt.Key_A,
            QtCore.Qt.Key_S,
            QtCore.Qt.Key_D,
            QtCore.Qt.Key_F
        ]

        for i, key in enumerate(btns_select_for_corr_keys):
            self.set_key_shortcut(key, self.make_click_btn_select_for_corr(i))

        # correct nuclei
        self.set_key_shortcut(QtCore.Qt.Key_C, self.show_nucleus_planes)

        # save changes
        self.set_key_shortcut(QtCore.Qt.Key_V, self.save_nucleus_corr)

        # apply changes
        self.set_key_shortcut(QtCore.Qt.Key_B, self.apply_nucleus_corr)

    def set_key_shortcut(self, key, function):
        """
        Set shortcut

        :param key:
        :param function:
        :return:
        """
        self.connect(QtGui.QShortcut(
            QtGui.QKeySequence(key), self),
            QtCore.SIGNAL('activated()'),
            function)

    def make_click_btn_select_for_corr(self, buttonID):
        """
        Make function to click on button

        :param buttonID:
        :return:
        """
        def click_btn_select_for_corr():
            # simulate click
            self.btns_select_for_corr[buttonID].click()

        return click_btn_select_for_corr

    def change_chk_is_nucleus(self):
        """
        Toggle checkbox for nucleus

        :return:
        """
        checked = True

        if self.chk_is_nucleus.isChecked() is True:
            checked = False

        self.chk_is_nucleus.setChecked(checked)

        # save
        self.save_nucleus_corr(save_to_exp=False)

    def prep_ctn_nucleus_edt(self):
        """
        Create layout for edit nuclei

        :return:
        """
        container = QtGui.QGridLayout()

        # prepare tabs
        self.tb_draw_nuclei = QtGui.QWidget()
        self.tb_select_non_nuclei = QtGui.QWidget()
        self.tb_select_corr_nuclei = QtGui.QWidget()
        self.tb_correct_nuclei = QtGui.QWidget()

        self.tb_draw_nuclei.setLayout(self.prep_ctn_draw_nuclei())
        self.tb_select_non_nuclei.setLayout(self.prep_ctn_select_non_nuclei())
        self.tb_select_corr_nuclei.setLayout(self.prep_ctn_select_corr_nuclei())
        self.tb_correct_nuclei.setLayout(self.prep_ctn_correct_nuclei())

        # add tabs
        self.tbs_selection = QtGui.QTabWidget()

        self.tbs_selection.addTab(self.tb_draw_nuclei, gui_labels.tb_draw_nuclei)
        self.tbs_selection.addTab(self.tb_select_non_nuclei, gui_labels.tb_sel_non_nuclei)
        self.tbs_selection.addTab(self.tb_select_corr_nuclei, gui_labels.tb_sel_corr_nuclei)
        self.tbs_selection.addTab(self.tb_correct_nuclei, gui_labels.tb_correct_nuclei)

        self.tbs_selection.currentChanged.connect(self.change_nuclei_edt)

        # sorting nuclei list
        container.addLayout(self.prep_ctn_nuclei_sort(), 0, 0)

        # selection feature
        container.addWidget(self.tbs_selection, 1, 0, 1, 3)

        # nucleus information
        container.addLayout(self.prep_ctn_nucleus_select(), 2, 0)
        container.addLayout(self.prep_ctn_navigation(), 2, 1)

        return container

    def prep_ctn_nuclei_sort(self):
        """
        Prepare container to sort nuclei

        :return:
        """
        # selection to orders
        self.sel_nuclei_sort = QtGui.QComboBox()

        # cycle through criteria for sorting
        for criteria in cfg.filter_criteria_nuclei:
            self.sel_nuclei_sort.addItem(criteria)

        # add listener
        self.sel_nuclei_sort.activated[str].connect(self.sort_nuclei_examples)

        # create container
        container = QtGui.QGridLayout()

        container.addWidget(QtGui.QLabel(gui_labels.label_sort), 0, 0)
        container.addWidget(self.sel_nuclei_sort, 0, 1)
        container.setColumnStretch(2, 1)

        return container

    def change_nuclei_edt(self, tab_index):
        """
        Change nuclei edit criteria

        :param tab_index:
        :return:
        """
        # update tab corresponding to change
        self.cur_tab = tab_index

        # show nucleus parameters
        self.show_nucleus()

    def prep_ctn_correct_nuclei(self):
        """
        Container to correct nuclei

        :return:
        """
        container = QtGui.QGridLayout()

        container.addLayout(self.prep_ctn_nucleus_box(), 0, 1, 1, 1)
        container.addWidget(self.prep_ctn_nucleus_planes(), 0, 2, 4, 1)

        return container

    def prep_ctn_select_corr_nuclei(self):
        """
        Prepare container to change the order of nuclei

        :return:
        """
        # buttons
        self.btn_move_criteria_examples_up = QtGui.QPushButton(gui_labels.btn_up)
        self.btn_move_criteria_examples_down = QtGui.QPushButton(gui_labels.btn_down)

        # button listeners
        self.btn_move_criteria_examples_up.clicked.connect(self.move_criteria_examples_up)
        self.btn_move_criteria_examples_down.clicked.connect(self.move_criteria_examples_down)

        # example figure
        self.fig_nuclei_sort_hist = plt.figure(figsize=(3, 2))
        self.fig_nuclei_sort_examples = plt.figure(figsize=(5, 4))

        # canvases
        self.cnv_nuclei_sort_hist = FigureCanvas(self.fig_nuclei_sort_hist)
        self.cnv_nuclei_sort_examples = FigureCanvas(self.fig_nuclei_sort_examples)

        # slider
        self.sld_nuclei_sort = QtGui.QSlider(QtCore.Qt.Horizontal, self)

        # buttons to select nuclei for correction
        self.btns_select_for_corr = list()
        self.ctns_btns_select_for_corr = list()

        # array for list IDs that are called when the button is pushed
        self.lID_for_btn_select_for_corr = list()

        # create buttons for the example range
        for x in range(0, 2):
            # create containers
            self.ctns_btns_select_for_corr.append(QtGui.QGridLayout())

            for pos in range(0, cfg.nucleus_select_example_range):
                self.btns_select_for_corr.append(QtGui.QPushButton())
                self.btns_select_for_corr[-1].setEnabled(False)
                self.btns_select_for_corr[-1].clicked.connect(
                    self.make_change_to_correct_nucleus(x * cfg.nucleus_select_example_range + pos))

                # add to container
                self.ctns_btns_select_for_corr[-1].addWidget(self.btns_select_for_corr[-1], 0, pos)

                # add entry to list
                self.lID_for_btn_select_for_corr.append(-1)

        # build container
        container = QtGui.QGridLayout()

        container.addWidget(self.cnv_nuclei_sort_hist, 0, 1, 1, 2)

        container.addLayout(self.ctns_btns_select_for_corr[0], 1, 0, 1, 4)
        container.addWidget(self.cnv_nuclei_sort_examples, 2, 0, 2, 4)
        container.addLayout(self.ctns_btns_select_for_corr[1], 4, 0, 1, 4)

        container.addWidget(self.btn_move_criteria_examples_down, 5, 0)
        container.addWidget(self.sld_nuclei_sort, 5, 1, 1, 2)
        container.addWidget(self.btn_move_criteria_examples_up, 5, 3)

        return container

    def sort_nuclei_examples(self, criteria):
        """
        Sort examples of nuclei according to the chosen criteria

        :return:
        """
        self.cur_sort_criteria = criteria

        # clear figures
        self.fig_nuclei_sort_examples.clear()
        self.fig_nuclei_sort_hist.clear()

        # sort nuclei
        self.segmentation.nuclei.sort_nuclei(criteria.lower())

        # get sorted list
        sorted_params = self.segmentation.nuclei.get_param_list_from_nuclei(criteria.lower(), only_accepted=True)

        # show histogram
        Plot.view_histogram_of_value_list(self.fig_nuclei_sort_hist,
                                          sorted_params,
                                          cfg.criteria_select_hist_bins)

        # slider
        self.sld_nuclei_sort.valueChanged[int].connect(getattr(self, 'change_criteria_example_' + criteria.lower()))
        self.sld_nuclei_sort.setMinimum(0)
        self.sld_nuclei_sort.setMaximum(len(sorted_params))

        # draw canvas
        self.cnv_nuclei_sort_examples.draw()
        self.cnv_nuclei_sort_hist.draw()

    def change_criteria_example_volume(self, example_selected):
        """
        Change criteria example for volume

        :return:
        """
        self.change_criteria_example('volume', example_selected)

    def change_criteria_example_depth(self, example_selected):
        """
        Change criteria example for depth

        :return:
        """
        self.change_criteria_example('depth', example_selected)

    def change_criteria_example_surface(self, example_selected):
        """
        Change criteria example for surface

        :return:
        """
        self.change_criteria_example('surface', example_selected)

    def change_criteria_example_membrane_int(self, example_selected):
        """
        Change criteria example for membrane intensity

        :return:
        """
        self.change_criteria_example('membrane_int', example_selected)

    def change_criteria_example_dapi_int(self, example_selected):
        """
        Change criteria example for dapi intensity

        :return:
        """
        self.change_criteria_example('dapi_int', example_selected)

    def move_criteria_examples_down(self):
        self.move_criteria_examples(-1)

    def move_criteria_examples_up(self):
        self.move_criteria_examples(1)

    def move_criteria_examples(self, direction):
        """
        Move examples down

        :return:
        """
        # get value from slider
        cur_pos = self.sld_nuclei_sort.value()

        # calc new value
        if direction > 0:
            new_pos = cur_pos + int(cfg.nucleus_select_example_range * 2)
        else:
            new_pos = cur_pos - int(cfg.nucleus_select_example_range * 2)

        if new_pos < 0:
            new_pos = 0

        nIDs = self.nuclei.get_nIDs(only_accepted=True)
        if new_pos > len(nIDs):
            new_pos = len(nIDs)

        # change examples
        self.change_criteria_example(self.cur_sort_criteria.lower(), new_pos)

    def change_criteria_example(self, param, example_selected=None):
        """
        Change example in +/- range

        :param param:
        :param example_selected:
        :return:
        """
        if example_selected is None:
            example_selected = self.nuclei.get_sID_from_nID(self.cur_nID, only_accepted=True)

            # if the current nID is not in the list
            # take the first one
            if example_selected is None:
                example_selected = 0

        # set slider
        self.sld_nuclei_sort.setValue(example_selected)

        min_range = example_selected - cfg.nucleus_select_example_range

        if min_range < 0:
            min_range = 0

        max_range = example_selected + cfg.nucleus_select_example_range

        nIDs = self.nuclei.get_nIDs(only_accepted=True)

        if max_range > len(nIDs):
            max_range = len(nIDs)

        # prepare examples
        examples = list()

        # clear buttons
        for button in self.btns_select_for_corr:
            button.setText('')
            button.setEnabled(False)

        # reset list for list IDs
        self.lID_for_btn_select_for_corr = list()

        # show examples from each part
        counter = 0

        for i in range(min_range, max_range):
            # get nucleus
            nID = self.nuclei.get_nID_from_sID(i, only_accepted=True)

            # get image boxes
            #lamin_box = self.nuclei.get_img_boxes(nID, 'lamin')
            #lamin_slice = lamin_box[:, round(lamin_box.shape[1] / 2), :]
            info_img = self.nuclei.get_extra_infos(nID, 'img')
            lamin_slice = self.nuclei.get_extra_infos(nID, 'lamin_slice')

            example_title = '%i\n%.2f' % (nID, self.nuclei.get_param(param, nID))

            # prepare image
            examples = Plot.prepare_output(examples, info_img,
                                           example_title, cfg.criteria_select_eg_colour)
            examples = Plot.prepare_output(examples, lamin_slice,
                                           example_title, cfg.criteria_select_eg_lamin_colour)

            # update button
            self.btns_select_for_corr[counter].setText(str(nID))
            self.btns_select_for_corr[counter].setEnabled(True)

            # set lID
            self.lID_for_btn_select_for_corr.append(nID)

            counter += 1

        clear_figures = False

        # clear at end
        if (max_range + cfg.nucleus_select_example_range * 2) > len(nIDs):
            clear_figures = True

        # clear at beginning
        if min_range < cfg.nucleus_select_example_range * 2:
            clear_figures = True

        # clear figure every X nuclei
        if example_selected % cfg.nucleus_select_example_update_every < cfg.nucleus_select_example_range:
            clear_figures = True

        if clear_figures is True:
            self.fig_nuclei_sort_examples.clear()

        # show examples
        Plot.show_images(self.fig_nuclei_sort_examples, examples, cols=cfg.nucleus_select_example_range * 2)

        # update canvas
        self.cnv_nuclei_sort_examples.draw()

    def make_change_to_correct_nucleus(self, btn_id):
        """
        Make function to change to correct nucleus

        :param btn_id:
        :return:
        """
        def made_change_to_correct_nucleus():
            # set nucleus
            self.cur_nID = self.lID_for_btn_select_for_corr[btn_id]

            # change tab
            self.change_tab(self.TB_CORR_NUCLEI)

        return made_change_to_correct_nucleus

    def make_change_tab(self, tab):
        """
        Make change tab function

        :param tab:
        :return:
        """
        def made_change_tab():
            self.change_tab(tab)

        return made_change_tab

    def change_tab(self, tab):
        """
        Change to another tab

        :param tab:
        :return:
        """
        # change tab
        self.tbs_selection.setCurrentIndex(tab)

    def prep_ctn_select_non_nuclei(self):
        """
        Prepare container to show position of nucleus
        in the pouch image

        :return:
        """
        container = QtGui.QGridLayout()

        # prepare figure and canvas
        self.fig_pouch_pos = plt.figure(figsize=(5,5))
        self.cnv_pouch_pos = FigureCanvas(self.fig_pouch_pos)

        # zoom slider
        self.sld_pouch_pos_zoom = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.sld_pouch_pos_zoom.setMinimum(1)
        self.sld_pouch_pos_zoom.setMaximum(cfg.nucleus_select_non_nuclei_zoom_max)
        self.sld_pouch_pos_zoom.setValue(self.pouch_pos_zoom)
        self.sld_pouch_pos_zoom.valueChanged[int].connect(self.change_pouch_pos_zoom)

        # prepare checkbox
        self.chk_is_nucleus = QtGui.QCheckBox(gui_labels.nuc_is_nucleus)

        # add figure
        container.addWidget(self.cnv_pouch_pos, 0, 0)
        container.addWidget(self.sld_pouch_pos_zoom, 0, 1)
        container.addWidget(self.chk_is_nucleus, 1, 0)

        container.setColumnStretch(0, 1)

        return container

    def make_change_pouch_pos_zoom(self, direction):
        """
        Make function to change zoom in direction

        :param direction:
        :return:
        """
        def change_pouch_pos_zoom():
            # set zoom

            zoom = self.pouch_pos_zoom + direction

            # correct zoom
            if zoom < 1:
                zoom = 1
            if zoom > cfg.nucleus_select_non_nuclei_zoom_max:
                zoom = cfg.nucleus_select_non_nuclei_zoom_max

            # set slider
            self.sld_pouch_pos_zoom.setValue(zoom)

            # set zoom
            self.change_pouch_pos_zoom(zoom)

        return change_pouch_pos_zoom

    def change_pouch_pos_zoom(self, zoom):
        """
        Change zoom of pouch position

        :param zoom:
        :return:
        """
        # change zoom
        self.pouch_pos_zoom = zoom

        # reload image
        self.show_pouch_pos()

    def prep_ctn_draw_nuclei(self):
        """
        Prepare container to show drawing interface for nuclei

        :return:
        """
        container = QtGui.QGridLayout()

        # prepare figure and canvas
        self.fig_draw_nuclei = plt.figure(figsize=(5,5))
        self.cnv_draw_nuclei = FigureCanvas(self.fig_draw_nuclei)

        # zoom slider
        self.sld_draw_nuclei_zoom = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.sld_draw_nuclei_zoom.setMinimum(1)
        self.sld_draw_nuclei_zoom.setMaximum(cfg.nucleus_select_draw_nuclei_zoom_max)
        self.sld_draw_nuclei_zoom.setValue(self.draw_nuclei_zoom)
        self.sld_draw_nuclei_zoom.valueChanged[int].connect(self.change_draw_nuclei_zoom)

        # Z slider
        self.sld_draw_nuclei_z = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.sld_draw_nuclei_z.setMinimum(0)
        self.sld_draw_nuclei_z.setMaximum(self.segmentation.stacks.lamin.shape[0] - 1)
        self.sld_draw_nuclei_z.setValue(self.draw_nuclei_z)
        self.sld_draw_nuclei_z.valueChanged[int].connect(self.change_draw_nuclei_z)

        # add figure
        container.addLayout(self.prep_ctn_draw_nuclei_tools(), 0, 0)
        container.addWidget(self.cnv_draw_nuclei, 0, 1, 2, 1)
        container.addWidget(self.sld_draw_nuclei_z, 0, 2, 2, 1)
        container.addWidget(self.sld_draw_nuclei_zoom, 0, 3, 2, 1)

        container.setColumnStretch(1, 1)
        container.setRowStretch(1, 1)

        return container

    def prep_ctn_draw_nuclei_tools(self):
        """
        Prepare container for tools of draw nuclei

        :return:
        """
        container = QtGui.QGridLayout()

        # add radio button groups
        container.addLayout(self.prep_ctn_draw_nuclei_rbg('mode'), 0, 0)
        container.addLayout(self.prep_ctn_draw_nuclei_rbg('tool'), 1, 0)
        container.addLayout(self.prep_ctn_draw_nuclei_rbg('layer'), 2, 0)

        return container

    def prep_ctn_draw_nuclei_rbg(self, var_name):
        """
        Prepare container for radio button group

        :param var_name:
        :return:
        """
        container = QtGui.QGridLayout()

        container.addWidget(QtGui.QLabel(getattr(gui_labels, 'rb_draw_nuclei_' + var_name)), 0, 0)

        # button group
        rbg_group = QtGui.QButtonGroup(container)

        rbs = list()
        # create selection buttons
        for i, cur_desc in enumerate(getattr(self, 'RBG_' + var_name.upper())):
            rbs.append(QtGui.QRadioButton(getattr(gui_labels,
                                                  'rb_draw_nuclei_'
                                                  + var_name
                                                  + '_' + cur_desc.lower())))

            if i == 0:
                rbs[-1].setChecked(True)

            # listener
            rbs[-1].clicked.connect(self.make_change_rb_draw_nuclei(var_name, i))

            # add to group
            rbg_group.addButton(rbs[-1])

            # add to container
            container.addWidget(rbs[-1], i + 1, 0)

        return container

    def make_change_rb_draw_nuclei(self, var_name, index):
        """
        Make change radio button for draw nuclei

        :param var_storage:
        :param index:
        :return:
        """
        def change_rb_draw_nuclei():
            # set variable
            setattr(self, 'draw_nuclei_' + var_name, index)

            # draw image
            self.show_draw_nuclei()

        return change_rb_draw_nuclei

    def get_draw_nuclei_cur_layer(self):
        """
        return current layer

        :return:
        """
        self.draw_nuclei_cur_layer = self.segmentation.stacks.nuclei

        if self.draw_nuclei_layer == self.RBG_LAYER.index('NONUC'):
            self.draw_nuclei_cur_layer = self.correction.stacks.nonuc
        elif self.draw_nuclei_layer == self.RBG_LAYER.index('FILA'):
            self.draw_nuclei_cur_layer = self.correction.stacks.fila
        elif self.draw_nuclei_layer == self.RBG_LAYER.index('FILTERED'):
            self.draw_nuclei_cur_layer = self.correction.stacks.filtered
        elif self.draw_nuclei_layer == self.RBG_LAYER.index('ADDED'):
            self.draw_nuclei_cur_layer = self.correction.stacks.added
        elif self.draw_nuclei_layer == self.RBG_LAYER.index('LABELS'):
            self.draw_nuclei_cur_layer = self.segmentation.stacks.labels
        elif self.draw_nuclei_layer == self.RBG_LAYER.index('OVERLAP'):
            self.draw_nuclei_cur_layer = self.correction.stacks.overlap
        elif self.draw_nuclei_layer == self.RBG_LAYER.index('REMERGED'):
            self.draw_nuclei_cur_layer = self.correction.stacks.remerge

        return self.draw_nuclei_cur_layer

    def change_draw_nuclei_z(self, z):
        """
        Change z of draw nuclei position

        :param z:
        :return:
        """
        # change zoom
        self.draw_nuclei_z = z

        # reload image
        self.show_draw_nuclei()

    def change_draw_nuclei_zoom(self, zoom):
        """
        Change zoom of draw nuclei position

        :param zoom:
        :return:
        """
        # change zoom
        self.draw_nuclei_zoom = zoom

        # reload image
        self.show_draw_nuclei()

    def prep_ctn_nucleus_box(self):
        """
        Prepare container for nucleus box

        :return:
        """
        container = QtGui.QGridLayout()

        # prepare figure and canvas
        self.fig_nucleus_box = plt.figure(figsize=(5, 5))
        self.cvn_nucleus_box = FigureCanvas(self.fig_nucleus_box)

        # add figure
        container.addWidget(self.cvn_nucleus_box, 0, 0)

        # add edit boxes
        container.addLayout(self.prep_ctn_nucleus_edt_Z(), 1, 0)

        container.setColumnStretch(0, 1)
        container.setRowStretch(0, 1)

        return container

    def prep_ctn_nucleus_planes(self):
        """
        Prepare container for nucleus planes

        :return:
        """
        self.ctn_nucleus_planes = QtGui.QScrollArea()

        #self.ctn_nucleus_planes.setGeometry(QtCore.QRect(0, 0, 300, 800))
        #container.setFrameStyle(QtGui.QFrame.NoFrame)

        return self.ctn_nucleus_planes

    def add_to_ctn_nucleus_planes(self):
        """
        Add a nucleus segmentation to the plane container

        :param nID:
        :return:
        """
        # how many planes?
        planes = self.stack_boxes.nuclei.shape[0]

        # create figure
        self.fig_nucleus_planes = plt.figure(
            figsize=(3 * cfg.nucleus_planes_plane_size, planes * cfg.nucleus_planes_plane_size))
        self.cnv_nucleus_planes = FigureCanvas(self.fig_nucleus_planes)

        # add to container
        self.ctn_nucleus_planes.setWidget(self.cnv_nucleus_planes)

        # stop current planes
        if self.thr_nucleus_planes is not None:
            plot_thread.thread = False
            self.thr_nucleus_planes.join()
            plot_thread.thread = True

        self.thr_nucleus_planes = threading.Thread(target=self.plot_nucleus_planes)
        self.thr_nucleus_planes.start()

    def plot_nucleus_planes(self):
        """
        Plot nucleus plane

        :return:
        """
        # plot nuclei
        Plot.show_nucleus_planes(self.fig_nucleus_planes,
                                 self.stack_boxes,
                                 self.segmentation.nuclei.get_nucleus_centroids(self.cur_nID))

        self.cnv_nucleus_planes.draw()

    def prep_ctn_nucleus_edt_Z(self):
        """
        Prepare container to edit nucleus Z start and stop

        :return:
        """
        # prepare nucleus information and edit possibilities
        container = QtGui.QGridLayout()

        # labels
        container.addWidget(QtGui.QLabel(gui_labels.nuc_z_start), 0, 0)
        container.addWidget(QtGui.QLabel(gui_labels.nuc_z_stop), 0, 2)

        # edit boxes and buttons
        self.edt_z_start = QtGui.QLineEdit()
        self.edt_z_stop = QtGui.QLineEdit()
        self.btn_details = QtGui.QPushButton(gui_labels.btn_details)
        self.btn_details.clicked.connect(self.show_nucleus_planes)

        container.addWidget(self.edt_z_start, 0, 1)
        container.addWidget(self.edt_z_stop, 0, 3)
        container.addWidget(self.btn_details, 0, 4)

        return container

    def prep_ctn_nucleus_select(self):
        """
        Prepare container to select a nucleus

        :return:
        """
        # prepare nucleus information and edit possibilities
        container = QtGui.QGridLayout()

        # nucleus ID
        self.lbl_nID = QtGui.QLabel(gui_labels.selection_none)
        container.addWidget(QtGui.QLabel(gui_labels.nuc_id), 0, 0)
        container.addWidget(self.lbl_nID, 0, 1)

        # create edit field
        self.edt_select_nID = QtGui.QLineEdit()
        container.addWidget(self.edt_select_nID, 0, 2)

        # create button
        self.btn_nucleus_select = QtGui.QPushButton(gui_labels.btn_select)

        # connect to listener
        self.btn_nucleus_select.clicked.connect(self.nucleus_select)

        # add to container
        container.addWidget(self.btn_nucleus_select, 0, 3)

        return container

    def prep_ctn_navigation(self):
        """
        Prepare container for navigation

        :return:
        """
        container = QtGui.QGridLayout()

        # create buttons
        self.btn_next = QtGui.QPushButton(gui_labels.btn_next)
        self.btn_prev = QtGui.QPushButton(gui_labels.btn_prev)

        self.btn_save_corr = QtGui.QPushButton(gui_labels.btn_save)
        self.btn_apply_corr = QtGui.QPushButton(gui_labels.btn_apply)
        self.btn_save_to_disk = QtGui.QPushButton(gui_labels.btn_save_to_disk)

        # add event listener
        self.btn_next.clicked.connect(self.show_nucleus_next)
        self.btn_prev.clicked.connect(self.show_nucleus_prev)

        self.btn_save_corr.clicked.connect(self.save_nucleus_corr)
        self.btn_apply_corr.clicked.connect(self.apply_nucleus_corr)
        self.btn_save_to_disk.clicked.connect(self.save_to_disk)

        # add to container
        container.addWidget(self.btn_next, 0, 0)
        container.addWidget(self.btn_prev, 0, 1)
        container.addWidget(self.btn_save_corr, 0, 2)
        container.addWidget(self.btn_apply_corr, 0, 3)
        container.addWidget(self.btn_save_to_disk, 0, 4)

        # set next as default
        self.btn_next.setDefault(True)

        return container

    def show_nucleus_next(self):
        self.show_nucleus(1)

    def show_nucleus_prev(self):
        self.show_nucleus(-1)

    def show_nucleus(self, direction=0):
        """
        Show nucleus in the direction indicated

        :param direction:
        :return:
        """

        # calculate next nucleus
        self.set_cur_nID(self.get_next_nID(self.cur_nID, direction))

        # set nID text box
        self.edt_select_nID.setText(str(self.cur_nID))

        # select new nucleus
        self.nucleus_select()

    def get_next_nID(self, nID=None, direction=0):
        """
        Get the next nucleus in the sorted sequence

        :param nID:
        :param direction:
        :return:
        """
        next_nID = None

        if nID is None:
            nID = self.cur_nID

        # get sequence ID of nucleus
        sID = self.nuclei.get_sID_from_nID(nID, only_accepted=True)

        if sID is not None:
            # calc next nucleus
            if direction > 0:
                sID = sID + 1
            elif direction < 0:
                sID = sID - 1

            # get next ID
            next_nID = self.nuclei.get_nID_from_sID(sID, only_accepted=True)

        if next_nID is None:
            next_nID = self.cur_nID

        return next_nID

    def update_nucleus_fields(self):
        """
        Update edit fields for nucleus

        :return:
        """
        # is it a nucleus?
        if self.correction.is_correction_nonuc(self.cur_nID) is False:
            self.chk_is_nucleus.setChecked(True)
        else:
            self.chk_is_nucleus.setChecked(False)

        # start and stop
        self.edt_z_start.setText(str(
            int(float(self.segmentation.nuclei.get_nucleus_centroids(self.cur_nID)[0, 0]))
        ))
        self.edt_z_stop.setText(str(
            int(float(self.segmentation.nuclei.get_nucleus_centroids(self.cur_nID)[-1, 0]))
        ))

        # TODO how to best return a validated collection for the params
        # get validated params
        #nucleus_params = self.segmentation.get_validated_params_for_nucleus(self.cur_nID)

        # show validated params
        #print('nucleus params')
        #for nucleus_filter in nucleus_params[0].keys():

        #    print('\t %s: %.2f %r' % (nucleus_filter,
        #                              nucleus_params[0][nucleus_filter]['value'],
        #                              nucleus_params[0][nucleus_filter]['error'])
        #          )

        #print('labels params')
        #for labels_filter in nucleus_params[1].keys():
        #    print('\t %s: min:%.2f max:%.2f %r' % (labels_filter,
        #                                           nucleus_params[1][labels_filter]['min'],
        #                                           nucleus_params[1][labels_filter]['max'],
        #                                           nucleus_params[1][labels_filter]['error'])
        #          )

    def set_cur_nID(self, id=0):
        """
        Set the current nucleus

        :param id:
        :return:
        """
        valid_id = False

        # is the id in limit?
        if id < len(self.nuclei.get_nIDs()) and id >= 0:
            valid_id = True

            # set current as previous nucleus
            self.prev_nID = self.cur_nID

            # set ID and nucleus
            self.cur_nID = id

        return valid_id

    def nucleus_select(self):
        """
        Show nucleus with the specfied nID

        :return:
        """

        if self.set_cur_nID(int(self.edt_select_nID.text())) is True:
            self.lbl_nID.setText(str(self.cur_nID))

        # which tab is currently shown?
        if self.cur_tab == self.TB_DRAW_NUCLEI:
            # show pouch position
            self.show_draw_nuclei()
        elif self.cur_tab == self.TB_SEL_NON_NUCLEI:
            # show pouch position
            self.show_pouch_pos()
        elif self.cur_tab == self.TB_SEL_CORR_NUCLEI:
            # update examples
            self.change_criteria_example(self.cur_sort_criteria.lower())
        elif self.cur_tab == self.TB_CORR_NUCLEI:
            # show nucleus box
            self.show_nucleus_boxes()

        # update nucleus fields
        self.update_nucleus_fields()

    def show_draw_nuclei(self, patches=None):
        """
        Show interface to draw nuclei

        :return:
        """
        lamin_img = list()

        # get current layer
        layer = self.get_draw_nuclei_cur_layer()

        layer_img = list()

        # get z of nucleus
        lamin_img = Plot.prepare_output(lamin_img, self.segmentation.stacks.lamin[self.draw_nuclei_z],
                                        cmap='gray', zoom=self.draw_nuclei_zoom)
        if layer is not None:
            layer_img = Plot.prepare_output(layer_img, layer[self.draw_nuclei_z],
                                            cmap='Dark2', zoom=self.draw_nuclei_zoom)

        # clear figure
        self.fig_draw_nuclei.clear()

        # show image
        axes = Plot.show_images(self.fig_draw_nuclei, lamin_img, overlay=layer_img)[0]

        # add listener
        axes.figure.canvas.mpl_connect('button_press_event',
                                          self.draw_nuclei_on_press)
        axes.figure.canvas.mpl_connect('motion_notify_event',
                                          self.draw_nuclei_on_motion)
        axes.figure.canvas.mpl_connect('button_release_event',
                                          self.draw_nuclei_on_release)

        # add patches
        if patches is not None:
            for patch in patches:
                axes.add_patch(patch)

        self.cnv_draw_nuclei.draw()

    def draw_nuclei_mpl_get_pos(self, event):
        """
        Return position based on an event

        :param event:
        :return:
        """
        pos = None

        if event is not None and event.xdata is not None and event.ydata is not None:
            pos = [
                self.draw_nuclei_z,
                int(event.ydata),
                int(event.xdata)
            ]

        return pos

    def draw_nuclei_on_release(self, event):
        """
        Draw nuclei mouse release

        :return:
        """
        self.draw_nuclei_is_pressed = False

    def draw_nuclei_on_press(self, event):
        """
        Draw nuclei mouse pressed

        :return:
        """
        self.draw_nuclei_is_pressed = True

        # get position
        pos = self.draw_nuclei_mpl_get_pos(event)

        # which mode?
        if pos is not None:
            if self.draw_nuclei_mode == self.RBG_MODE.index('SEL'):
                self.draw_nuclei_select_nucleus(pos)
            elif self.draw_nuclei_mode == self.RBG_MODE.index('DEL'):
                self.draw_nuclei_eval_delete_nuclei(pos)
            elif self.draw_nuclei_mode == self.RBG_MODE.index('ADD'):
                self.draw_nuclei_add_nucleus(pos)

    def draw_nuclei_on_motion(self, event):
        """
        Draw nuclei mouse pressed

        :return:
        """
        # get position
        pos = self.draw_nuclei_mpl_get_pos(event)

        # which mode?
        if pos is not None:
            if self.draw_nuclei_mode == self.RBG_MODE.index('SEL'):
                pass
            elif self.draw_nuclei_mode == self.RBG_MODE.index('DEL'):
                # draw rectangle
                #self.draw_nuclei_cursor_rect(pos[1:], self.draw_nuclei_del_size)

                # delete nuclei
                if self.draw_nuclei_is_pressed is True:
                    self.draw_nuclei_eval_delete_nuclei(pos)

    def draw_nuclei_eval_delete_nuclei(self, pos):
        """
        Evaluate whether to delete or undo

        :param pos:
        :return:
        """
        # delete if on nuclei layer otherwise undo
        if self.draw_nuclei_layer == self.RBG_LAYER.index('NUCLEI'):
            deleted_nuclei = self.draw_nuclei_delete_nuclei(pos)

            # were nuclei added?
            for nucleus in deleted_nuclei:
                # this method will delete the nucleus from the list if present
                print('delete nucleus %i' % nucleus)
                self.correction.del_correction_added(nucleus)
        else:
            self.draw_nuclei_undo_delete_nuclei(pos)

        # apply corrections
        #self.apply_nucleus_corr()

    def draw_nuclei_cursor_rect(self, xy_pos, size):
        """
        Draw a rectangle at cursor position

        :param pos:
        :return:
        """
        rect_pos = (xy_pos[1] - size/2, xy_pos[0] - size/2)

        draw_nuclei_patch = Rectangle(rect_pos,
                                      size, size,
                                      color=cfg.nuclei_segment_overlay_colour,
                                      alpha=cfg.nuclei_segment_overlay_alpha)

        # add patch to preview
        self.show_draw_nuclei(patches=[draw_nuclei_patch])

    def draw_nuclei_delete_nuclei(self, pos):
        """
        Delete nuclei based on position

        :param pos:
        :return:
        """
        pos_range = NucleoSelect.calc_rect_at_pos(pos, self.draw_nuclei_del_size)

        # lookup nuclei
        selected_nuclei = self.segmentation.nuclei.get_nID_by_pos_range(pos_range)

        # delete nuclei
        deleted_nuclei = list()
        if selected_nuclei is not None:
            for nucleus in selected_nuclei:
                self.correction.add_correction_nonuc(nucleus)

                deleted_nuclei.append(nucleus)

        return deleted_nuclei

    def draw_nuclei_undo_delete_nuclei(self, pos):
        """
        Delete nuclei based on position

        :param pos:
        :return:
        """
        # TODO this method does not work properly

        # calc range
        pos_range = NucleoSelect.calc_rect_at_pos(pos, self.draw_nuclei_del_size)

        # lookup nuclei
        selected_nuclei = self.segmentation.nuclei.get_nID_by_pos_range(pos_range)

        # delete nuclei
        if selected_nuclei is not None:
            for nucleus in selected_nuclei:
                if self.draw_nuclei_layer == self.RBG_LAYER.index('NONUC'):
                    self.correction.del_correction_nonuc(nucleus)
                elif self.draw_nuclei_layer == self.RBG_LAYER.index('FILA'):
                    self.draw_nuclei_delete_nuclei(pos)
                    self.correction.del_correction_fila(nucleus)
                elif self.draw_nuclei_layer == self.RBG_LAYER.index('FILTERED'):
                    self.correction.del_correction_filtered(nucleus)
                elif self.draw_nuclei_layer == self.RBG_LAYER.index('ADDED'):
                    self.draw_nuclei_delete_nuclei(pos)
                    self.correction.del_correction_added(nucleus)

        # update correction
        # TODO: Takes a long time
        #self.correction.save_corrections()

    @staticmethod
    def calc_rect_at_pos(pos, size):
        """
        Return rectangle boundaries at a position

        :param pos:
        :param size:
        :return:
        """
        # calc range
        pos_range = [
            pos[0],  # min
            pos[1] - int(size / 2),
            pos[2] - int(size / 2),
            pos[0],  # max
            pos[1] + int(size / 2),
            pos[2] + int(size / 2)
        ]

        return pos_range

    def draw_nuclei_select_nucleus(self, pos):
        """
        Select nucleus based on position

        :param pos:
        :return:
        """

        # lookup nucleus
        nID = self.segmentation.nuclei.get_nID_by_pos(pos)

        if nID is not None and nID >= 0:
            # set current nucleus
            self.set_cur_nID(nID)

            # change tab
            self.change_tab(self.TB_CORR_NUCLEI)

    def draw_nuclei_add_nucleus(self, pos):
        """
        Add nucleus based on position

        :param pos:
        :return:
        """

        # lookup nucleus in nuclei
        nucleus = self.segmentation.nuclei.get_nID_by_pos(pos)

        # lookup nucleus in raw nuclei
        raw_nucleus = self.segmentation.nuclei.get_nID_by_pos(pos, only_accepted=False)

        # add raw nucleus to nuclei
        if raw_nucleus is not None:
            # delete from non-nuclei
            if self.correction.is_correction_nonuc(raw_nucleus) is True:
                self.correction.del_correction_nonuc(raw_nucleus)

                print('nucleus recovered from nonuc')
            else:  # delete from filtered nuclei
                self.correction.del_correction_filtered(raw_nucleus)

                print('nucleus recovered from filtered')
        elif nucleus is None:  # if there is no in raw lookup raw label at position
            raw_label = self.segmentation.get_raw_label_by_pos(pos)

            # create a one plane nucleus
            if raw_label is not None:
                one_plane_nID = self.segmentation.nuclei.create_nucleus(pos[0], raw_label)

                # add to correction list
                self.correction.add_correction_remerge(one_plane_nID)

                print('nucleus added for remerge')

                # merge label up and down in the raw labels props stack
                #self.segmentation.nuclei.remerge_nucleus(one_plane_nID,
                #                                         0, (self.segmentation.stacks.lamin.shape[0] - 1),
                #                                         merge_depth=True, force_raw_labels_props=True)

                # Does the nucleus overlap with another nucleus?
                # TODO calculate overlap with other nuclei if needed
                # overlapping_nuclei = self.segmentation.nuclei.get_overlapping_nuclei(one_plane_nID)

                # FIX: ignore overlap and correct by hand by deleting the wrong nuclei
                # first and then remerging the new one
                overlapping_nuclei = None

                #print('overlapping nuclei %i' % len(overlapping_nuclei))

                if overlapping_nuclei is not None:
                    for overlapping_nucleus in overlapping_nuclei:
                        # add to overlap
                        self.correction.add_correction_overlap(overlapping_nucleus)

                        # add to nonuc
                        self.correction.add_correction_nonuc(overlapping_nucleus)

                # add nucleus to correction
                self.correction.add_correction_added(one_plane_nID)

        # apply corrections
        # takes a long time to execute
        # self.apply_nucleus_corr()

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

    def show_pouch_pos(self):
        """
        Show position of nucleus in the pouch

        :return:
        """
        pouch_img = list()

        nucleus_centroids = self.segmentation.nuclei.get_nucleus_centroids(self.cur_nID)

        # get z of nucleus
        nucleus_pos_z = round((nucleus_centroids[0, 0] + nucleus_centroids[-1, 0])/2)
        nucleus_pos_middle = round(len(nucleus_centroids)/2)
        nucleus_pos_xy = (nucleus_centroids[nucleus_pos_middle, 2] - 3,
                          nucleus_centroids[nucleus_pos_middle, 1] + 3)

        pouch_img = Plot.prepare_output(pouch_img, self.segmentation.stacks.lamin[nucleus_pos_z],
                                        'X', 'gray', text_pos=nucleus_pos_xy, zoom=self.pouch_pos_zoom)

        # clear figure
        self.fig_pouch_pos.clear()

        # show image
        Plot.show_images(self.fig_pouch_pos, pouch_img)

        self.cnv_pouch_pos.draw()

    def show_nucleus_boxes(self):
        """
        Show nucleus boxes and enable to cycle through the list

        :param direction:
        :return:
        """
        # clear figure
        self.fig_nucleus_box.clear()

        nucleus_boxes = self.segmentation.nuclei.get_img_boxes(self.cur_nID)
        nucleus_centroids = self.segmentation.nuclei.get_nucleus_centroids(self.cur_nID)
        nucleus_areas = self.segmentation.nuclei.get_nucleus_areas(self.cur_nID)

        # show the nucleus box
        self.stack_boxes = Plot.show_nucleus_box(self.fig_nucleus_box,
                                                 nucleus_boxes, nucleus_centroids, nucleus_areas,
                                                 self.segmentation.stacks)

        # refresh canvas
        self.cvn_nucleus_box.draw()

        # show planes
        if cfg.nucleus_planes_autoload is True:
            self.show_nucleus_planes()

    def show_nucleus_planes(self):
        """
        Show planes of current nucleus

        :return:
        """
        # show the segmentation per plane
        self.add_to_ctn_nucleus_planes()

    def save_nucleus_corr(self, save_to_exp=True):
        """
        Save correction for z

        :return:
        """
        # delete nucleus
        if self.chk_is_nucleus.isChecked() is False:
            print('Save nucleus deletion %i' % self.cur_nID)

            self.correction.add_correction_nonuc(self.cur_nID)
        else:
            print('Save nucleus correction %i' % self.cur_nID)

            # make sure the nucleus is not in the nonuc list
            self.correction.del_correction_nonuc(self.cur_nID)

            # correct nucleus
            self.correction.add_correction_fila(self.cur_nID,
                                                int(self.edt_z_start.text()),
                                                int(self.edt_z_stop.text()))

        # save corrections to experiment
        if save_to_exp is True:
            self.correction.save_corrections()

    def apply_nucleus_corr(self, save=False):
        """
        Apply correction for z

        :return:
        """
        # save corrections
        self.correction.save_corrections()

        # apply corrections and do not save to disk
        self.correction.apply_corrections(save=save)

        # reload drawing
        self.show_draw_nuclei()

    def save_to_disk(self):
        """
        Save changes and corrections to disk

        :return:
        """

        self.apply_nucleus_corr(save=True)
