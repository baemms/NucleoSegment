"""
Parent class to build selection criteria mask
"""

import matplotlib.pyplot as plt

# matplot for Qt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# Qt libraries
from PyQt4 import QtGui, QtCore

import frontend.gui.labels as gui_labels
import storage.config as cfg
from frontend.figures.plot import Plot
from processing.segmentation import Segmentation

class CriteriaSelection(QtGui.QDialog):

    def __init__(self, image_info, force_nuclei_load=False, parent=None):
        super(CriteriaSelection, self).__init__(parent)

        # nuclei selection?
        if not hasattr(self, 'is_nuclei_selection'):
            self.is_nuclei_selection = False

        # load image infos
        self.image_info = image_info

        # prepare criteria
        self.criteria = dict()

        # init segmentation
        self.segmentation = Segmentation(self.image_info)

        # load segmentation
        self.segmentation.load(force_nuclei_load=force_nuclei_load)

        # buttons
        self.btn_reset = QtGui.QPushButton(gui_labels.btn_reset)
        self.btn_load = QtGui.QPushButton(gui_labels.btn_load)
        self.btn_save = QtGui.QPushButton(gui_labels.btn_save)
        self.btn_close = QtGui.QPushButton(gui_labels.btn_close)

        # listeners
        self.btn_reset.clicked.connect(self.reset_criteria)
        self.btn_load.clicked.connect(self.load_criteria)
        self.btn_save.clicked.connect(self.save_criteria)
        self.btn_close.clicked.connect(self.close)

    def prep_ctn_criteria_select(self, lookup_nucleus):
        """
        Prepare main container for criteria selection

        :return:
        """
        container = QtGui.QGridLayout()

        # select criteria
        criteria = cfg.filter_criteria_labels

        if lookup_nucleus is True:
            criteria = cfg.filter_criteria_nuclei

        counter = 0
        for param in criteria:
            # get container
            param_ctn = self.prep_criteria(param.lower(), lookup_nucleus=lookup_nucleus)

            # add container
            if param_ctn is not None:
                container.addLayout(param_ctn, counter, 0)

                counter += 1

        # submit buttons
        container.addLayout(self.prep_ctn_submit(), counter, 0)

        return container

    def reset_criteria(self):
        """
        Reset edit boxes

        :return:
        """
        # go through edit boxes and empty
        for criteria_raw in self.criteria:
            self.criteria[criteria_raw]['edt_min'].setText('')
            self.criteria[criteria_raw]['edt_max'].setText('')

    def load_criteria(self):
        """
        Load selection criteria

        :return:
        """
        # load criteria
        self.loaded_criteria = self.segmentation.get_nuclei_criteria(True)

        # set min/max fields for critieria
        for criteria_raw in self.criteria:
            # lower case criteria
            criteria_upper = criteria_raw.upper()

            # edit boxes
            if self.loaded_criteria is not None and criteria_upper in self.loaded_criteria.keys():
                # set value in edit box
                if self.loaded_criteria[criteria_upper]['MIN'] is not None:
                    self.criteria[criteria_raw]['edt_min'].setText(str(self.loaded_criteria[criteria_upper]['MIN']))

                if self.loaded_criteria[criteria_upper]['MAX'] is not None:
                    self.criteria[criteria_raw]['edt_max'].setText(str(self.loaded_criteria[criteria_upper]['MAX']))

    def save_criteria(self):
        """
        Save selection criteria for labels

        :return:
        """
        # create criteria mapping
        criteria = dict()

        # go through criteria and get the values
        for param in self.loaded_criteria.keys():

            # is criteria on mask?
            if param.lower() in self.criteria.keys():
                # get min and max
                min_input = self.criteria[param.lower()]['edt_min'].text()
                max_input = self.criteria[param.lower()]['edt_max'].text()

                min_value = None
                if min_input:
                    min_value = '%.1f' % float(min_input)

                max_value = None
                if max_input:
                    max_value = '%.1f' % float(max_input)
            else:
                min_value = self.loaded_criteria[param]['MIN']
                max_value = self.loaded_criteria[param]['MAX']

            # write to mapping
            criteria[param] = dict()
            criteria[param]['MIN'] = min_value
            criteria[param]['MAX'] = max_value

        # set criteria
        self.loaded_criteria = criteria

        # save to file
        self.segmentation.save_nuclei_criteria(criteria)

    def close(self):
        """
        Close criteria selection and show the main window again

        :return:
        """
        # show parent
        self.parent().showNormal()

        # close processing
        super(QtGui.QDialog, self).close()

    def prep_criteria(self, param, lookup_nucleus):
        """
        Prepare a container with the parameter

        :param param:
        :return:
        """
        container = None

        prep_ctn = True

        # is param part of the nuclei?
        if lookup_nucleus is True:
            if self.segmentation.is_param_in_nuclei(param) is False:
                prep_ctn = False

        if prep_ctn is True:
            # prepare dictionary
            self.prep_criteria_dict(param)

            # prepare container
            container = self.prep_criteria_ctn(param)

        return container

    def prep_criteria_dict(self, param):
        """
        Prepare dictionary for param

        :param param:
        :return:
        """
        self.criteria[param] = dict()

        # edit boxes
        self.criteria[param]['edt_min'] = QtGui.QLineEdit()
        self.criteria[param]['edt_max'] = QtGui.QLineEdit()

        # figures
        self.criteria[param]['fig_hist'] = plt.figure(figsize=(5, 2))
        self.criteria[param]['fig_example'] = plt.figure(figsize=(5, 2))

        # canvases
        self.criteria[param]['cnv_hist'] = FigureCanvas(self.criteria[param]['fig_hist'])
        self.criteria[param]['cnv_example'] = FigureCanvas(self.criteria[param]['fig_example'])

        # misc widgets
        self.criteria[param]['sld_hist'] = QtGui.QSlider(QtCore.Qt.Horizontal, self)

    def prep_criteria_ctn(self, param):
        """
        Container to set criteria param

        :return:
        """
        container = QtGui.QGridLayout()

        # labels
        container.addWidget(QtGui.QLabel(getattr(gui_labels, 'crit_sel_label_' + param)), 0, 0)
        container.addWidget(QtGui.QLabel(gui_labels.crit_sel_min), 1, 0)
        container.addWidget(QtGui.QLabel(gui_labels.crit_sel_max), 2, 0)
        container.setRowStretch(3, 1)

        # edit boxes
        container.addWidget(self.criteria[param]['edt_min'], 1, 1)
        container.addWidget(self.criteria[param]['edt_max'], 2, 1)

        # get sorted labels
        self.criteria[param]['sorted_labels'] = self.segmentation.get_sorted_prop_list(param, self.is_nuclei_selection)
        self.criteria[param]['sorted_props'] = self.segmentation.sorted_probs.copy()

        # show histogram
        Plot.view_histogram_of_value_list(self.criteria[param]['fig_hist'],
                                          self.criteria[param]['sorted_labels'],
                                          cfg.criteria_select_hist_bins)

        # add to container
        container.addWidget(self.criteria[param]['cnv_hist'], 0, 2, 3, 1)
        container.addWidget(self.criteria[param]['cnv_example'], 0, 3, 4, 1)

        # slider
        self.criteria[param]['sld_hist'].valueChanged[int].connect(self.make_change_criteria_example(param))
        self.criteria[param]['sld_hist'].setMinimum(0)
        self.criteria[param]['sld_hist'].setMaximum(len(self.criteria[param]['sorted_props']))
        container.addWidget(self.criteria[param]['sld_hist'], 3, 2)

        return container

    def make_change_criteria_example(self, param):
        """
        Make function to call change criteria

        :param param:
        :return:
        """
        def change_criteria_example_param(example_selected):
            self.change_criteria_example(param, example_selected)

        return change_criteria_example_param

    def change_criteria_example(self, param, example_selected):
        """
        Change example in +/- range

        :param param:
        :param example_selected:
        :return:
        """
        min_range = example_selected - cfg.criteria_select_eg_range

        if min_range < 0:
            min_range = 0

        max_range = example_selected + cfg.criteria_select_eg_range

        if max_range > len(self.criteria[param]['sorted_props']):
            max_range = len(self.criteria[param]['sorted_props'])

        # prepare examples
        examples = list()

        # show examples from each part
        for i in range(min_range, max_range):
            example_title = '%i\n%.2f' % (self.criteria[param]['sorted_props'][i]['nID'],
                                            self.criteria[param]['sorted_props'][i][param])

            # prepare image
            examples = Plot.prepare_output(examples,
                                           self.criteria[param]['sorted_props'][i]['img'],
                                           example_title,
                                           cfg.criteria_select_eg_colour)

        # clear figure
        if len(examples[0]) < (cfg.criteria_select_eg_range * 2):
            self.criteria[param]['fig_example'].clear()

        # show examples
        Plot.show_images(self.criteria[param]['fig_example'], examples, cols=cfg.criteria_select_eg_cols)

        # update canvas
        self.criteria[param]['cnv_example'].draw()
