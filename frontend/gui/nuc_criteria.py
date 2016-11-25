"""
Choose appropriate criteria to filter potential nuclei and make the selection
process easier
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
from frontend.gui.criteria_selection import CriteriaSelection

class NucleiCriteria(CriteriaSelection):

    def __init__(self, image_info, parent=None):
        self.is_nuclei_selection = True

        super(NucleiCriteria, self).__init__(image_info, parent=parent)

        # TEST: which parameters help you to select nuclei?
        #self.segmentation.calc_nuclei_params()

        # buttons
        self.btn_filter = QtGui.QPushButton(gui_labels.btn_filter)

        # set main layout
        self.setLayout(self.prep_ctn_criteria_select(True))

        # load criteria if they are set
        self.load_criteria()

    def prep_ctn_submit(self):
        """
        Prepare container to submit criteria selection

        :return:
        """
        container = QtGui.QGridLayout()

        # add buttons
        container.setColumnStretch(0, 1)
        container.addWidget(self.btn_reset, 0, 2)
        container.addWidget(self.btn_load, 0, 3)
        container.addWidget(self.btn_save, 0, 4)
        container.addWidget(self.btn_filter, 0, 5)
        container.addWidget(self.btn_close, 0, 6)

        # add listener
        self.btn_filter.clicked.connect(self.filter_nuclei)

        return container

    def filter_nuclei(self):
        """
        Filter nuclei based on selection criteria

        :return:
        """
        # filter nuclei
        self.segmentation.segment(process=False, merge=False, filter=True)

        # save results
        self.segmentation.save()
