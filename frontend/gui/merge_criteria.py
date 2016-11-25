"""
Choose appropriate criteria to filter blobs and then merge them to
potential nuclei
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

class MergeCriteria(CriteriaSelection):

    def __init__(self, image_info, parent=None):
        super(MergeCriteria, self).__init__(image_info, parent=parent)

        # buttons
        self.btn_merge = QtGui.QPushButton(gui_labels.btn_merge)

        # set main layout
        self.setLayout(self.prep_ctn_criteria_select(False))

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
        container.addWidget(self.btn_merge, 0, 5)
        container.addWidget(self.btn_close, 0, 6)

        # add listener
        self.btn_merge.clicked.connect(self.merge_labels)

        return container

    def merge_labels(self):
        """
        Merge labels based on criteria selection

        :return:
        """
        # merge labels
        self.segmentation.segment(process=False, merge=True, filter=False)

        # save results
        self.segmentation.update(force_nuclei_raw=True)
