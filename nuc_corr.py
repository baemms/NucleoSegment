"""
Run through the created segmentation and look at the nuclei.
Correct nuclei signals that do not seem to be correct.

This will be done half automatically and the ambition is to create a game-like
interface to compare and correct the nuclei.

Maybe show a sequence of different random nuclei and try to select the odd ones out.
"""

import sys

import matplotlib
# set Qt4 for matplot
matplotlib.use('Qt4Agg')

# Qt libraries
from PyQt4 import QtGui

# import classes
from storage.image import ImageHandler
from processing.segmentation import Segmentation
from frontend.gui.nuc_select import NucleoSelect

# load image infos
image_infos = ImageHandler.load_image_infos()

# to skip parent for revisions
id_to_skip = None

# apply defined filters to images
for image_info in image_infos:
    # revision? then skip the initial segmentation and do the revisions
    if Segmentation.is_revision_by_ID(image_info['ID']):
        id_to_skip = Segmentation.extract_parent_from_ID(image_info['ID'])

    if id_to_skip != image_info['ID']:
        # initialise segmentation
        segmentation = Segmentation(image_info)

        # load results
        segmentation.load()

        # cycle through the nuclei and correct their start and stop planes if necessary
        app = QtGui.QApplication(sys.argv)

        nuc_selector = NucleoSelect(segmentation)
        nuc_selector.show()
        nuc_selector.raise_()
        nuc_selector.activateWindow()

        sys.exit(app.exec_())
