"""
Using pandas to store nuclei in csv instead of pickling the whole list
"""
import numpy as np
import pandas as pd
import re

# import classes
from storage.image import ImageHandler
from processing.segmentation import Segmentation
from storage.struct import Struct

import storage.config as cfg

# open window to select nuclei criteria
infos = ImageHandler.load_image_infos()

# select a specific info
selected_info_ID = 'N1-19-9'
selected_info = None

for info in infos:
    if info['ID'] == selected_info_ID:
        selected_info = info

# process segmentation
seg = Segmentation(selected_info)
#seg.segment(process=True, merge=True, filter=False)
#seg.save()
seg.load()
