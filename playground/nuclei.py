"""
To work with the nuclei list
"""

import pickle
import numpy as np

from processing.filter import Dilation
from frontend.figures.plot import Plot

# load nuclei
with open('results/N1-19-1/merge/nuclei.dat', 'rb') as fin:
    nuclei = pickle.load(fin)

# take a nucleus
nucleus = nuclei[0]

# add nucleus to image
img = np.zeros((100, 100))
coords = nucleus['coords'][0]

for coord in coords[1]:
    img[coord[0], coord[1]] = 1

# dilate
dilate = Dilation()
dilated = dilate.apply(img, {'size': 1, 'bin': None})

# show comparison
Plot.view_images((img, dilated))

# update coordinates
new_coords = np.argwhere(dilated)

coords = (coords[0], new_coords)
coords
