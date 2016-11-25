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
selected_info_ID = 'N1-19-21'
selected_info = None

for info in infos:
    if info['ID'] == selected_info_ID:
        selected_info = info

# process segmentation
seg = Segmentation(selected_info)
#seg.segment(process=True, merge=True, filter=False)
#seg.save()
seg.load()

# prepare arrays for panda conversion
cols = {
    'data_params': ['volume', 'surface', 'depth'],
    'data_int': ['lamin_int', 'dapi_int', 'membrane_int'],
    'data_centre': ['z', 'y', 'x'],
    'data_centroid': ['z', 'y', 'x'],
    'data_coords': ['z', 'y', 'x'],
    'data_area': ['z', 'area'],
    #'img_box': ['pos_z', 'pos_y', 'pos_x',
    #            'lamin', 'membrane', 'dapi']
    #'img_cropped_box': ['pos_z', 'pos_y', 'pos_x',
    #                    'nucleus', 'lamin', 'membrane', 'dapi', 'rgb'],
    'img_projection': ['z', 'y', 'x']
}

indices = dict()
data = dict()
for col in cols.keys():
    indices[col] = list()
    data[col] = list()

# go through nuclei and prepare the individual datalists
for lID, nucleus in enumerate(seg.nuclei):
    # params
    indices['data_params'].append(nucleus['nID'])
    data['data_params'].append(list())

    for column in cols['data_params']:
        data['data_params'][-1].append(nucleus[column])

    # intensities
    indices['data_int'].append(nucleus['nID'])
    data['data_int'].append(list())

    for column in cols['data_int']:
        data['data_int'][-1].append(nucleus[column])

    # centre
    indices['data_centre'].append(nucleus['nID'])
    data['data_centre'].append(list())

    for pos in nucleus['centre']:
        data['data_centre'][-1].append(pos)

    # convert label props to arrays
    labels_props = seg.get_array_lists_for_nucleus_label_props(nucleus)

    # map into index and data lists and add indicies
    for prop in ['centroid', 'coords', 'area']:
        # mapping
        for data_row in labels_props[prop]:
            data['data_%s' % prop].append(data_row)
            indices['data_%s' % prop].append(nucleus['nID'])

    # get projections
    indices['img_projection'].append(nucleus['nID'])
    data['img_projection'].append(list())

    for img_projection in cols['img_projection']:
        data['img_projection'][-1].append(nucleus['projection_%s' % img_projection])

# create panda dataframes
save_dataframes = Struct()
for col in cols.keys():
    setattr(save_dataframes, col, pd.DataFrame(
        np.array(data[col]),
        index=np.array(indices[col]).astype(int),
        columns=np.array(cols[col])
    ))

data_path = seg.get_results_dir().nuclei_data
print('test d')
"""
# save dataframes as csv

for col in cols.keys():
    print('store %s' % col)
    path = data_path + getattr(cfg, 'file_nuclei_%s' % col)
    getattr(save_dataframes, col).to_csv(path, sep=cfg.CSV_DEL)
"""

# read dataframes
load_dataframes = Struct()
for col in cols.keys():
    print('load %s' % col)
    path = data_path + getattr(cfg, 'file_nuclei_%s' % col)
    setattr(load_dataframes, col, pd.read_csv(path, delimiter=cfg.CSV_DEL, index_col=0))

# assemble nuclei
nuclei = seg.build_nuclei_from_dataframe(load_dataframes, cols)

