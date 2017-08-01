"""
Configuration for the main program as singleton pattern ie/ initiated only once
"""

import configparser
import os

CFG_PATH = os.path.join('config', 'nuc_seg.ini')

"""
CSV conventions
"""
CSV_DEL = ';'
CSV_QUOT = '|'

"""
Processing steps
"""
PROC_PARAMS = '='
PROC_PARAMS_DEL = ','
PROC_PARAMS_RANGE = '-'

# read config
config = configparser.ConfigParser()
config.read(CFG_PATH)

general_verbose = bool(int(config['General']['VERBOSE']))

quant_tiling_dim = int(config['Quantification']['TILING_DIM'])
quant_tiling_count = int(config['Quantification']['TILING_COUNT'])

# get path
path_raw = config['Paths']['RAW'] + os.sep

# Filter information
file_filter_mapping = os.path.join('config', config['Files']['FILTER_MAPPING'])
file_processing = os.path.join('config', config['Files']['PROCESSING'])

# map values to variables
file_input = os.path.join('config', config['Files']['INPUT'])

# criteria for nuclei
file_nuc_criteria = os.path.join('config', config['Files']['NUC_CRITERIA'])

file_nuclei_param_ranges = os.path.join('config', config['Files']['NUCLEI_PARAM_RANGES'])

"""
Segmentation parameter
"""
merge_min_overlap = int(config['NucleiMerge']['MIN_OVERLAP'])
merge_depth = int(config['NucleiMerge']['DEPTH'])
merge_threads_nuc_params = int(config['NucleiMerge']['THREADS_NUC_PARAMS'])
merge_threads_labels_props = int(config['NucleiMerge']['THREADS_LABELS_PROPS'])
merge_lamin_donut_ring = int(config['NucleiMerge']['LAMIN_DONUT_RING'])

file_nuclei_raw = config['Files']['NUCLEI_RAW']
file_lookup_raw = config['Files']['LOOKUP_RAW']
file_labels_props_raw = config['Files']['LABELS_PROPS_RAW']
file_stack_labels = config['Files']['STACK_LABELS']
file_stack_nuclei = config['Files']['STACK_NUCLEI']
file_stack_membin = config['Files']['STACK_MEMBIN']
file_stack_lamin = config['Files']['STACK_LAMIN']

file_nuclei_data_params = config['Files']['NUCLEI_DATA_PARAMS']
file_nuclei_data_int = config['Files']['NUCLEI_DATA_INT']
file_nuclei_data_centre = config['Files']['NUCLEI_DATA_CENTRE']
file_nuclei_data_centroid = config['Files']['NUCLEI_DATA_CENTROID']
file_nuclei_data_coords = config['Files']['NUCLEI_DATA_COORDS']
file_nuclei_data_area = config['Files']['NUCLEI_DATA_AREA']

file_nuclei_img_box = config['Files']['NUCLEI_IMG_BOX']
file_nuclei_img_cropped_box = config['Files']['NUCLEI_IMG_CROPPED_BOX']
file_nuclei_img_projection = config['Files']['NUCLEI_IMG_PROJECTION']
file_nuclei_img_show = config['Files']['NUCLEI_IMG_SHOW']

path_results = config['Paths']['RESULTS'] + os.sep
path_tmp = config['Paths']['TMP'] + os.sep
path_results_stacks_raw = config['Paths']['RESULTS_STACKS_RAW'] + os.sep
path_results_nucleus_params_raw = config['Paths']['RESULTS_NUC_PARAMS_RAW'] + os.sep
path_results_nuclei = config['Paths']['RESULTS_NUCLEI'] + os.sep
path_nuclei_data = config['Paths']['NUCLEI_DATA'] + os.sep

label_props_to_get_keys = config['Segmentation']['LABEL_PROPS_TO_GET_KEYS'].split(',')

"""
Selection for segmentation
"""
nuclei_segment_overlay_colour = config['NucleiSegment']['OVERLAY_COLOUR']
nuclei_segment_pouch_colour = config['NucleiSegment']['POUCH_COLOUR']
nuclei_segment_overlay_alpha = float(config['NucleiSegment']['OVERLAY_ALPHA'])
nuclei_segment_pouch_alpha = float(config['NucleiSegment']['POUCH_ALPHA'])
nuclei_segment_rand_patches = int(config['NucleiSegment']['RAND_PATCHES'])
nuclei_segment_rand_patches_size = int(config['NucleiSegment']['RAND_PATCHES_SIZE'])
nuclei_segment_rand_nuclei = int(config['NucleiSegment']['RAND_NUCLEI'])
nuclei_segment_rand_nuclei_raster = int(config['NucleiSegment']['RAND_NUCLEI_RASTER'])

"""
Nuclei select parameters
"""
nucleus_box_offset = int(config['NucleiSelect']['OFFSET'])
nucleus_planes_plane_size = int(config['NucleiSelect']['PLANE_SIZE'])
nucleus_planes_autoload = bool(int(config['NucleiSelect']['PLANE_AUTOLOAD']))
nucleus_select_example_range = int(config['NucleiSelect']['EXAMPLE_RANGE'])
nucleus_select_example_default = config['NucleiSelect']['EXAMPLE_DEFAULT']
nucleus_select_example_autoupdate = bool(int(config['NucleiSelect']['EXAMPLE_AUTOUPDATE']))
nucleus_select_example_update_every = int(config['NucleiSelect']['EXAMPLE_UPDATE_EVERY'])
nucleus_select_non_nuclei_zoom = int(config['NucleiSelect']['NON_NUCLEI_ZOOM'])
nucleus_select_non_nuclei_zoom_max = int(config['NucleiSelect']['NON_NUCLEI_ZOOM_MAX'])
nucleus_select_draw_nuclei_zoom = int(config['NucleiSelect']['DRAW_NUCLEI_ZOOM'])
nucleus_select_draw_nuclei_zoom_max = int(config['NucleiSelect']['DRAW_NUCLEI_ZOOM_MAX'])
nucleus_select_corr_nuclei_overlay_alpha = float(config['NucleiSelect']['CORR_NUCLEI_OVERLAY_ALPHA'])
nucleus_select_corr_nuclei_overlay_colour = config['NucleiSelect']['CORR_NUCLEI_OVERLAY_COLOUR']

"""
Procesing steps parameters
"""
image_processing_default_range_offset = int(config['ProcessingSteps']['DEFAULT_RANGE_OFFSET'])
image_processing_default_range_int = int(config['ProcessingSteps']['DEFAULT_RANGE_INT'])
image_processing_default_zoom = int(config['ProcessingSteps']['DEFAULT_ZOOM'])
image_processing_image_size = int(config['ProcessingSteps']['IMG_SIZE'])

nucleus_calc_elps_rot = int(config['NucleusCalc']['ELPS_ROT'])

"""
Plot parameters
"""
fontsdict_plot = dict()
fontsdict_plot['fontsize'] = config['Fontsdict']['PLOT_FONTSIZE']

"""
Corrections parameters
"""
path_corrections = config['Paths']['CORRECTIONS'] + os.sep
path_correction_stacks = os.path.join('corrections', config['Paths']['CORRECTION_STACKS']) + os.sep
path_results_stacks_corr = config['Paths']['RESULTS_STACKS_CORR'] + os.sep
path_results_nucleus_params_corr = config['Paths']['RESULTS_NUC_PARAMS_CORR'] + os.sep

file_corr_fila = config['Files']['CORR_FILA']
file_corr_nonuc = config['Files']['CORR_NONUC']
file_corr_filtered = config['Files']['CORR_FILTERED']
file_corr_added = config['Files']['CORR_ADDED']
file_corr_overlap = config['Files']['CORR_OVERLAP']
file_corr_remerge = config['Files']['CORR_REMERGE']
file_nuclei_corr = config['Files']['NUCLEI_CORR']
file_lookup_corr = config['Files']['LOOKUP_CORR']
file_labels_props_corr = config['Files']['LABELS_PROPS_CORR']
file_stack_non_nuclei = config['Files']['STACK_NON_NUCLEI']

"""
Merge parameters
"""
file_nuclei = config['Files']['NUCLEI']
file_labels_props = config['Files']['LABELS_PROPS']
file_stack_nuclam = config['Files']['STACK_NUCLAM']
path_merge = config['Paths']['MERGE'] + os.sep
path_merge_stacks = os.path.join('merge', config['Paths']['MERGE_STACKS']) + os.sep

merge_post_dil = int(config['PostProcessing']['DILATION'])
nuclei_bbox_range = int(config['PostProcessing']['NUCLEI_BBOX_RANGE'])

"""
Filter criteria parameters
"""
filter_criteria_labels = config['FilterCriteria']['LABELS'].split(',')
filter_criteria_nuclei = config['FilterCriteria']['NUCLEI'].split(',')

"""
Criteria selection parameters
"""
criteria_select_eg_num = int(config['CriteriaSelection']['EXAMPLE_NUMBER'])
criteria_select_eg_cols = int(config['CriteriaSelection']['EXAMPLE_COLS'])
criteria_select_eg_range = int(config['CriteriaSelection']['EXAMPLE_RANGE'])
criteria_select_eg_colour = config['CriteriaSelection']['EXAMPLE_COLOUR']
criteria_select_eg_lamin_colour = config['CriteriaSelection']['EXAMPLE_LAMIN_COLOUR']
criteria_select_hist_bins = int(config['CriteriaSelection']['HIST_BINS'])
criteria_select_nuc_box_offset = int(config['CriteriaSelection']['NUC_BOX_OFFSET'])
criteria_select_int_midbox_offset = int(config['CriteriaSelection']['INT_MIDBOX_OFFSET'])
criteria_select_lamin_donut_ring = int(config['CriteriaSelection']['LAMIN_DONUT_RING'])

"""
Classifier parameters
"""
clf_method = config['Classifier']['METHOD']
clf_estimators = int(config['Classifier']['ESTIMATORS'])
clf_train_params = config['Classifier']['TRAIN_PARAMS'].split(',')
clf_sig_threshold = float(config['Classifier']['SIG_THR'])
path_classifier = config['Paths']['CLF'] + os.sep
file_classifier = path_classifier + config['Files']['CLF']

"""
Pandas datastructures
"""
pd_struct_nuclei_cols = {
    # to be calculated once merged
    'data_params': ['volume', 'surface', 'depth', 'colour', 'rejected',
                    'lamin_int', 'dapi_int', 'membrane_int',
                    'nuc_edge_dist','area_topbot_ratio',
                    'area_depth_ratio', 'nuc_bbox', 'volume_depth_ratio',
                    'surface_volume_ratio', 'neighbours', 'neighbours_distance',
                    'nuc_centre', 'direction', 'apical_dist', 'nuclei_in_direction',
                    'minor_axis', 'major_axis', 'mami_axis',
                    'minor_axis_orientation', 'major_axis_orientation',
                    'direction_orientation', 'contact_surface', 'compactness',
                    'closeness', 'sphericity'],
    # 'data_centre': ['z', 'y', 'x'],
    # from individual labels
    'data_centroid': ['z', 'y', 'x'],
    'data_bbox': ['z', 'min_row', 'min_col', 'max_row', 'max_col'],
    'data_z_params': ['z', 'area', 'perimeter'],
    'data_coords': ['z', 'y', 'x', 'is_edge']
}

PD_STRUCT_TYPE_VAL = 0
PD_STRUCT_TYPE_TUPLE = 1
PD_STRUCT_TYPE_1D = 2
PD_STRUCT_TYPE_2D = 3
PD_STRUCT_TYPE_1D_VAL = 4

pd_struct_nuclei_col_types = {
    'data_params': PD_STRUCT_TYPE_VAL,
    'data_int': PD_STRUCT_TYPE_VAL,
    'data_centre': PD_STRUCT_TYPE_TUPLE,
    'data_centroid': PD_STRUCT_TYPE_1D,
    'data_coords': PD_STRUCT_TYPE_2D,
    'data_area': PD_STRUCT_TYPE_1D_VAL
}

"""
Param range selection
"""
param_range_params = config['ParamRange']['PARAMS'].split(',')
