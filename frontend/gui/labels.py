"""
Mapping for GUI labels
"""

import configparser
import os

import storage.config as cfg

LABELS_PATH = 'config/gui_labels.ini'

# read labels
labels = configparser.ConfigParser()
labels.read(LABELS_PATH)

# Nucleus editor
pouch_location = labels['nucleus_edit']['POUCH_LOCATION']
nuc_xy = labels['nucleus_edit']['NUC_XY']
nuc_yz = labels['nucleus_edit']['NUC_YZ']
nuc_xz = labels['nucleus_edit']['NUC_XZ']
nuc_id = labels['nucleus_edit']['NUC_ID']
nuc_select = labels['nucleus_edit']['NUC_SELECT']
nuc_z_start = labels['nucleus_edit']['NUC_Z_START']
nuc_z_stop = labels['nucleus_edit']['NUC_Z_STOP']
nuc_is_nucleus = labels['nucleus_edit']['NUC_IS_NUCLEUS']
nuc_z_offset = labels['nucleus_edit']['NUC_Z_OFFSET']
tb_draw_nuclei = labels['nucleus_edit']['TB_DRAW_NUCLEI']
tb_sel_non_nuclei = labels['nucleus_edit']['TB_SEL_NON_NUCLEI']
tb_sel_corr_nuclei = labels['nucleus_edit']['TB_SEL_CORR_NUCLEI']
tb_correct_nuclei = labels['nucleus_edit']['TB_CORR_NUCLEI']

rb_draw_nuclei_mode = labels['nucleus_edit']['RB_DRAW_NUCLEI_MODE']
rb_draw_nuclei_mode_sel = labels['nucleus_edit']['RB_DRAW_NUCLEI_MODE_SEL']
rb_draw_nuclei_mode_add = labels['nucleus_edit']['RB_DRAW_NUCLEI_MODE_ADD']
rb_draw_nuclei_mode_del = labels['nucleus_edit']['RB_DRAW_NUCLEI_MODE_DEL']

rb_draw_nuclei_tool = labels['nucleus_edit']['RB_DRAW_NUCLEI_TOOL']
rb_draw_nuclei_tool_point = labels['nucleus_edit']['RB_DRAW_NUCLEI_TOOL_POINT']
rb_draw_nuclei_tool_brush = labels['nucleus_edit']['RB_DRAW_NUCLEI_TOOL_BRUSH']

rb_draw_nuclei_layer = labels['nucleus_edit']['RB_DRAW_NUCLEI_LAYER']
rb_draw_nuclei_layer_nuclei = labels['nucleus_edit']['RB_DRAW_NUCLEI_LAYER_NUCLEI']
rb_draw_nuclei_layer_nonuc = labels['nucleus_edit']['RB_DRAW_NUCLEI_LAYER_NONUC']
rb_draw_nuclei_layer_fila = labels['nucleus_edit']['RB_DRAW_NUCLEI_LAYER_FILA']
rb_draw_nuclei_layer_filtered = labels['nucleus_edit']['RB_DRAW_NUCLEI_LAYER_FILTERED']
rb_draw_nuclei_layer_labels = labels['nucleus_edit']['RB_DRAW_NUCLEI_LAYER_LABELS']
rb_draw_nuclei_layer_added = labels['nucleus_edit']['RB_DRAW_NUCLEI_LAYER_ADDED']
rb_draw_nuclei_layer_overlap = labels['nucleus_edit']['RB_DRAW_NUCLEI_LAYER_OVERLAP']

# image choice
img_exp = labels['image_choice']['IMG_EXP']
img_num = labels['image_choice']['IMG_NUM']
img_rev = labels['image_choice']['IMG_REV']
img_date = labels['image_choice']['IMG_DATE']
img_file = labels['image_choice']['IMG_FILE']
img_chnl = labels['image_choice']['IMG_CHNL']
img_select = labels['image_choice']['IMG_SELECT']
img_crop = labels['image_choice']['IMG_CROP']
img_crop_x = labels['image_choice']['IMG_CROP_X']
img_crop_y = labels['image_choice']['IMG_CROP_Y']
img_crop_z = labels['image_choice']['IMG_CROP_Z']

btn_save_set = labels['image_choice']['BTN_SAVE_SET']
btn_image_process = labels['image_choice']['BTN_IMG_PROC']
btn_merge_criteria = labels['image_choice']['BTN_MERGE_CRIT']
btn_nuclei_criteria = labels['image_choice']['BTN_NUC_CRIT']
btn_nuclei_select = labels['image_choice']['BTN_NUC_SEL']
btn_create_revision = labels['image_choice']['BTN_CREATE_REV']
btn_add_rand_patches = labels['image_choice']['BTN_ADD_RAND_PATCHES']
btn_save_rand_patches = labels['image_choice']['BTN_SAVE_RAND_PATCHES']
btn_add_rand_nuclei = labels['image_choice']['BTN_ADD_RAND_NUCLEI']
btn_save_rand_nuclei = labels['image_choice']['BTN_SAVE_RAND_NUCLEI']

tb_image_regions_roi = labels['image_choice']['TB_IMG_REGIONS_ROI']
tb_image_regions_train = labels['image_choice']['TB_IMG_REGIONS_TRAIN']
tb_train_rand_patches = labels['image_choice']['TB_TRAIN_RAND_PATCHES']
tb_train_rand_nuclei = labels['image_choice']['TB_TRAIN_RAND_NUCLEI']

sel_load_exp = labels['image_choice']['SEL_LOAD_EXP']

# image processing
proc_img_range_start = labels['image_processing']['IMG_RANGE_START']
proc_img_range_stop = labels['image_processing']['IMG_RANGE_STOP']
proc_img_range_int = labels['image_processing']['IMG_RANGE_INT']
proc_img_zoom = labels['image_processing']['IMG_ZOOM']
proc_step_ori = labels['image_processing']['STEP_ORI']

# criteria select
crit_sel_label_area = labels['criteria_select']['LABEL_AREA']
crit_sel_label_circularity = labels['criteria_select']['LABEL_CIRC']
crit_sel_label_eccentricity = labels['criteria_select']['LABEL_ECC']
crit_sel_label_surface = labels['criteria_select']['LABEL_SURFACE']
crit_sel_label_lamin_donut_ratio = labels['criteria_select']['LABEL_LAM_DON_RAT']
crit_sel_label_donut_ratio = labels['criteria_select']['LABEL_DONUT_RATIO']
crit_sel_label_edge_dist = labels['criteria_select']['LABEL_EDGE_DIST']
crit_sel_label_volume = labels['criteria_select']['LABEL_VOLUME']
crit_sel_label_depth = labels['criteria_select']['LABEL_DEPTH']
crit_sel_label_membrane_int = labels['criteria_select']['LABEL_MEMBRANE_INT']
crit_sel_label_dapi_int = labels['criteria_select']['LABEL_DAPI_INT']
crit_sel_label_nuc_edge_dist = labels['criteria_select']['LABEL_NUC_EDGE_DIST']
crit_sel_label_nuc_proba = labels['criteria_select']['LABEL_NUC_PROBA']
crit_sel_label_area_topbot_ratio = labels['criteria_select']['LABEL_AREA_TOPBOT_RATIO']
crit_sel_label_area_depth_ratio = labels['criteria_select']['LABEL_AREA_DEPTH_RATIO']
crit_sel_label_int_midbox = labels['criteria_select']['LABEL_INT_MIDBOX']
crit_sel_max = labels['criteria_select']['MAX']
crit_sel_min = labels['criteria_select']['MIN']

# generic
selection_none = labels['generic']['NO_SELECTION']
btn_next = labels['generic']['BTN_NEXT']
btn_prev = labels['generic']['BTN_PREV']
btn_details = labels['generic']['BTN_DETAILS']
btn_select = labels['generic']['BTN_SELECT']
btn_apply = labels['generic']['BTN_APPLY']
btn_save = labels['generic']['BTN_SAVE']
btn_save_to_disk = labels['generic']['BTN_SAVE_TO_DISK']
btn_browse = labels['generic']['BTN_BROWSE']
btn_add_new = labels['generic']['BTN_ADD_NEW']
btn_process = labels['generic']['BTN_PROCESS']
btn_load = labels['generic']['BTN_LOAD']
btn_segment = labels['generic']['BTN_SEGMENT']
btn_close = labels['generic']['BTN_CLOSE']
btn_reset = labels['generic']['BTN_RESET']
btn_merge = labels['generic']['BTN_MERGE']
btn_filter = labels['generic']['BTN_FILTER']
btn_update_preview = labels['generic']['BTN_UPDATE_PREVIEW']
btn_up = labels['generic']['BTN_UP']
btn_down = labels['generic']['BTN_DOWN']
label_sort = labels['generic']['LABEL_SORT']
