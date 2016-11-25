"""
Nuclei segmentation and quantification
"""

# import classes
from storage.image import ImageHandler
from processing.segmentation import Segmentation

# load image infos
image_infos = ImageHandler.load_image_infos()

# to skip parent for revisions
id_to_skip = None

# apply defined filters to images
for image_info in image_infos:
    # revision? then skip the initial segmentation and do the revisions
    if Segmentation.is_revision_by_ID(image_info['ID']):
        id_to_skip = Segmentation.extract_parent_from_ID(image_info['ID'])

        # create a new segmentation
        non_nuclei_seg = Segmentation(image_info, True)

        # segment stack
        non_nuclei_seg.segment()

        # save results
        non_nuclei_seg.save()
    else:
        # initial segmentation
        if id_to_skip != image_info['ID']:
            # initialise segmentation
            segmentation = Segmentation(image_info)

            # segment stack
            segmentation.segment()

            # save results
            segmentation.save()
