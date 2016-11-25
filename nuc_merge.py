"""
Merge initial and revision segmentation to a final segmentation
"""

# import classes
from storage.image import ImageHandler
from processing.segmentation import Segmentation

# load image infos
image_infos = ImageHandler.load_image_infos()

revs_to_merge = list()

# apply defined filters to images
for image_info in image_infos:
    # revision? then skip
    if Segmentation.is_revision_by_ID(image_info['ID']):
        revs_to_merge.append(Segmentation.extract_rev_from_ID(image_info['ID']))
    else:
        # initialise segmentation
        segmentation = Segmentation(image_info)

        # load results
        segmentation.load()

        # merge
        for rev in revs_to_merge:
            segmentation.merge_parent_with_rev(rev)

        # save merge
        segmentation.save_merge_segmentation()

        # reset revs to merge
        revs_to_merge = list()
