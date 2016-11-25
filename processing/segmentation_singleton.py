"""
Segmentation algorithms. This includes the merging and correction of nuclei.
"""

from processing.segmentation import Segmentation

__seg_instance = None

def get_instance():
    """
    Simulate singleton-like behaviour to enable access to the segmentation system wide

    :return:
    """
    global __seg_instance

    if __seg_instance is None:
        __seg_instance = Segmentation()

    return __seg_instance
