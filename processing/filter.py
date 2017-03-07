"""
Image filters based on individual classes
"""

import numpy as np
import numbers as nmb
import scipy.ndimage as ndi

from skimage.morphology import disk, skeletonize,\
    binary_closing, binary_opening, binary_dilation, binary_erosion,\
    dilation, closing, opening, erosion, label
from skimage import feature
from skimage import segmentation
from skimage.filters import rank, threshold_otsu

class Filter(object):
    """
    Image processing filters
    """

    def apply(self, img, param=None):
        """
        Apply filter methodology

        :param ori_img:
        :param param:
        :return:
        """
        raise NotImplementedError('Implement apply for filter')


class Dummy(Filter):
    """
    Return original image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return img


class Equalise(Filter):
    """
    Equalise signal by substracting the minimum from the image
    and then adjusting the values using the maximum
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        mod_img = img - img.min()
        mod_img = (mod_img/mod_img.max())*255

        return mod_img


class Homogenise(Filter):
    """
    Homogenise the image histogram
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return rank.equalize(img, selem=disk(param['size']))


class Median(Filter):
    """
    Median of the image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return rank.median(img, disk(param['size']))


class Gaussian(Filter):
    """
    Gaussian of the image

    (i) 3D ready
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return ndi.filters.gaussian_filter(img, (param['size']/100))


class Enhance(Filter):
    """
    Enhance contrast of the image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return rank.enhance_contrast(img, disk(param['size']))


class Threshold(Filter):
    """
    Use different thresholds for the image

    (i) 3D ready
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        thresholded_frame = img

        if img.max() > 0:
            if(param['method'] == 'OTSU'):
                threshold = threshold_otsu(img)

            thresholded_frame = img > (threshold * (param['size']/100))

        return thresholded_frame


class Skeleton(Filter):
    """
    Make a skeleton of the image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        frame = skeletonize(img)

        return frame


class Closing(Filter):
    """
    Close image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        if param['bin'] is not None and param['bin'] == 'y':
            result = binary_closing(img, selem=disk(param['size']))
        else:
            result = closing(img, selem=disk(param['size']))

        return result


class Opening(Filter):
    """
    Open image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        if param['bin'] is not None and param['bin'] == 'y':
            result = binary_opening(img, selem=disk(param['size']))
        else:
            result = opening(img, selem=disk(param['size']))

        return result


class Erosion(Filter):
    """
    Erode image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """

        if param['bin'] is not None and param['bin'] == 'y':
            result = binary_erosion(img, selem=disk(param['size']))
        else:
            result = erosion(img, selem=disk(param['size']))

        return result


class Dilation(Filter):
    """
    Dilate image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        if param['bin'] is not None and param['bin'] == 'y':
            result = binary_dilation(img, selem=disk(param['size']))
        else:
            result = dilation(img, selem=disk(param['size']))

        return result


class DistanceTransform(Filter):
    """
    Calculate distance transform of image

    (i) 3D ready
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        result = ndi.distance_transform_edt(img)

        return result


class LocalMax(Filter):
    """
    Calculate local maximum of image

    (i) 3D ready
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        if len(img.shape) > 2:
            result = feature.peak_local_max(img, indices=False,
                                            footprint=np.ones((param['size'], param['size'], param['size'])))
        else:
            result = feature.peak_local_max(img, indices=False,
                                            footprint=np.ones((param['size'], param['size'])))

        return result


class Invert(Filter):
    """
    Invert image

    (i) 3D ready
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return np.invert(img)


class Label(Filter):
    """
    Label image

    (i) 3D ready
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        result = label(img, connectivity=param['size'])

        return result


class BorderCorrection(Filter):
    """
    Set borders of image to 0
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        # remove border at edges of the frame to correct for watershed artifacts growing on the side
        border_corrected = img.copy()

        if param['size'] > 0:
            border_corrected[0:param['size'], :] = 0
            border_corrected[-param['size']:-1, :] = 0
            border_corrected[:, 0:param['size']] = 0
            border_corrected[:, -param['size']:-1] = 0
        else:
            border_corrected[0, :] = 0
            border_corrected[-1, :] = 0
            border_corrected[:, 0] = 0
            border_corrected[:, -1] = 0

        return border_corrected


class ColourCorrection(Filter):
    """
    Adjust colour of image

    (i) 3D ready
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        # adjust the colour range to 2**8
        corrected_frame = (img * 2**8).astype(int)

        return corrected_frame


class ConvertBit(Filter):
    """
    Convert image to a certain bit size

    (i) 3D ready
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        conv_img = img

        if param['size'] == 16:
            conv_img = img.astype('uint16')

        return conv_img


class Maxima(Filter):
    """
    Get maxima of image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return ndi.maximum_filter(img, size=param['size'], mode='constant')


class DetectBlobs(Filter):
    """
    Detect blobs in an image
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return feature.blob_log(img, min_sigma=param['size'])

class FillHoles(Filter):
    """
    Fill holes in binary images
    """

    def apply(self, img, param=None):
        """
        see Filter.apply()
        """
        return ndi.binary_fill_holes(img)
