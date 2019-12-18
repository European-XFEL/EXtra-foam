from .miscellaneous import normalize_auc
from .sampling import down_sample, quick_min_max, slice_curve, up_sample
from .data_structures import Stack

from .helpers import intersection

from .image_proc import (
    nanmeanImageArray,
    movingAverageImage, movingAverageImageArray,
    nanToZeroImage, nanToZeroImageArray,
    maskImage, maskImageArray,
    subDarkImage, subDarkImageArray
)

from .data_model import (
    RawImageDataFloat, RawImageDataDouble,
    MovingAverageArrayFloat, MovingAverageArrayDouble,
    MovingAverageFloat, MovingAverageDouble
)


def mask_image(image, *,
               image_mask=None,
               threshold_mask=None):
    """Mask image by image mask and/or threshold mask.

    The masked pixel value will will be set to 0.

    :param numpy.ndarray image: image to be masked. Shape = (y, x)
    :param numpy.ndarray/None image_mask: image mask. It assumes the shape
        of the image_mask is the same as the image.
    :param tuple/None threshold_mask: (min, max) of the threshold mask.
    """
    if image_mask is None and threshold_mask is None:
        nanToZeroImage(image)
    else:
        if image_mask is None:
            maskImage(image, *threshold_mask)
        elif threshold_mask is None:
            maskImage(image, image_mask)
        else:
            maskImage(image, image_mask, *threshold_mask)


def mask_image_array(images, *,
                     threshold_mask=None,
                     image_mask=None):
    """Mask an array of images by image mask and/or threshold mask.

    The masked pixel value will will be set to 0.

    :param numpy.ndarray image: image array to be masked.
        Shape = (index, y, x)
    :param numpy.ndarray/None image_mask: image mask. It assumes the shape
        of the image_mask is the same as the image.
    :param tuple/None threshold_mask: (min, max) of the threshold mask.
    """
    if image_mask is None and threshold_mask is None:
        nanToZeroImageArray(images)
    else:
        if image_mask is None:
            maskImageArray(images, *threshold_mask)
        elif threshold_mask is None:
            maskImageArray(images, image_mask)
        else:
            maskImageArray(images, image_mask, *threshold_mask)

