from math import ceil
import numpy as np


class LPDConfiguration:
    """This class contains all informations about the LPD
    configuration as it was that run.
    """

    # SM is the size in pixels of a supermodule, that's a group of four modules
    SM = 256

    # These are the offset of each module. they are referred to by their index.
    # Module 0, then, looks at index 0 in both maps to figure out where to be
    # placed in the assembled image.
    dx_map = [0, 0, SM, SM, 0, 0, SM, SM, SM*2, SM*2,
              SM*3, SM*3, SM*2, SM*2, SM*3, SM*3]
    dy_map = [0, SM, SM, 0, SM*2, SM*3, SM*3, SM*2,
              SM*2, SM*3, SM*3, SM*2, 0, SM, SM, 0]

    # The size of the pixels in the image in millimeter
    pixel_size = 0.5e-3

    expected_parameters = {
        "hole_size": "the size of the center detector hole, in mm"
                     ", used to be a negative value",
        "q_offset": "I don't know what that is, but it's 3",
    }

    def __init__(self, **kwargs):
        msg = "The expected parameters are {}".format(self.expected_parameters)
        correct_user_input = all([k in kwargs.keys() for k in
                                  self.expected_parameters.keys()])
        assert correct_user_input, msg

        [setattr(self, k, v) for k, v in kwargs.items()]
        self.hole_psize = abs(ceil(self.hole_size / self.pixel_size))


def stitch_image(config, image_data):
    """Combined images without offsets
    :param LPDConfiguration config: the detector's properties
    :param np.array image_data: stacked image data, as provided
            by `euxfel_h5tools.stack_detector_data`
    :return full_imgs: an np.array with all images assembled for a train
    """
    images = np.zeros([image_data.shape[0], config.SM * 4, config.SM * 4],
                      dtype='int16')

    for pulse_id in range(image_data.shape[0]):
        for module_id in range(image_data.shape[1]):
            idx = module_id
            image = image_data[pulse_id][module_id]

            rot = np.rot90(image, 2)
            images[pulse_id, config.dy_map[idx]:config.dy_map[idx] + config.SM,
                   config.SM * 4 - config.dx_map[idx] -
                   config.SM:config.SM * 4 - config.dx_map[idx]] = rot

    return images


def offset_image(config, full_img):
    """Offsets the combined images
    :param LPDConfiguration config: the detector's properties
    :param np.array image_data:  a single pulse's assembled image, as provided
            by `stitch_image`
    :return full_imgs: an np.array with the image offset for a bunch
    """
    cimgs = np.zeros([config.SM * 4 + config.hole_psize+config.q_offset,
                      config.SM * 4 + config.hole_psize+config.q_offset],
                     dtype='int16')

    # Align module in the full image. With a known `hole_size`, ie. the gap
    # in the center of the image, we can adjust the position of each module
    # around it
    if config.hole_size > 0:
        # Q1:
        y_slice = slice(0, config.SM * 2)
        x_slice = slice(config.hole_psize, config.hole_psize + config.SM * 2)
        cimgs[y_slice, x_slice] = full_img[0:config.SM * 2, 0:config.SM * 2]

        # Q2:
        y_slice = slice(config.SM * 2 + config.hole_psize, config.SM * 4 + config.hole_psize)
        x_slice = slice(config.SM * 2, config.SM * 4)
        cimgs[y_slice, x_slice] = full_img[config.SM * 2:config.SM * 4, config.SM * 2:config.SM * 4]

        # Q3:
        y_slice = slice(config.hole_psize, config.SM * 2 + config.hole_psize)
        x_slice = slice(config.hole_psize + config.SM * 2, config.hole_psize + config.SM * 4)
        cimgs[y_slice, x_slice] = full_img[0:config.SM * 2, config.SM * 2:config.SM * 4]

        # Q4:
        cimgs[config.SM*2:config.SM*4, 0:config.SM*2] = full_img[config.SM*2:config.SM*4, 0:config.SM*2]
    else:
        # Q1:
        y_slice = slice(0, config.SM * 2)
        x_slice = slice(config.SM * 2 + config.q_offset, config.SM * 4 + config.q_offset)
        cimgs[y_slice, x_slice] = full_img[0:config.SM * 2, config.SM * 2:config.SM * 4]

        # Q2:
        y_slice = slice(config.SM*2+config.q_offset, config.SM*4+config.q_offset)
        x_slice = slice(config.SM*2+config.hole_psize+config.q_offset,
                        config.SM*4+config.hole_psize+config.q_offset)
        cimgs[y_slice, x_slice] = full_img[config.SM * 2:config.SM * 4, config.SM * 2:config.SM * 4]

        # Q3:
        y_slice = slice(config.SM * 2 + config.hole_psize + config.q_offset,
                        config.SM * 4 + config.hole_psize + config.q_offset)
        x_slice = slice(config.hole_psize, config.SM * 2 + config.hole_psize)
        cimgs[y_slice, x_slice] = full_img[config.SM * 2:config.SM * 4, 0:config.SM * 2]

        # Q4:
        y_slice = slice(config.hole_psize, config.hole_psize + config.SM * 2)
        x_slice = slice(0, config.SM * 2)
        cimgs[y_slice, x_slice] = full_img[0:config.SM*2, 0:config.SM*2]

    # This removes border pixels, to eliminate false positives
    cimgs[116:148, 319:335] = 0
    cimgs[116:148, 431:457] = 0
    cimgs[116:148, 463:479] = 0
    cimgs[244:276, 383:512] = 0
    cimgs[212:244, 303:319] = 0
    cimgs[212:244, 448:464] = 0
    cimgs[706:738, 855:871] = 0
    cimgs[547:579, 662:695] = 0
    cimgs[951:983, 324:340] = 0

    return cimgs
