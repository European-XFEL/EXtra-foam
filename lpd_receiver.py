from collections import namedtuple
from math import ceil
import numpy as np


stream_tuple_config = ("pulse image train module")
ModulePulseData = namedtuple("ModulePulseData", stream_tuple_config)


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


class LPDReceiver:
    """Given an LPDConfiguration and a KaraboBridge client,
    this will get the next data by pulse, and assemble it.
    Use so:

        config = LPDConfiguration(...)
        bridge = KaraboBridge("tcp://localhost:4545")
        receiver = LPDReceiver(config, bridge)
        getter = receiver.get_pulse_data()
        for train_data in getter:
            do_stuff(train_data)
    """

    def __init__(self, config, bridge):
        self.config = config
        self.bridge = bridge

    def get_pulse_data(self):
        """Get the data for analysis
           This function is a generator use as follows:

                receiver = ImageBuilder.get_pulse_data()
                data = next(receiver)
                type(data) == tuple

            You will then receive the data seamlessly for each pulse
            individually, regardless of their train.
            If you do care about trains, iterate and check the
            ModulePulseData.train
        """
        while True:
            stream_in = self.bridge.next()
            data = stream_in.pop("FXE_DET_LPD1M-1/DET/combined")['image.data']

            # Get the trainId from any other source in the data
            tid = stream_in.popitem()[1]['detector.trainId']

            for pulse_id in range(data.shape[0]):
                pulse_data = []
                for module_id in range(data.shape[1]):
                    pulse_data.append(ModulePulseData(
                                      pulse=pulse_id,
                                      image=data[pulse_id][module_id],
                                      module=module_id,
                                      train=tid)
                                      )
                yield tuple(pulse_data)

    def stitch_image(self, pulse_data):
        """Stitch module data together into a single coherant image
           :param pulse_data: an iterable of ModulePulseData
           :return cimgs: the assembled image offset with
                          known parameters (hole_pixel_size, q_offset)
           :return full_img: the assembled image, without any offset
        """
        # Combined images without offset
        full_img = np.zeros([self.config.SM * 4, self.config.SM * 4],
                            dtype='int16')
        # Combined images with offset
        cimgs = np.zeros([self.config.SM * 4 + self.config.hole_psize+self.config.q_offset,
                          self.config.SM * 4 + self.config.hole_psize+self.config.q_offset],
                         dtype='int16')

        for module_data in pulse_data:
            idx = module_data.module
            image = module_data.image

            rot = np.rot90(image, 2)
            full_img[self.config.dy_map[idx]:self.config.dy_map[idx] + self.config.SM,
                     self.config.SM * 4 - self.config.dx_map[idx]-self.config.SM:self.config.SM * 4 - self.config.dx_map[idx]] = rot

        # Align module in the full image. With a known `hole_size`, ie. the gap
        # in the center of the image, we can adjust the position of each module
        # around it
        if self.config.hole_size > 0:
            # Q1:
            y_slice = slice(0, self.config.SM * 2)
            x_slice = slice(self.config.hole_psize, self.config.hole_psize + self.config.SM * 2)
            cimgs[y_slice, x_slice] = full_img[0:self.config.SM * 2, 0:self.config.SM * 2]

            # Q2:
            y_slice = slice(self.config.SM * 2 + self.config.hole_psize, self.config.SM * 4 + self.config.hole_psize)
            x_slice = slice(self.config.SM * 2, self.config.SM * 4)
            cimgs[y_slice, x_slice] = full_img[self.config.SM * 2:self.config.SM * 4, self.config.SM * 2:self.config.SM * 4]

            # Q3:
            y_slice = slice(self.config.hole_psize, self.config.SM * 2 + self.config.hole_psize)
            x_slice = slice(self.config.hole_psize + self.config.SM * 2, self.config.hole_psize + self.config.SM * 4)
            cimgs[y_slice, x_slice] = full_img[0:self.config.SM * 2, self.config.SM * 2:self.config.SM * 4]

            # Q4:
            cimgs[self.config.SM*2:self.config.SM*4, 0:self.config.SM*2] = full_img[self.config.SM*2:self.config.SM*4, 0:self.config.SM*2]
        else:
            # Q1:
            y_slice = slice(0, self.config.SM * 2)
            x_slice = slice(self.config.SM * 2 + self.config.q_offset, self.config.SM * 4 + self.config.q_offset)
            cimgs[y_slice, x_slice] = full_img[0:self.config.SM * 2, self.config.SM * 2:self.config.SM * 4]

            # Q2:
            y_slice = slice(self.config.SM*2+self.config.q_offset, self.config.SM*4+self.config.q_offset)
            x_slice = slice(self.config.SM*2+self.config.hole_psize+self.config.q_offset,
                            self.config.SM*4+self.config.hole_psize+self.config.q_offset)
            cimgs[y_slice, x_slice] = full_img[self.config.SM * 2:self.config.SM * 4, self.config.SM * 2:self.config.SM * 4]

            # Q3:
            y_slice = slice(self.config.SM * 2 + self.config.hole_psize + self.config.q_offset,
                            self.config.SM * 4 + self.config.hole_psize + self.config.q_offset)
            x_slice = slice(self.config.hole_psize, self.config.SM * 2 + self.config.hole_psize)
            cimgs[y_slice, x_slice] = full_img[self.config.SM * 2:self.config.SM * 4, 0:self.config.SM * 2]

            # Q4:
            y_slice = slice(self.config.hole_psize, self.config.hole_psize + self.config.SM * 2)
            x_slice = slice(0, self.config.SM * 2)
            cimgs[y_slice, x_slice] = full_img[0:self.config.SM*2, 0:self.config.SM*2]

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

        return cimgs, full_img
