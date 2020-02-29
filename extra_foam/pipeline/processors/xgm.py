"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray, MovingAverageScalar
from ..exceptions import UnknownParameterError
from ...utils import profiler
from ...database import Metadata as mt
from ...ipc import process_logger as logger


class XgmProcessor(_BaseProcessor):
    """XGM data processor.

    Process instrument and pipeline data from XGM.
    """
    _intensity_ma = MovingAverageScalar()
    _x_ma = MovingAverageScalar()
    _y_ma = MovingAverageScalar()
    _pulse_intensity_ma = MovingAverageArray()

    _intensity_ppt = "pulseEnergy.photonFlux"
    _x_pos_ppt = "beamPosition.ixPos"
    _y_pos_ppt = "beamPosition.iyPos"

    _intensity_channels = ["data.intensityTD",
                           "data.intensitySa1TD",
                           "data.intensitySa2TD",
                           "data.intensitySa3TD"]

    def __init__(self):
        super().__init__()

        self._ma_window = 1

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.GLOBAL_PROC)
        self._update_moving_average(cfg)

    def _update_moving_average(self, cfg):
        if 'reset_ma_xgm' in cfg:
            # reset moving average
            del self._intensity_ma
            del self._x_ma
            del self._y_ma
            del self._pulse_intensity_ma
            self._meta.hdel(mt.GLOBAL_PROC, 'reset_ma_xgm')

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self.__class__._intensity_ma.window = v
            self.__class__._x_ma.window = v
            self.__class__._y_ma.window = v
            self.__class__._pulse_intensity_ma.window = v

        self._ma_window = v

    @profiler("XGM Processor")
    def process(self, data):
        """Override."""
        processed = data['processed']
        raw = data['raw']
        catalog = data['catalog']

        # parse sources
        xgm_srcs = catalog.from_category('XGM')
        if not xgm_srcs:
            return

        control_srcs = []
        pipeline_srcs = []
        for src in xgm_srcs:
            if src.split(' ')[0].split(":")[-1] == "output":
                pipeline_srcs.append(src)
            else:
                control_srcs.append(src)

        # process control data
        for src in control_srcs:
            v = raw[src]
            ppt = src.split(' ')[1]

            if ppt == self._intensity_ppt:
                self._intensity_ma = v
                processed.xgm.intensity = self._intensity_ma
            elif ppt == self._x_pos_ppt:
                self._x_ma = v
                processed.xgm.x = self._x_ma
            elif ppt == self._y_pos_ppt:
                self._y_ma = v
                processed.xgm.y = self._y_ma
            else:
                raise UnknownParameterError(f'[XGM] Unknown property: {ppt}')

            self.filter_train_by_vrange(v, catalog.get_vrange(src), src)

        # process pipeline data
        pipeline_src_tracker = dict()
        for src in pipeline_srcs:
            arr = raw[src]
            device_id, ppt = src.split(' ')

            if ppt in self._intensity_channels:
                # check the duplication of the "intensity" keys
                # Note: I make the warning here in order not to complicate
                #       the implementation of the source tree. If I put
                #       the restriction on the tree, then I have to
                #       duplicate and hard-code the intensity list again!
                if device_id in pipeline_src_tracker:
                    prev_ppt = pipeline_src_tracker[device_id]
                    if ppt != prev_ppt:
                        logger.warning(f"Only one of '{ppt}' and '{prev_ppt}' "
                                       f"should be selected! "
                                       f"{prev_ppt} is ignored")
                pipeline_src_tracker[device_id] = ppt

                self._pulse_intensity_ma = np.array(
                    arr[catalog.get_slicer(src)], dtype=np.float32)
                processed.pulse.xgm.intensity = self._pulse_intensity_ma
            else:
                raise UnknownParameterError(f'[XGM] Unknown property: {ppt}')

            # apply pulse filter
            self.filter_pulse_by_vrange(self._pulse_intensity_ma,
                                        catalog.get_vrange(src),
                                        processed.pidx,
                                        src)
