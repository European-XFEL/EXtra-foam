"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

CorrelationProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import AbstractProcessor
from ..data_model import ProcessedData
from ..exceptions import ProcessingError
from ...algorithms import slice_curve
from ...config import FomName


class CorrelationProcessor(AbstractProcessor):
    """Add correlation information into processed data.

    Attributes:
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    def __init__(self):
        super().__init__()

        self.fom_name = None
        self.fom_itgt_range = None

    def process(self, proc_data, raw_data=None):
        """Override."""
        if self.fom_name is None:
            return

        elif self.fom_name == FomName.PUMP_PROBE_FOM:
            _, foms, _ = proc_data.pp.fom
            if foms.size == 0:
                raise ProcessingError("Pump-probe result is not available!")
            fom = foms[-1]

        elif self.fom_name == FomName.ROI1:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            if roi1_hist.size == 0:
                return
            fom = roi1_hist[-1]

        elif self.fom_name == FomName.ROI2:
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            if roi2_hist.size == 0:
                return
            fom = roi2_hist[-1]

        elif self.fom_name == FomName.ROI_SUM:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            if roi1_hist.size == 0:
                return
            fom = roi1_hist[-1] + roi2_hist[-1]

        elif self.fom_name == FomName.ROI_SUB:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            if roi1_hist.size == 0:
                return
            fom = roi1_hist[-1] - roi2_hist[-1]

        if self.fom_name == FomName.AI_MEAN:
            momentum = proc_data.ai.momentum
            if momentum is None:
                raise ProcessingError(
                    "Azimuthal integration result is not available!")
            intensity = proc_data.ai.intensity_mean

            # calculate figure-of-merit
            fom = slice_curve(intensity, momentum, *self.fom_itgt_range)[0]
            fom = np.sum(np.abs(fom))

        else:
            raise ProcessingError(f"Unknown FOM name: {self.fom_name}!")

        for param in ProcessedData.get_correlators():
            _, _, info = getattr(proc_data.correlation, param)
            if info['device_id'] == "Any":
                # orig_data cannot be empty here
                setattr(proc_data.correlation, param, (proc_data.tid, fom))
            else:
                try:
                    device_data = raw_data[info['device_id']]
                except KeyError:
                    raise ProcessingError(
                        f"Device '{info['device_id']}' is not in the data!")

                try:
                    if info['property'] in device_data:
                        ppt = info['property']
                    else:
                        # From the file
                        ppt = info['property'] + '.value'

                    setattr(proc_data.correlation, param,
                            (device_data[ppt], fom))

                except KeyError:
                    raise ProcessingError(
                        f"'{info['device_id']}'' does not have property "
                        f"'{info['property']}'")