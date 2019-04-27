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

from .base_processor import LeafProcessor
from ..data_model import ProcessedData
from ..exceptions import ProcessingError
from ...algorithms import slice_curve
from ...config import FomName
from ...helpers import profiler


class CorrelationProcessor(LeafProcessor):
    """Add correlation information into processed data.

    Attributes:
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fom_name = None
        self.fom_itgt_range = None

    @profiler("Correlation processor")
    def run(self, processed, raw=None):
        """Override."""
        if self.fom_name is None or self.fom_name == FomName.UNDEFINED:
            return

        if self.fom_name == FomName.PUMP_PROBE_FOM:
            _, foms, _ = processed.pp.fom
            if foms.size == 0:
                raise ProcessingError("Pump-probe result is not available!")
            fom = foms[-1]

        elif self.fom_name == FomName.ROI1:
            _, roi1_hist, _ = processed.roi.roi1_hist
            if roi1_hist.size == 0:
                return
            fom = roi1_hist[-1]

        elif self.fom_name == FomName.ROI2:
            _, roi2_hist, _ = processed.roi.roi2_hist
            if roi2_hist.size == 0:
                return
            fom = roi2_hist[-1]

        elif self.fom_name == FomName.ROI_SUM:
            _, roi1_hist, _ = processed.roi.roi1_hist
            _, roi2_hist, _ = processed.roi.roi2_hist
            if roi1_hist.size == 0:
                return
            fom = roi1_hist[-1] + roi2_hist[-1]

        elif self.fom_name == FomName.ROI_SUB:
            _, roi1_hist, _ = processed.roi.roi1_hist
            _, roi2_hist, _ = processed.roi.roi2_hist
            if roi1_hist.size == 0:
                return
            fom = roi1_hist[-1] - roi2_hist[-1]

        elif self.fom_name == FomName.AI_MEAN:
            momentum = processed.ai.momentum
            if momentum is None:
                raise ProcessingError(
                    "Azimuthal integration result is not available!")
            intensity = processed.ai.intensity_mean

            # calculate figure-of-merit
            fom = slice_curve(intensity, momentum, *self.fom_itgt_range)[0]
            fom = np.sum(np.abs(fom))

        else:
            name = str(self.fom_name).split(".")[-1]
            raise ProcessingError(f"Unknown FOM name: {name}!")

        for param in ProcessedData.get_correlators():
            _, _, info = getattr(processed.correlation, param)
            if info['device_id'] == "Any":
                # orig_data cannot be empty here
                setattr(processed.correlation, param, (processed.tid, fom))
            else:
                try:
                    device_data = raw[info['device_id']]
                except KeyError:
                    raise ProcessingError(
                        f"Device '{info['device_id']}' is not in the data!")

                try:
                    if info['property'] in device_data:
                        ppt = info['property']
                    else:
                        # From the file
                        ppt = info['property'] + '.value'

                    setattr(processed.correlation, param,
                            (device_data[ppt], fom))

                except KeyError:
                    raise ProcessingError(
                        f"'{info['device_id']}'' does not have property "
                        f"'{info['property']}'")