"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor
from ...config import config, KaraboType


class CtrlDataProcessor(_BaseProcessor):
    """Control data processor.

    Process the control data (update at maximum 10 Hz),
    e.g. motor positions, monochromator energies, etc. The control
    data of XGM are processed in XgmProcessor.
    """

    _user_defined_key = config["SOURCE_USER_DEFINED_CATEGORY"]

    def __init__(self):
        super().__init__()

    def update(self):
        pass

    def process(self, data):
        """Override."""
        raw = data['raw']
        catalog = data['catalog']

        # parse sources
        for ctg in ["Magnet", "Monochromator", "Motor",
                    self._user_defined_key]:
            srcs = catalog.from_category(ctg)

            # process control data
            for src in srcs:
                # Double-check that the source is control data, this is
                # particularly needed for the user-defined keys.
                if not catalog.get_type(src) == KaraboType.CONTROL_DATA:
                    continue

                v = raw[src]
                self.filter_train_by_vrange(
                    v, catalog.get_vrange(src), src)
