"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor


class Broker(_BaseProcessor):
    """Broker class."""
    def __init__(self):
        super().__init__()

    def update(self):
        """Override."""
        pass

    def process(self, data):
        """Override."""
        pass