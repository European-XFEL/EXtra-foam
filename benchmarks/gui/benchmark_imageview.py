"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from collections import deque

import numpy as np

from PyQt5.QtCore import QTimer

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets import ImageViewF

app = mkQApp()


class BenchmarkImageViewSpeed:
    def __init__(self):
        self._dt = deque(maxlen=60)

        self._timer = QTimer()
        self._timer.timeout.connect(self.update)

        self._data = np.random.normal(size=(50, 1024, 1280))
        self._prev_t = None
        self._count = 0

        self._view = ImageViewF()
        self._view.show()

    def start(self):
        self._prev_t = time.time()
        self._timer.start(0)

    def update(self):
        self._view.setImage(self._data[self._count % 10])
        self._count += 1

        now = time.time()
        self._dt.append(now - self._prev_t)
        self._prev_t = now
        fps = len(self._dt) / sum(self._dt)

        self._view.setTitle(f"{fps:.2f} fps")

        app.processEvents()  # force complete redraw for every plot


if __name__ == '__main__':
    bench = BenchmarkImageViewSpeed()
    bench.start()
    app.exec_()
