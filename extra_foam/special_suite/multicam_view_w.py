"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import functools

from PyQt5.QtWidgets import QGridLayout, QWidget

from .multicam_view_proc import MultiCamViewProcessor
from .special_analysis_base import (
    create_special, QThreadKbClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)
from ..gui.plot_widgets import ImageViewF
from ..gui.ctrl_widgets.smart_widgets import SmartLineEdit


_N_CAMERAS = 4


class MultiCamViewCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Multi-Camera view control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_channels = []
        self.properties = []
        for i in range(_N_CAMERAS):
            # TODO: remove the default value in the future
            # Here the output channel is allowed to be empty.
            self.output_channels.append(
                SmartLineEdit("FXE_OGT1_BIU/CAM/CAMERA:daqOutput"))
            self.properties.append(SmartLineEdit("data.image.pixels"))

        self._non_reconfigurable_widgets = [
            *self.output_channels,
            *self.properties,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()

        rows = []
        for i, (ch, ppt) in enumerate(zip(self.output_channels, self.properties)):
            rows.append((f"Output channel {i+1}", ch))
            rows.append((f"Property {i+1}", ppt))

        self.addRows(layout, rows)

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass


class CameraView(ImageViewF):
    """CameraView class.

    Visualize a single camera image.
    """
    def __init__(self, index, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=False, hide_axis=False, parent=parent)

        self._index = index

    def updateF(self, data):
        """Override."""
        self.setImage(data["images"][self._index])
        self.setTitle(data["channels"][self._index])


@create_special(MultiCamViewCtrlWidget, MultiCamViewProcessor, QThreadKbClient)
class MultiCamViewWindow(_SpecialAnalysisBase):
    """Main GUI for multi-camera view."""

    icon = "multi_cam_view.png"
    _title = "Multi-camera view"
    _long_title = "Multi-camera view"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False)

        self._views = [
            CameraView(i, parent=self) for i in range(_N_CAMERAS)
        ]

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QWidget()
        right_layout = QGridLayout()
        for i, view in enumerate(self._views):
            right_layout.addWidget(view, i // 2, i % 2)
        right_panel.setLayout(right_layout)

        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        for i, (ch, ppt) in enumerate(zip(self._ctrl_widget_st.output_channels,
                                          self._ctrl_widget_st.properties)):
            ch.value_changed_sgn.connect(
                functools.partial(self._worker_st.onOutputChannelChanged, i))
            ch.returnPressed.emit()

            ppt.value_changed_sgn.connect(
                functools.partial(self._worker_st.onPropertyChanged, i))
            ppt.returnPressed.emit()
