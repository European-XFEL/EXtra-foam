"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QSplitter

from extra_foam.algorithms import SimpleSequence
from extra_foam.gui.misc_widgets import FColor
from extra_foam.gui.plot_widgets import PlotWidgetF, TimedPlotWidgetF
from extra_foam.pipeline.exceptions import ProcessingError

from .special_analysis_base import (
    create_special, profiler, QThreadFoamClient, QThreadWorker,
    _BaseAnalysisCtrlWidgetS, _SpecialAnalysisBase
)


_PREDEFINED_VECTORS = [
    "", "XGM intensity", "Digitizer pulse integral", "ROI FOM"
]

_DIGITIZER_CHANNELS = [
    'A', 'B', 'C', 'D', 'ADC'
]


class VectorViewProcessor(QThreadWorker):
    """Vector view processor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._vector1 = ''
        self._vector2 = ''

        self._vector1_full = SimpleSequence(max_len=6000)
        self._vector2_full = SimpleSequence(max_len=6000)

    def onVector1Change(self, value: str):
        self._vector1 = value

    def onVector2Change(self, value: str):
        self._vector2 = value

    @profiler("Vector view processor")
    def process(self, data):
        """Override."""
        processed = data["processed"]

        vec1, vec2 = self._fetch_data(processed)

        if vec1 is not None and vec2 is not None:
            if len(vec1) != len(vec2):
                raise ProcessingError(f"Vectors have different lengths: "
                                      f"{len(vec1)} and {len(vec2)}!")

            if len(self._vector1_full) != len(self._vector2_full):
                self.reset()
            self._vector1_full.extend(vec1)
            self._vector2_full.extend(vec2)

        self.log.info(f"Train {processed.tid} processed")

        return {
            "vector1": vec1,
            "vector2": vec2,
            "vector1_full": self._vector1_full.data(),
            "vector2_full": self._vector2_full.data(),
        }

    def _fetch_data(self, processed):
        ret = []
        for name in [self._vector1, self._vector2]:
            vec = None
            if name == 'ROI FOM':
                vec = processed.pulse.roi.fom
                if vec is None:
                    raise ProcessingError(
                        "Pulse-resolved ROI FOM is not available!")

            elif name == 'XGM intensity':
                vec = processed.pulse.xgm.intensity
                if vec is None:
                    raise ProcessingError("XGM intensity is not available!")

            elif name == 'Digitizer pulse integral':
                digit = processed.pulse.digitizer
                vec = digit[digit.ch_normalizer].pulse_integral
                if vec is None:
                    raise ProcessingError(
                        "Digitizer pulse integral is not available!")

            ret.append(vec)
        return ret

    def reset(self):
        """Override."""
        self._vector1_full.reset()
        self._vector2_full.reset()


class VectorViewCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Vector view control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vector1_cb = QComboBox()
        for item in _PREDEFINED_VECTORS:
            if item:
                # vector1 cannot be empty
                self.vector1_cb.addItem(item)

        self.vector2_cb = QComboBox()
        for item in _PREDEFINED_VECTORS:
            self.vector2_cb.addItem(item)

        self._non_reconfigurable_widgets = [
            self.vector1_cb,
            self.vector2_cb,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        layout.addRow("Vector 1: ", self.vector1_cb)
        layout.addRow("Vector 2: ", self.vector2_cb)

    def initConnections(self):
        """Override."""
        pass


class VectorPlot(PlotWidgetF):
    """VectorPlot class.

    Visualize the vector.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self._plot1 = self.plotCurve(name="Vector1", pen=FColor.mkPen('b'))
        self._plot2 = self.plotCurve(name="Vector2",
                                     pen=FColor.mkPen('r'),
                                     y2=True)
        self.setLabel("bottom", "Pulse index")
        self.setLabel("left", "Vector1")
        self.setLabel("right", "Vector2")
        self.setTitle("Vector view")
        self.addLegend(offset=(-40, 20))

    def updateF(self, data):
        """Override."""
        vec1 = data['vector1']
        if vec1 is None:
            self._plot1.setData([], [])
        else:
            self._plot1.setData(np.arange(len(vec1)), vec1)

        vec2 = data['vector2']
        if vec2 is None:
            self._plot2.setData([], [])
        else:
            self._plot2.setData(np.arange(len(vec2)), vec2)


class InTrainVectorCorrelationPlot(PlotWidgetF):
    """InTrainVectorCorrelationPlot class.

    Visualize correlation between two vectors within a train.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self._plot = self.plotScatter()

        self.setLabel("bottom", "Vector1")
        self.setLabel("left", "Vector2")
        self.setTitle("Correlation (train)")

    def updateF(self, data):
        """Override."""
        vec1, vec2 = data['vector1'], data['vector2']
        if vec1 is None or vec2 is None:
            self._plot.setData([], [])
        else:
            self._plot.setData(vec1, vec2)


class VectorCorrelationPlot(TimedPlotWidgetF):
    """VectorCorrelationPlot class.

    Visualize correlation between two vectors in history.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self._plot = self.plotScatter(brush=FColor.mkBrush('g', alpha=120))

        self.setLabel("bottom", "Vector1")
        self.setLabel("left", "Vector2")
        self.setTitle("Correlation")

    def refresh(self):
        """Override."""
        vec1, vec2 = self._data['vector1_full'], self._data['vector2_full']
        if vec1 is None or vec2 is None:
            self._plot.setData([], [])
        else:
            self._plot.setData(vec1, vec2)


@create_special(VectorViewCtrlWidget, VectorViewProcessor, QThreadFoamClient)
class VectorViewWindow(_SpecialAnalysisBase):
    """Main GUI for vector view."""

    icon = "vector_view.png"
    _title = "Vector view"
    _long_title = "Vector view"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False, with_levels=False)

        self._vector = VectorPlot(parent=self)
        self._corr_in_train = InTrainVectorCorrelationPlot(parent=self)
        self._corr = VectorCorrelationPlot(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        corr_panel = QSplitter(Qt.Horizontal)
        corr_panel.addWidget(self._corr_in_train)
        corr_panel.addWidget(self._corr)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._vector)
        right_panel.addWidget(corr_panel)

        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.vector1_cb.currentTextChanged.connect(
            self._worker_st.onVector1Change)
        self._ctrl_widget_st.vector1_cb.currentTextChanged.emit(
            self._ctrl_widget_st.vector1_cb.currentText())

        self._ctrl_widget_st.vector2_cb.currentTextChanged.connect(
            self._worker_st.onVector2Change)
        self._ctrl_widget_st.vector2_cb.currentTextChanged.emit(
            self._ctrl_widget_st.vector2_cb.currentText())
