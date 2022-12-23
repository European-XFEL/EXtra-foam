"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from dataclasses import dataclass
from collections import defaultdict

import h5py
import numpy as np

from PyQt5.QtGui import QCursor, QIntValidator
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (QSplitter, QComboBox, QTabWidget, QStackedWidget,
                             QLabel, QHBoxLayout, QWidget, QStyle, QToolTip,
                             QTabBar, QToolButton, QPushButton, QApplication,
                             QSpacerItem, QFormLayout, QFileDialog, QMessageBox)

from ..gui.pyqtgraph import LinearRegionItem
from ..gui.gui_helpers import center_window
from ..gui.misc_widgets import SequentialColor, FColor
from ..gui.plot_widgets import ImageViewF, PlotWidgetF
from ..algorithms import SimpleVectorSequence
from ..gui.ctrl_widgets import SmartLineEdit

from . import logger
from .xes_timing_proc import XesTimingProcessor, DisplayOption
from .special_analysis_base import (
    create_special, ClientType, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)


class DetectorView(ImageViewF):
    """DetectorView class.

    Visualize the detector image.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(num_rois=2, parent=parent)

        self._auto_level = False
        self._title_prefix = "Detector avg. for current delay"

    def updateF(self, data):
        """Override."""
        if data is None:
            return

        img = data["img_avg"]
        if img is not None:
            self.setImage(data["img_avg"], auto_levels=self._auto_level)

            if self._auto_level:
                self._auto_level = False

            title = f"{self._title_prefix} ({data['delay']})"
            if self.title() != title:
                self.setTitle(title)

    def onImgDisplayChanged(self, _):
        self._auto_level = True


class XesSignalPlot(PlotWidgetF):
    """XesSignalPlot class.

    Visualize XES curves for different delays.
    """
    def __init__(self, roi_idx, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('left', "Difference")
        self.setLabel('bottom', "X axis (pixels)")
        self.setTitle("XES signal")

        self._roi_idx = roi_idx
        self._roi_x = 0
        self._plots = { }
        self._pens = list(SequentialColor.mkPen(len(SequentialColor.pool)))

        self._legend = self.addLegend(offset=(-10, -10))
        self.hideLegend()

    def updateF(self, data):
        """Override."""
        if data is None:
            return

        if not self._legend.isVisible():
            self.showLegend()

        # If necessary, redraw the plots for all the delays
        if data["refresh_plots"]:
            for delay, delay_data in data["xes"].items():
                self.plot_delay(delay, delay_data)
        else:
            # Otherwise just update the plot for the current delay
            current_delay = data["delay"]
            self.plot_delay(current_delay, data["xes"][current_delay])

    def plot_delay(self, delay, delay_data):
        diff = delay_data.difference[self._roi_idx]
        if diff is None:
            return

        if delay not in self._plots:
            self._plots[delay] = self.plotCurve(pen=self._pens[len(self._plots) % len(self._pens)])
            self._legend.addItem(self._plots[delay], f"Delay: {delay}")

        self._plots[delay].setData(self._roi_x + np.arange(len(diff)), diff)

    @pyqtSlot(tuple)
    def onRoiChanged(self, roi_params):
        self._roi_x = roi_params[3]

    def reset(self):
        super().reset()

        self._plots.clear()
        self._legend.clear()
        self.hideLegend()


class DelayScanSlice(PlotWidgetF):
    def __init__(self, delay_scan_view, parent=None):
        super().__init__(parent=parent)

        self.setLabel("left", "Difference")
        self.setLabel("bottom", "Delay [ps]")
        self.setTitle("XES signal vs pump-probe delay")

        self._x_min = None
        self._x_max = None
        self._view = delay_scan_view
        self._plot = self.plotCurve()

    def updateF(self, data):
        if data is None or self._x_min is None or self._x_max is None:
            return

        delay_scan = self._view.buffer.data()
        lineout = np.nanmean(delay_scan[:, self._x_min:self._x_max], axis=1)
        if len(self._view._delays) == len(lineout):
            self._plot.setData(self._view._delays, lineout)

    def onLinearRoiChanged(self, x_min, x_max):
        self._x_min = x_min
        self._x_max = x_max

class CorrelatorPlot(PlotWidgetF):
    """ 
    Correlation plot class.

    Correlation between the AUC of JNGF projection vs average integral(or amplitude) 
    of the peaks in one train
    
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setLabel("left", "AUC JNGF")
        self.setLabel("bottom", "I0 Digitizer")
        self.setTitle("Correlation JNGF vs Digitizer")

        self._plot = self.plotScatter(brush=FColor.mkBrush("w"))
        self.digitizer_data = []
        self.auc_data = []
        self.train_count = 0
        self.train_array = []

    def updateF(self, data):
        if data is None:
            return

        self.digitizer_data.append(data["digi_int_avg"])

        if data["auc"]:
            self.auc_data.append(data["auc"])
        else:
            self.auc_data.append(np.nan)
        self.train_count+=1
        
        self.train_array.append(self.train_count)
        self._plot.setData(self.digitizer_data, self.auc_data)

    def reset(self):
        super().reset()

        self.auc_data.clear()
        self.digitizer_data.clear()
        self.train_array.clear()

class DigitizerPlot(PlotWidgetF):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setLabel("left", "Amplitude")
        self.setLabel("bottom", "Samples")
        self.setTitle("Digitizer " )
        self._plot = self.plotCurve()
        self.digitizer_range = []

    def updateF(self, data):
        if data is None:
            return
        
        if not data["digi_range"]:
            self.digitizer_range = [0,40000]
        else:
            self.digitizer_range = data["digi_range"].split(":")
            
        self.digi_sample_min = int(self.digitizer_range[0])
        self.digi_sample_max = int(self.digitizer_range[1])

        digitizer_data = data["digi_data"]
       
        self._plot.setData(np.arange(len(digitizer_data[self.digi_sample_min:self.digi_sample_max])), 
        digitizer_data[self.digi_sample_min:self.digi_sample_max])


class DelayScanView(ImageViewF):
    linear_roi_changed_sgn = pyqtSignal(int, int)

    def __init__(self, roi, parent=None):
        super().__init__(parent=parent)

        self.setAspectLocked(False)
        self._mouse_hover_v_rounding_decimals = 4

        self._roi = roi
        self._delays = []
        self._buffer = SimpleVectorSequence(10, max_len=10)

        self.linear_roi = LinearRegionItem()
        self.linear_roi.sigRegionChangeFinished.connect(self.onLinearRoiChanged)
        self.addItem(self.linear_roi)

    def onLinearRoiChanged(self, roi):
        r = roi.getRegion()
        self.linear_roi_changed_sgn.emit(*map(int, r))

    def updateF(self, data):
        if data is None:
            return

        delay = data["delay"]
        auto_level = False
        first_image = self.is_first_image
        if delay not in self._delays:
            # If this is a new delay, record it and auto-level the heatmap
            self._delays.append(delay)
            auto_level = True

        latest_xes = data["xes"][delay].difference[self._roi.index]
        delay_index = self._delays.index(delay)
        buffer_entries = self._buffer.data().shape[0]

        # If necessary, rebuild the whole delay plot
        if data["refresh_plots"]:
            # Get the current ROI width to set the right size of the buffer
            width, _ = map(int, self._roi.size())
            self.linear_roi.setBounds([0, width])
            self._buffer = SimpleVectorSequence(width, max_len=5000, dtype=np.float32)

            # Add each delay back, in the order that we first saw it
            for delay in self._delays:
                diff = data["xes"][delay].difference[self._roi.index]

                # The very last delay might be new and not have enough data for
                # the XES signal to be calculated, in which case we skip it.
                if diff is not None and len(diff) == self._buffer.size:
                    self._buffer.append(diff)

            auto_level = True

        # Otherwise, append to or update the buffer
        else:
            # If the XES hasn't been calculated yet (like when a new ROI has
            # just been created), then this will be None and we can ignore
            # it. Or, the user might have changed the ROI size in between
            # trains, which means that the array size will have changed and the
            # buffer will need to be rebuilt in the next train.
            if latest_xes is None or len(latest_xes) != self._buffer.size:
                return

            # If we haven't seen this delay before, add a new entry to the buffer
            if delay_index == buffer_entries:
                self._buffer.append(latest_xes)
            else:
                # Otherwise, update the existing entry
                self._buffer[delay_index][...] = latest_xes

        self.setImage(self._buffer.data(), auto_levels=auto_level)

        # If this is the first image we get, set the linear ROI to a reasonable
        # default.
        if first_image:
            width = self._buffer.sizesplit

    @pyqtSlot(tuple)
    def onRoiChanged(self, roi_params):
        _, activated, _, _, _, width, _ = roi_params

        if not activated:
            self._buffer.reset()

    @property
    def buffer(self):
        return self._buffer

    def reset(self):
        super().reset()

        self._delays.clear()

class FuzzyCombobox(QComboBox):
    def __init__(self, device_keywords=[], property_keywords=[]):
        super().__init__()

        self.addItem("")
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.device_keywords = device_keywords
        self.property_keywords = property_keywords

    def updateSources(self, sources):
        model = self.model()

        for source in sources:
            result = model.match(model.index(0, 0), Qt.UserRole, source)
            if len(result) == 0:
                display_name = self.format_source(source)
                self.addItem(display_name, userData=source)

        # If no item is currently selected, attempt to automatically select splitone
        if self.currentText() == "" and len(self.device_keywords) > 0:
            # Note that we skip the first (empty) row when iterating
            all_sources = [model.data(model.index(row, 0), role=Qt.UserRole)
                           for row in range(1, model.rowCount())]

            # Filter by device
            candidates = [source for source in all_sources
                          if any(kw.lower() in source.split()[0].lower() for kw in self.device_keywords)]

            # If possible, filter by property
            if len(self.property_keywords) > 0:
                candidates = [source for source in candidates
                              if any(kw.lower() in source.split()[1].lower() for kw in self.property_keywords)]
            
            # If we find some candidates, just use the first one
            if len(candidates) > 0:
                self.setCurrentText(self.format_source(candidates[0]))

    def format_source(self, source):
        device, prop = source.split()

        if ":" in device:
            return f"{device}[{prop}]"
        else:
            return f"{device}.{prop}"

    def enterEvent(self, event):
        QToolTip.showText(QCursor.pos(), self.toolTip(), self)


@dataclass
class WidgetData:
    args: list
    tab_data: list = None
    docked: bool = True

class DockingTabWidget(QTabWidget):
    def __init__(self, with_close_btn=True):
        super().__init__()

        self.setMovable(True)
        self.with_close_btn = with_close_btn
        self.widget_data = { }

    def addTab(self, widget, *args):
        return self.insertTab(self.count(), widget, *args)

    def insertTab(self, idx, widget, *args):
        idx = super().insertTab(idx, widget, *args)

        self._setTabButtons(idx)

        # If we haven't seen this widget before, record it, and override its
        # closeEvent() to handle docking properly.
        if widget not in self.widget_data:
            self.widget_data[widget] = WidgetData(args)

            def dockingCloseEvent(event):
                if not self.widget_data[widget].docked:
                    widget.setWindowFlags(Qt.Widget)
                    self._dockTab(widget)
                    event.ignore()
                else:
                    widget.__originalCloseEvent(event)

            widget.__originalCloseEvent = widget.closeEvent
            widget.closeEvent = dockingCloseEvent

        return idx

    def _setTabButtons(self, idx):
        widget = self.widget(idx)

        # Set widget buttons
        undock_btn = QToolButton()
        undock_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        undock_btn.clicked.connect(lambda: self._undockTab(widget))

        buttons_hbox = QHBoxLayout()
        buttons_hbox.addWidget(undock_btn)
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_hbox)

        if self.with_close_btn:
            close_btn = QToolButton()
            close_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
            # close_btn.clicked.connect(lambda: self._removeTab(splitter))
            buttons_hbox.addWidget(close_btn)

        self.tabBar().setTabButton(idx, QTabBar.RightSide, buttons_widget)

    def _dockTab(self, widget):
        widget_data = self.widget_data[widget]

        idx = self.addTab(widget, *widget_data.args)
        self.tabBar().setTabData(idx, widget_data.tab_data)
        widget_data.docked = True

    def _undockTab(self, widget):
        idx = self.indexOf(widget)
        label = self.tabText(idx)
        self.widget_data[widget].docked = False
        self.widget_data[widget].tab_data = self.tabBar().tabData(idx)
        self.removeTab(idx)

        # Find all windows
        windows = [w for w in QApplication.topLevelWidgets()
                   if w.isWindow() and not w.isHidden()]
        # Get the main window geometry
        for window in windows:
            if isinstance(window, _SpecialAnalysisBase):
                geom = window.frameGeometry()

        # Position and resize the widget sensibly
        widget.resize(geom.width() - 150, geom.height() - 150)
        # Offset each successive window
        center_offset = len(windows) * 50
        center = geom.translated(center_offset, center_offset).center()
        widget.move(center - widget.frameGeometry().center())

        # Show the widget as a window
        widget.setWindowFlags(Qt.Window)
        widget.setWindowTitle(label)
        widget.show()


class XesCtrlWidget(_BaseAnalysisCtrlWidgetS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img_display_cb = QComboBox()
        self.img_display_cb.addItem(DisplayOption.PUMPED.value)
        self.img_display_cb.addItem(DisplayOption.UNPUMPED.value)
        self.img_display_cb.addItem(DisplayOption.DIFFERENCE.value)

        self.pumped_train_cb = QComboBox()
        self.pumped_train_cb.addItem("Odd")
        self.pumped_train_cb.addItem("Even")

        self.digitizer_type_analysis_cb = QComboBox()
        self.digitizer_type_analysis_cb.addItem("Digitizer amplitude peaks")
        self.digitizer_type_analysis_cb.addItem("Digitizer integral peaks")

        self.detector_cb = FuzzyCombobox(device_keywords=["jf", "jngfr", "jungfrau"],
                                         property_keywords=["data.adc"])
        self.delay_cb = FuzzyCombobox(device_keywords=["ppodl"],
                                             property_keywords=["actualPosition"])
        self.target_delay_cb = FuzzyCombobox(device_keywords=["ppodl"],
                                             property_keywords=["targetPosition"])
        self.digitizer_cb = FuzzyCombobox(device_keywords=["ADC"],
                                             property_keywords=["samples"])

        self.digitizer_range_cb = SmartLineEdit()
        # self._digitizer_min.setValidator(QIntValidator(1, 10000))

        for cb in [self.detector_cb, self.delay_cb, self.target_delay_cb, self.digitizer_cb]:
            cb.setToolTip("This property will be auto-selected if possible (click the Start button).<br><br> <b>Warning:</b> changing this property will force the program to reset.")

        self.save_btn = QPushButton("Save XES data")

        self._non_reconfigurable_widgets = [
            self.detector_cb,
            self.delay_cb,
            self.target_delay_cb,
            self.save_btn,
            self.digitizer_cb,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()
        layout.addRow("Display: ", self.img_display_cb)
        layout.addRow("Pumped trains are: ", self.pumped_train_cb)
        layout.addRow("Detector: ", self.detector_cb)
        layout.addRow("Delay property: ", self.delay_cb)
        layout.addRow("Target delay property: ", self.target_delay_cb)
        layout.addRow("Digitizer: ", self.digitizer_cb)
        layout.addRow("Digitizer analysis:", self.digitizer_type_analysis_cb)
        layout.addRow("Digitizer slice(min:max):", self.digitizer_range_cb)
#    hbox.addWidget(r1)
#    hbox.addWidget(r2)
#    hbox.addStretch()
#    fbox.addRow(QLabel("sex"),hbox)
        layout.setItem(layout.count(), QFormLayout.SpanningRole, QSpacerItem(0, 20))
        layout.addRow(self.save_btn)


    def initConnections(self):
        """Override."""
        pass


    @pyqtSlot(dict)
    def onNewTrainData(self, data):
        keys = set(key for key in data.keys() if not key.endswith(".timestamp") 
        and not key.endswith(".timestamp.tid") )
        fast_data = set(key for key in keys if ":" in key and not "digitizer" in key)
        slow_data = set(key for key in keys if not ":" in key and not "digitizer" in key)
        digitizer_data = set(key for key in keys if "raw.samples" in key)
   
        self.detector_cb.updateSources(fast_data)
        self.delay_cb.updateSources(slow_data)
        self.target_delay_cb.updateSources(slow_data)
        self.digitizer_cb.updateSources(digitizer_data)

@create_special(XesCtrlWidget, XesTimingProcessor)
class XesTimingWindow(_SpecialAnalysisBase):
    """Main GUI for XES timing."""

    icon = "icon_ppXES.svg"
    _title = "ppXES (JNGFR)"
    _long_title = "X-ray Emission Spectroscopy tool for the JF"
    _client_support = ClientType.KARABO_BRIDGE

    def __init__(self, topic):
        super().__init__(topic, with_dark=False)

        self._view = DetectorView(parent=self)
        self._visualized_rois = set()
        # self.digitizer_range = "0:100000"

        self.initUI()
        self.initConnections()
        self.startWorker()

        center_window(self)

    def initUI(self):
        """Override."""
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(QLabel("Enable some ROIs to analyse XES signals"))
        layout.addStretch()
        default_widget = QWidget()
        default_widget.setLayout(layout)

        self._tab_widget = DockingTabWidget(with_close_btn=False)
        self._stacked_widget = QStackedWidget()
        self._stacked_widget.addWidget(default_widget)
        self._stacked_widget.addWidget(self._tab_widget)
        
        correlator = CorrelatorPlot(parent=self)
        digitizer = DigitizerPlot(parent=self)
        digitizer_ui = QSplitter(Qt.Vertical)
        digitizer_ui.addWidget(digitizer)
        digitizer_ui.addWidget(correlator)
        hsplitter = QSplitter()
        hsplitter.addWidget(self._view)
        hsplitter.addWidget(digitizer_ui)
        hsplitter.setStretchFactor(0, 10)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(hsplitter)
        right_panel.addWidget(self._stacked_widget)
        right_panel.setSizes([int(2 * self._TOTAL_H / 5), int(3 * self._TOTAL_H / 5)])
        
        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([int(self._TOTAL_W / 4), int(3 * self._TOTAL_W / 4)])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        worker = self._worker_st
        ctrl = self._ctrl_widget_st

        # Tell the worker what image data to return for display
        display_cb = ctrl.img_display_cb
        display_cb.currentTextChanged.connect(worker.onImgDisplayChanged)
        display_cb.currentTextChanged.connect(self._view.onImgDisplayChanged)
        display_cb.currentTextChanged.emit(display_cb.currentText())

        # Tell the worker which trains to treat as pumped
        pumped_train = ctrl.pumped_train_cb
        pumped_train.currentIndexChanged.connect(worker.onEvenTrainsPumpedChanged)
        pumped_train.currentIndexChanged.emit(pumped_train.currentIndex())

        # Tell the worker whether integral or amplitude of digitizer peaks
        digitizer_type_analysis = ctrl.digitizer_type_analysis_cb
        digitizer_type_analysis.currentIndexChanged.connect(worker.onDigitizerAnalysisTypeChanged)
        digitizer_type_analysis.currentIndexChanged.emit(digitizer_type_analysis.currentIndex())
        

        ctrl.detector_cb.currentIndexChanged.connect(self.onDetectorChanged)
        ctrl.delay_cb.currentIndexChanged.connect(self.onDelayDeviceChanged)
        ctrl.target_delay_cb.currentIndexChanged.connect(self.onTargetDelayDeviceChanged)
        ctrl.digitizer_cb.currentIndexChanged.connect(self.onDigitizerDeviceChanged)
        ctrl.digitizer_range_cb.value_changed_sgn.connect(self.onDigitizerSliceChanged)
        ctrl.digitizer_range_cb.value_changed_sgn.emit(ctrl.digitizer_range_cb.text())
        ctrl.save_btn.clicked.connect(self.onSaveData)

        for roi_ctrl in self._com_ctrl_st.roi_ctrls:
            roi_ctrl.roi_geometry_change_sgn.connect(self.onRoiChanged)

        # Update the options for the current data
        worker.new_train_data_sgn.connect(self._ctrl_widget_st.onNewTrainData)

    def onDetectorChanged(self, index):
        key = self._ctrl_widget_st.detector_cb.currentData()
        self._worker_st.setDetectorDevice(*key.split())
  
    def onDelayDeviceChanged(self, index):
        key = self._ctrl_widget_st.delay_cb.currentData()
        self._worker_st.setDelayDevice(*key.split())

    def onTargetDelayDeviceChanged(self, index):
        key = self._ctrl_widget_st.target_delay_cb.currentData()
        self._worker_st.setTargetDelayDevice(*key.split())

    def onDigitizerDeviceChanged(self, index):
        key = self._ctrl_widget_st.digitizer_cb.currentData()
        self._worker_st.setDigitizerDevice(*key.split())

    def onDigitizerSliceChanged(self):
        digitizer_range = self._ctrl_widget_st.digitizer_range_cb.text()
        self._worker_st.SetDigitizerSlice(*[digitizer_range])

    def onSaveData(self):
        file_path = QFileDialog.getSaveFileName(filter="HDF5 (*.h5)")[0]
        if len(file_path) == 0:
            return

        if not file_path.endswith(".h5"):
            file_path += ".h5"

        data = self._worker_st.xes_delay_data
        rois = { idx: geom for idx, geom in self._worker_st._rois_geom_st.items()
                if geom is not None }

        delays = np.array(list(data.keys()))
        pumped_train_avgs = []
        unpumped_train_avgs = []
        xes_signals = defaultdict(list)

        # Load delay data
        for delay in delays:
            pumped_train_avgs.append(data[delay].pumped_train_avg)
            unpumped_train_avgs.append(data[delay].unpumped_train_avg)

            for roi_idx in rois:
                xes_signals[roi_idx].append(data[delay].difference[roi_idx])

        # Convert to numpy arrays for convenience when saving with h5py
        pumped_train_avgs = np.array(pumped_train_avgs)
        unpumped_train_avgs = np.array(unpumped_train_avgs)

        try:
            with h5py.File(file_path, "w") as f:
                f.create_dataset("delays", delays.shape, delays.dtype, data=delays)
                f.create_dataset("pumped_train_avgs", pumped_train_avgs.shape,
                                 pumped_train_avgs.dtype, data=pumped_train_avgs)
                f.create_dataset("unpumped_train_avgs", unpumped_train_avgs.shape,
                                 unpumped_train_avgs.dtype, data=unpumped_train_avgs)

                ctrl = self._ctrl_widget_st
                f.attrs["even_trains_pumped"] = self._worker_st._even_trains_pumped
                f.attrs["detector"] = ctrl.detector_cb.currentText()
                f.attrs["delay_property"] = ctrl.delay_cb.currentText()
                f.attrs["target_delay_property"] = ctrl.target_delay_cb.currentText()

                for roi_idx in xes_signals:
                    group = f.create_group(f"roi_{roi_idx}")
                    x, y, w, h = rois[roi_idx]
                    group.attrs["x"] = x
                    group.attrs["y"] = y
                    group.attrs["width"] = w
                    group.attrs["height"] = h

                    xes_diff = np.array(xes_signals[roi_idx])
                    group.create_dataset("xes_difference", xes_diff.shape,
                                         xes_diff.dtype, data=xes_diff)

            logger.info("Data saved")
        except BlockingIOError:
            QMessageBox.warning(self, "Could not save data",
                                "Could not save data to this file (it's probably open in some other program, close any open programs using it and try again).")

    @pyqtSlot(tuple)
    def onRoiChanged(self, roi_params):
        idx, activated = roi_params[:2]

        # If a ROI has been activated, add a tab for it
        if activated and idx not in self._visualized_rois:
            xes_plot = XesSignalPlot(idx, parent=self)
            delay_scan = DelayScanView(self._view.rois[idx - 1], parent=self)
            delay_slice = DelayScanSlice(delay_scan, parent=self)
        
            vsplitter = QSplitter(Qt.Vertical)
            vsplitter.addWidget(delay_slice)
            vsplitter.addWidget(delay_scan)
            vsplitter.setStretchFactor(0, 10)
            vsplitter.setStretchFactor(1, 1)

            hsplitter = QSplitter()
            hsplitter.addWidget(xes_plot)
            hsplitter.addWidget(vsplitter)
            hsplitter.setStretchFactor(0, 10)
            hsplitter.setStretchFactor(1, 1)

            roi_ctrl = self._com_ctrl_st.roi_ctrls[idx - 1]
            roi_ctrl.roi_geometry_change_sgn.connect(delay_scan.onRoiChanged)
            roi_ctrl.roi_geometry_change_sgn.connect(xes_plot.onRoiChanged)
            xes_plot.onRoiChanged(roi_ctrl.params())
            delay_scan.onRoiChanged(roi_ctrl.params())

            delay_scan.linear_roi_changed_sgn.connect(delay_slice.onLinearRoiChanged)

            self._visualized_rois.add(idx)

            tab_idx = self._tab_widget.addTab(hsplitter, f"ROI {idx}")
            self._tab_widget.tabBar().setTabData(tab_idx, idx)
            self._tab_widget.setCurrentIndex(tab_idx)
            if self._stacked_widget.currentIndex() == 0:
                self._stacked_widget.setCurrentIndex(1)

        # Or if it's been deactivated, remove its tab
        elif not activated and idx in self._visualized_rois:
            self._visualized_rois.remove(idx)
            tab_idx = self.roiTabIndex(idx)
            hsplitter = self._tab_widget.widget(tab_idx)
            self._tab_widget.removeTab(tab_idx)

            # Delete the widget and its sub-widgets
            hsplitter.deleteLater()

            if self._tab_widget.count() == 0:
                self._stacked_widget.setCurrentIndex(0)

    def roiTabIndex(self, roi_idx):
        for i in range(self._tab_widget.count()):
            if self._tab_widget.tabBar().tabData(i) == roi_idx:
                return i

        return -1
