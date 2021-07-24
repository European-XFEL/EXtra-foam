"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Cammille Carinan <cammille.carinan@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from string import Formatter

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QColor, QIntValidator, QTransform
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QStyle, QTabBar,
    QVBoxLayout, QWidget)

from extra_foam.gui import pyqtgraph as pg
from extra_foam.gui.ctrl_widgets import (
    SmartLineEdit, SmartSliceLineEdit, SmartStringLineEdit
)
from extra_foam.gui.misc_widgets import FColor
from extra_foam.gui.plot_widgets import ImageViewF, PlotWidgetF

from .processor import SpotSizeProcessor
from ..special_analysis_base import (
    create_special, QThreadKbClient, _BaseAnalysisCtrlWidgetS,
    _SpecialAnalysisBase
)

# -----------------------------------------------------------------------------
# Image Views


class ImageView(ImageViewF):
    def updateF(self, data):
        self.setImage(data["view"]["image"])


# -----------------------------------------------------------------------------
# Plots


class ScaledPlot(PlotWidgetF):

    header = ''
    x_label = ''
    y_label = ''

    scale_value = 1
    scale_unit = "pixels"
    scale_axis = 'y'

    legend_template = ''

    def __init__(self, *, parent=None):
        super(ScaledPlot, self).__init__(parent=parent)
        if self.header:
            self.setTitle(self.header)
        self.set_xlabel()
        self.set_ylabel()

        self.legend_item = None
        if self.legend_template:
            self.add_legend_item(template=self.legend_template)

    def reset(self):
        super(ScaledPlot, self).reset()
        if self.legend_item is not None:
            self.legend_item.format()

    def set_scale_value(self, value):
        self.scale_value = value

    def set_scale_unit(self, unit):
        self.scale_unit = unit
        if self.scale_axis == 'x':
            self.set_xlabel()
        else:
            self.set_ylabel()

    def set_ylabel(self):
        if self.y_label:
            self.setLabel('left', f"{self.y_label} ({self.scale_unit})")

        if not self.y_label:
            return
        label = [self.y_label]
        if self.scale_axis == 'y':
            label.append(f"({self.scale_unit})")
        self.setLabel('left', " ".join(label))

    def set_xlabel(self):
        if not self.x_label:
            return
        label = [self.x_label]
        if self.scale_axis == 'x':
            label.append(f"({self.scale_unit})")
        self.setLabel('bottom', " ".join(label))

    def add_legend_item(self, template=''):
        # Initialize legend item
        self.legend_item = legend = LegendItem(template=template)
        legend.setParentItem(self._plot_area.getViewBox())
        legend.anchor(itemPos=(0, 1),
                      parentPos=(0, 1),
                      offset=(5, -5))
        return legend


class ProjectionPlot(ScaledPlot):

    header = 'Projection'
    x_label = 'Axis'
    y_label = 'Intensity'
    scale_axis = 'x'

    legend_template = ("Position: {pos}<br>"
                       "FWHM: {width}")

    def __init__(self, *, parent=None):
        super().__init__(parent=parent)
        self._raw = self.plotCurve(name="Current", pen=FColor.mkPen("p"))
        self._fit = self.plotCurve(name="Fitted", pen=FColor.mkPen("g"))

    def updateF(self, data):
        """Override."""
        proj = data["view"]["proj"]
        if proj is None:
            return

        # Plot raw projection
        x, y, fit = proj.x * self.scale_value, proj.y, proj.fit
        self._raw.setData(x, y)

        # Plot fitted projection
        if fit is None:
            x, fit = [], []  # clear the curve by setting empty arrays
        self._fit.setData(x, fit)

        # Update legends
        self.legend_item.format(
            pos=f"{proj.pos * self.scale_value:.2f}",
            width=f"{proj.width * self.scale_value:.2f}")


class ScatterPlot(ScaledPlot):

    name = None
    pen = FColor.mkPen("p")
    brush = FColor.mkBrush("p")
    point_size = 5

    def __init__(self, *, parent=None):
        super().__init__(parent=parent)
        self.scatter = self.plotScatter(name=self.name,
                                        pen=self.pen,
                                        brush=self.brush,
                                        size=self.point_size)

    def updateF(self, data):
        """Override."""
        raise NotImplementedError


class ScanPlot(ScatterPlot):

    header = "Bender Scan"
    x_label = "q - q_r"
    y_label = "FWHM"

    def updateF(self, data):
        """Override."""
        self.scatter.setData(data["device"],
                             data["view"]["widths"] * self.scale_value)


SYMBOL_COLUMN = 0
TEXT_COLUMN = 1

TEXT_SIZE = "11pt"
LEGEND_PEN = pg.mkPen(QColor(192, 192, 192, 200))
LEGEND_BRUSH = pg.mkBrush(QColor(0, 0, 0, 50))
LABEL_COLOR = QColor(0, 0, 0, 200)


class BlankFormatter(Formatter):
    def get_value(self, key, args, kwds):
        if isinstance(key, str):
            try:
                return kwds[key]
            except KeyError:
                return '-'
        else:
            return Formatter.get_value(key, args, kwds)


class LegendItem(pg.LegendItem):
    """A modified pyqtgraph LegendItem to enable annotations
       and text formatting on plots. Based on KaraboLegendItem from Karabo GUI.
    """

    formatter = BlankFormatter()

    def __init__(self, size=None, offset=None, template=None):
        super(LegendItem, self).__init__(
            size, offset, pen=LEGEND_PEN, brush=LEGEND_BRUSH,
            labelTextColor=LABEL_COLOR)

        self._template = template

        # Add internal label
        self._label = pg.LabelItem(color='w', size=TEXT_SIZE)
        self.layout.addItem(self._label, 0, 0)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.format()

    def addItem(self, item, name):
        """Reimplemented function of LegendItem

        :param item: A PlotDataItem from which the line and point style
                     of the item will be determined
        :param name: The title to display for this item. Simple HTML allowed.
        """
        label = pg.LabelItem(name, justify='left',
                             color=self.opts['labelTextColor'],
                             size=TEXT_SIZE)
        row = self.layout.rowCount()
        self.items.append((item, label))
        self.layout.addItem(item, row, SYMBOL_COLUMN)
        self.layout.addItem(label, row, TEXT_COLUMN)
        self.updateSize()

    def format(self, **kwargs):
        text = self.formatter.format(self._template, **kwargs)
        self._label.setText(text)


class Histogram(ScaledPlot):

    header = "   "  # Empty title to align with the partner scatter plot
    x_label = "Count"

    legend_template = ("Mean: {mean}<br>"
                       "Std: {std}")

    def __init__(self, *, parent=None):
        super(Histogram, self).__init__(parent=parent)
        self.setMaximumWidth(300)

        # Setup rotated plot
        self._plot_area.getViewBox().invertX()

        # Initialize plot item
        self.bar_item = self.plotBar()
        # Rotate the item as the orientation of the plot is left
        transform = QTransform()
        transform.rotate(270)  # degrees
        transform.scale(*(-1, 1))
        self.bar_item.setTransform(transform)

    def updateF(self, data):
        """Override."""
        raise NotImplementedError


class AvgPositionHistogram(Histogram):

    def updateF(self, data):
        hist = data["pos_train"]["mean_hist"]
        if hist is None:
            return

        self.bar_item.setData(hist.bins * self.scale_value, hist.count)
        self.legend_item.format(mean=f"{hist.mean * self.scale_value:.2f}",
                                std=f"{hist.std * self.scale_value:.2f}")


class StdPositionHistogram(Histogram):

    def updateF(self, data):
        hist = data["pos_train"]["std_hist"]
        if hist is None:
            return

        self.bar_item.setData(hist.bins * self.scale_value, hist.count)
        self.legend_item.format(mean=f"{hist.mean * self.scale_value:.2f}",
                                std=f"{hist.std * self.scale_value:.2f}")


class AvgPositionTrainPlot(ScatterPlot):

    header = "Average Beam Position"
    x_label = "Train ID"

    pen = FColor.mkPen("b")
    brush = FColor.mkBrush("b")

    def updateF(self, data):
        """Override."""
        self.scatter.setData(data["trainIds"],
                             data["pos_train"]["mean"] * self.scale_value)


class StdPositionTrainPlot(ScatterPlot):

    header = "Standard Deviation of the Beam Position"
    x_label = "Train ID"

    pen = FColor.mkPen("b")
    brush = FColor.mkBrush("b")

    def updateF(self, data):
        """Override."""
        self.scatter.setData(data["trainIds"],
                             data["pos_train"]["std"] * self.scale_value)


class PulsePlot(ScatterPlot):

    pen = FColor.mkPen("g")
    brush = FColor.mkBrush("g")
    point_size = 8

    x_label = "Pulse in a train"

    legend_template = ("Mean: {mean}<br>"
                       "Std: {std}")

    def __init__(self, *, parent=None):
        super(PulsePlot, self).__init__(parent=parent)
        # Initialize legend item
        self.stat_bar = self.plotStatisticsBar(beam=0.25,
                                               pen=self.pen)

    def updateF(self, data):
        """Override."""
        raise NotImplementedError


class AvgPositionPulsePlot(PulsePlot):

    header = "Beam position averaged over all trains"

    def updateF(self, data):
        x = data["pulses"]

        pos = data["pos_pulse"]
        y = pos["mean"] * self.scale_value
        std = pos["std"] * self.scale_value

        self.scatter.setData(x, y)
        self.stat_bar.setData(x, y, y_min=y-std, y_max=y+std)
        self.legend_item.format(mean=f"{y.mean():.2f}",
                                std=f"{std.mean():.2f}")


class AvgWidthPulsePlot(PulsePlot):

    header = "Beam FWHM averaged over all trains"

    def updateF(self, data):
        x = data["pulses"]

        width = data["width_pulse"]
        y = width["mean"] * self.scale_value
        std = width["std"] * self.scale_value

        self.scatter.setData(x, y)
        self.stat_bar.setData(x, y, y_min=y-std, y_max=y+std)
        self.legend_item.format(mean=f"{y.mean():.2f}",
                                std=f"{std.mean():.2f}")


# -----------------------------------------------------------------------------
# Control

AXES = ("x", "y")
DETECTOR = "SCS_CDIDET_MTE3/CAM/CAMERA:output"


class SpotSizeCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Gotthard pump-probe analysis control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.detector = SmartStringLineEdit(DETECTOR)
        self.motor = SmartStringLineEdit("SCS_KBS_HFM/MOTOR/BENDERB")
        self.load_dark_run_button = QPushButton("Load dark run")
        self.subtract_dark_checkbox = QCheckBox("Subtract")
        self.init_group = QGroupBox("Initialize run")

        # Run settings
        self.pulse_slice = SmartSliceLineEdit("::4")
        # Project on axis
        self.axis = QComboBox()
        for item in ("x", "y"):
            self.axis.addItem(item)

        # Bender scan settings
        # Train- or pulse-resolved combobox
        self.analysis_type = QComboBox()
        for item in ("Train", "Pulse"):
            self.analysis_type.addItem(item)
        # Pulse line edit
        validator = QIntValidator()
        validator.setRange(0, 63)
        self.pulse = SmartLineEdit("0")
        self.pulse.setValidator(validator)

        # Positional jitter settings
        # Histogram bins
        validator = QIntValidator()
        validator.setBottom(1)
        self.hist_bins = SmartLineEdit("10")
        self.hist_bins.setValidator(validator)

        # Calibration factor
        self.calib_group = QGroupBox("Calibration Factor")
        self.calib_group.setCheckable(True)
        self.calib_value_lineedit = SmartLineEdit("1")
        self.calib_unit_lineedit = SmartStringLineEdit("pixels")

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        # group widgets for loading dark
        dark_layout = QHBoxLayout()
        dark_layout.addWidget(self.load_dark_run_button)
        dark_layout.addWidget(self.subtract_dark_checkbox)
        dark_widget = QWidget()
        dark_widget.setLayout(dark_layout)

        # group widgets for run initialization
        init_layout = QFormLayout()
        init_layout.addRow("Detector:", self.detector)
        init_layout.addRow("Motor:", self.motor)
        init_layout.addRow(dark_widget)
        self.init_group.setLayout(init_layout)

        # group widgets for run settings
        run_layout = QFormLayout()
        # run_layout.addRow("Pulse slice:", self.pulse_slice)
        run_layout.addRow("Axis:", self.axis)
        run_group = QGroupBox("Run settings")
        run_group.setLayout(run_layout)

        # group widgets for bender scan settings
        self.pulse.setDisabled(not bool(self.analysis_type.currentIndex()))
        scan_layout = QFormLayout()
        scan_layout.addRow("Analysis Type:", self.analysis_type)
        scan_layout.addRow("Pulse:", self.pulse)
        scan_group = QGroupBox("Bender scan settings")
        scan_group.setLayout(scan_layout)

        # group widgets for positional jitter settings
        jitter_layout = QFormLayout()
        jitter_layout.addRow("Histogram bins:", self.hist_bins)
        jitter_group = QGroupBox("Positional jitter settings")
        jitter_group.setLayout(jitter_layout)

        # group widgets for calibration factor
        calib_label = QLabel("1 pixel =")
        calib_layout = QHBoxLayout()
        calib_layout.addWidget(calib_label)
        calib_layout.addWidget(self.calib_value_lineedit)
        calib_layout.addWidget(self.calib_unit_lineedit)
        self.calib_group.setLayout(calib_layout)

        # Delete existing layout
        QWidget().setLayout(self.layout())

        # Set a new layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.addWidget(self.init_group)
        layout.addWidget(run_group)
        # layout.addWidget(scan_group)
        layout.addWidget(jitter_group)
        layout.addWidget(self.calib_group)
        layout.addStretch(1)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        # disable_pulse_num = lambda x: self.pulse.setDisabled(not bool(x))
        # self.analysis_type.currentIndexChanged.connect(disable_pulse_num)

        self.calib_group.toggled.connect(self.enable_calibration)

    def onStartST(self):
        super(SpotSizeCtrlWidget, self).onStartST()
        self.init_group.setDisabled(True)

    def onStopST(self):
        super(SpotSizeCtrlWidget, self).onStopST()
        self.init_group.setDisabled(False)

    @pyqtSlot(bool)
    def enable_calibration(self, enabled):
        self.calib_value_lineedit.setDisabled(not enabled)
        self.calib_unit_lineedit.setDisabled(not enabled)


# -----------------------------------------------------------------------------
# Window


class DetachableDockWidget(QDockWidget):

    def __init__(self, title, parent=None):
        super(DetachableDockWidget, self).__init__(title, parent=parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea)
        self.setTitleBarWidget(QWidget())  # Remove title bar
        self.setFeatures(QDockWidget.DockWidgetFloatable)
        self.topLevelChanged.connect(self._on_floating_mode)
        self.dockLocationChanged.connect(self.add_button)

    def closeEvent(self, event):
        """Dock the widget instead of closing the window
           and potentially destroying the children."""
        self.setFloating(False)
        event.ignore()

    def add_button(self):
        parent = self.parent()
        if not isinstance(parent, QMainWindow):
            return

        # Create the button
        button = QPushButton()
        button.resize(15, 15)
        button.clicked.connect(self._set_floating_mode)
        # Set style
        icon = self.style().standardIcon(QStyle.SP_TitleBarNormalButton,
                                         widget=self)
        button.setIcon(icon)

        # Find index of the the dock widget on the tabbar and  add the button
        title = self.windowTitle()
        tabbar = parent.findChild(QTabBar)
        for index in range(tabbar.count()):
            if tabbar.tabText(index) != title:
                continue
            tabbar.setTabButton(index, QTabBar.RightSide, button)

    @pyqtSlot()
    def _set_floating_mode(self):
        self.setFloating(not self.isFloating())

    @pyqtSlot(bool)
    def _on_floating_mode(self, is_floating):
        if is_floating:
            self.setWindowFlags(Qt.CustomizeWindowHint
                                | Qt.Window
                                | Qt.WindowMinimizeButtonHint
                                | Qt.WindowMaximizeButtonHint
                                | Qt.WindowCloseButtonHint)
        self.show()


@create_special(ctrl_klass=SpotSizeCtrlWidget,
                worker_klass=SpotSizeProcessor,
                client_klass=QThreadKbClient)
class SpotSizeGrating(_SpecialAnalysisBase):
    """Main GUI for spot size grating analysis."""

    icon = "xes_timing.png"  # TODO: add a new icon
    _title = "Spot Size Grating"
    _long_title = "Spot size grating analysis"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False)
        # Initialize plot widgets
        self._image_view = ImageView(has_roi=True, parent=self)
        self._image_view.setTitle("Current image")
        self._projection_view = ProjectionPlot(parent=self)
        self._scan_view = ScanPlot(parent=self)

        # Train-resolve position jitter views
        self._avg_pos_hist = AvgPositionHistogram(parent=self)
        self._avg_pos_train = AvgPositionTrainPlot(parent=self)
        self._std_pos_hist = StdPositionHistogram(parent=self)
        self._std_pos_train = StdPositionTrainPlot(parent=self)

        # self._avg_pos_pulse = AvgPositionPulsePlot(parent=self)
        # self._avg_width_pulse = AvgWidthPulsePlot(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        main_dock = self._init_main_dock()
        self.addDockWidget(Qt.RightDockWidgetArea, main_dock)

        jitter_dock = self._init_jitter_dock()
        self.tabifyDockWidget(main_dock, jitter_dock)

        self.setDockOptions(self.AnimatedDocks | self.ForceTabbedDocks)

        cw = self.centralWidget()
        cw.setSizes([self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])
        self.resize(self._TOTAL_W, self._TOTAL_H)

    def _init_main_dock(self):
        # Set up layout
        layout = QGridLayout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(self._image_view, 0, 0)
        layout.addWidget(self._projection_view, 0, 1)
        layout.addWidget(self._scan_view, 1, 0, 1, 2)

        # Create a container widget and add the layout
        container = QWidget()
        container.setLayout(layout)

        # Put the container inside a dock
        dock = QDockWidget("Bender Scan", parent=self)
        dock.setTitleBarWidget(QWidget())  # no title!
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(container)

        return dock

    def _init_jitter_dock(self):
        # Set up layout
        layout = QGridLayout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        pos_hbox = QHBoxLayout()
        pos_hbox.addWidget(self._avg_pos_hist, stretch=1)
        pos_hbox.addWidget(self._avg_pos_train, stretch=3)
        pos_container = QWidget()
        pos_container.setLayout(pos_hbox)

        std_hbox = QHBoxLayout()
        std_hbox.addWidget(self._std_pos_hist, stretch=1)
        std_hbox.addWidget(self._std_pos_train, stretch=3)
        std_container = QWidget()
        std_container.setLayout(std_hbox)

        layout.addWidget(pos_container, 0, 0, 1, 2)
        layout.addWidget(std_container, 1, 0, 1, 2)
        # layout.addWidget(self._avg_pos_pulse, 2, 0)
        # layout.addWidget(self._avg_width_pulse, 2, 1)

        # Create a container widget and add the layout
        container = QWidget()
        container.setLayout(layout)

        # Put the container inside a dock
        dock = DetachableDockWidget("Positional Jitter", parent=self)
        dock.setWidget(container)
        return dock

    def initConnections(self):
        """Override."""
        control = self._ctrl_widget_st
        worker = self._worker_st

        # Dark
        control.load_dark_run_button.clicked.connect(self.onLoadDarkRun)
        control.subtract_dark_checkbox.toggled.connect(self.onSubtractDark)
        control.subtract_dark_checkbox.toggled.emit(
            control.subtract_dark_checkbox.isChecked())

        # ROI
        self._com_ctrl_st.roi_geometry_change_sgn.connect(
            self.onRoiGeometryChange)

        # Detector
        control.detector.value_changed_sgn.connect(self.onDetectorChanged)
        control.detector.returnPressed.emit()

        # Motor
        control.motor.value_changed_sgn.connect(self.onDeviceChanged)
        control.motor.returnPressed.emit()

        # Pulse slice
        control.pulse_slice.value_changed_sgn.connect(self.onPulseSliceChanged)
        control.pulse_slice.returnPressed.emit()

        # Pulse
        control.pulse.value_changed_sgn.connect(self.onPulseNumChanged)
        control.pulse.returnPressed.emit()

        # Analysis type
        analysis_type_signal = control.analysis_type.currentTextChanged
        analysis_type_signal.connect(self.onAnalysisTypeChanged)
        analysis_type_signal.emit(control.analysis_type.currentText())

        # Axis
        axis_signal = control.axis.currentTextChanged
        axis_signal.connect(self.onAxisChanged)
        axis_signal.emit(control.axis.currentText())

        # Histogram bins
        on_hist_bins_changed = lambda x: worker.onHistBinsChanged(int(x))
        control.hist_bins.value_changed_sgn.connect(on_hist_bins_changed)
        control.hist_bins.returnPressed.emit()

        # Calibration factor
        calib_value = control.calib_value_lineedit
        calib_value.value_changed_sgn.connect(self.onCalibValueChanged)
        calib_value.returnPressed.emit()

        calib_unit = control.calib_unit_lineedit
        calib_unit.value_changed_sgn.connect(self.onCalibUnitChanged)
        calib_unit.returnPressed.emit()

        control.calib_group.toggled.connect(self.onCalibToggled)

        # TODO: REMOVEME
        # control.axis.setCurrentIndex(1)
        control.subtract_dark_checkbox.toggle()

    # ----------------------------------------------------------------------
    # Slots

    @pyqtSlot(object)
    def onDetectorChanged(self, detector):
        self._worker_st.onDetectorChanged(detector)
        self._onResetST()

    @pyqtSlot(object)
    def onDeviceChanged(self, device):
        self._worker_st.onDeviceChanged(device)
        self._onResetST()

    @pyqtSlot()
    def onLoadDarkRun(self):
        dirpath = QFileDialog.getOpenFileName(
            caption='Open dark images',
            filter='NumPy Files (*.npy)',
            parent=self)[0]

        if not dirpath:
            return

        self._worker_st.onLoadDarkRun(dirpath)
        self._onResetST()

    @pyqtSlot(bool)
    def onSubtractDark(self, subtract):
        self._worker_st.onSubtractDark(subtract)
        self._onResetST()

    @pyqtSlot(object)
    def onPulseSliceChanged(self, pulse_slice):
        pulse_slice = slice(*pulse_slice)
        self._worker_st.onPulseSliceChanged(pulse_slice)

    @pyqtSlot(object)
    def onPulseNumChanged(self, pulse_num):
        pulse_num = int(pulse_num)
        self._worker_st.onPulseNumChanged(pulse_num)

    @pyqtSlot(str)
    def onAnalysisTypeChanged(self, analysis_type):
        self._worker_st.onAnalysisTypeChanged(analysis_type)

    @pyqtSlot(str)
    def onAxisChanged(self, axis):
        axis = AXES.index(axis)
        self._worker_st.onAxisChanged(axis)
        self._onResetST()

    @pyqtSlot(object)
    def onRoiGeometryChange(self, value):
        """Reset everything except for the image widget.
           This is similar from self._onResetST()"""
        for widget in self._plot_widgets_st:
            if isinstance(widget, ImageViewF):
                continue
            widget.reset()
        self._worker_st.onResetST()
        self._client_st.onResetST()

    @pyqtSlot(object)
    def onCalibValueChanged(self, value):
        value = float(value)
        for widget in self._plot_widgets_st:
            if isinstance(widget, ScaledPlot):
                widget.set_scale_value(value)

    @pyqtSlot(object)
    def onCalibUnitChanged(self, unit):
        for widget in self._plot_widgets_st:
            if isinstance(widget, ScaledPlot):
                widget.set_scale_unit(unit)

    @pyqtSlot(bool)
    def onCalibToggled(self, enabled):
        control = self._ctrl_widget_st
        value = control.calib_value_lineedit.text() if enabled else 1
        self.onCalibValueChanged(value)
        unit = control.calib_unit_lineedit.text() if enabled else "pixels"
        self.onCalibUnitChanged(unit)
