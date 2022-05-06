import sys
import time
import textwrap
import itertools
import traceback
import os.path as osp
from enum import Enum
from collections import defaultdict

import lttbc
import libcst as cst
import libcst.matchers as m
import libcst.metadata as cstmeta
import numpy as np
import xarray as xr

from PyQt5.QtGui import QFont, QBrush
from PyQt5.QtCore import Qt, QSettings, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (QSplitter, QPushButton, QWidget, QTabWidget,
                             QStackedWidget, QComboBox, QLabel, QAction, QMenu,
                             QStyle, QTabBar, QToolButton, QFileDialog,
                             QCheckBox, QGridLayout, QHBoxLayout, QMessageBox,
                             QTreeWidget, QTreeWidgetItem, QAbstractItemView,
                             QApplication)
from PyQt5.Qsci import QsciScintilla, QsciLexerPython, QsciAbstractAPIs

from metropc.client import ViewOutput
from metropc import error as mpc_error

from .. import ROOT_PATH
from ..utils import RectROI as MetroRectROI
from ..utils import LinearROI as MetroLinearROI
from ..algorithms import SimpleSequence
from ..pipeline.data_model import ProcessedData
from ..gui.plot_widgets import LinearROI
from ..gui.plot_widgets.image_items import RectROI
from ..gui.misc_widgets import FColor
from ..gui.gui_helpers import center_window
from ..gui.plot_widgets import ImageViewF, PlotWidgetF
from .special_analysis_base import ( create_special, ClientType,
                                     _BaseAnalysisCtrlWidgetS,
                                     _SpecialAnalysisBase )

from . import logger
from .correlator_proc import CorrelatorProcessor, traverse_slots


def sort_views(views):
    return sorted(views, key=lambda v: v.split("#")[1].lstrip("_"))


class SplitDirection(Enum):
    ABOVE = 0
    BELOW = 1
    LEFT = 2
    RIGHT = 3


class UberSplitter(QSplitter):
    def __init__(self, main_window):
        super().__init__()

        self.setChildrenCollapsible(False)
        self.setOrientation(Qt.Vertical)

        self._widgets = []
        self._main_window = main_window
        self._tab_widget = self._main_window._tab_widget
        self.addWidget(self._createWidget())

    def delete(self):
        for widget in self._widgets:
            widget.delete()

        self.deleteLater()

    def _createWidget(self):
        # Create widget
        widget = ViewWidget(self._main_window)
        widget.split_sgn.connect(self._split)
        widget.delete_btn.clicked.connect(lambda: self._onWidgetDeleted(widget))
        self._widgets.append(widget)

        # The first widget created must have its delete button disabled
        if len(self._widgets) == 1:
            self._widgets[0].delete_btn.setEnabled(False)
        elif len(self._widgets) == 2:
            # But when the second is added, the first widget must have it
            # enabled again.
            self._widgets[0].delete_btn.setEnabled(True)

        return widget

    @pyqtSlot(object)
    def _onWidgetDeleted(self, widget):
        self._widgets.remove(widget)

        if len(self._widgets) == 1:
            self._widgets[0].delete_btn.setEnabled(False)

    @pyqtSlot(object, SplitDirection)
    def _split(self, widget, direction):
        is_vertical_split = direction in [SplitDirection.ABOVE, SplitDirection.BELOW]
        split_orientation = Qt.Vertical if is_vertical_split else Qt.Horizontal

        # These two variables will be overwritten later if the widget belongs to
        # a different splitter
        splitter = self
        index = splitter.indexOf(widget)

        # If the source widget is a direct child and the requested split is in a
        # different orientation than this splitter, we need to create a new
        # splitter with the requested orientation, move the source widget into
        # it, and put the new splitter in the source widgets original place in
        # this splitter.
        if index != -1 and split_orientation != self.orientation():
            splitter = QSplitter(split_orientation)
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(self.replaceWidget(index, splitter))
            index = splitter.indexOf(widget)
        # If the source widget is not a direct child, we need to look for it in
        # one of the QSplitter children.
        elif index == -1:
            for i in range(self.count()):
                child = self.widget(i)
                if isinstance(child, QSplitter) and child.indexOf(widget) != -1:
                    # Child QSplitter's are only allowed in one orientation,
                    # opposite to the main orientation. So if the requested
                    # orientation matches the main orientation, then we set the
                    # index to the current QSplitter child.
                    if split_orientation == self.orientation():
                        index = i
                        break
                    # Otherwise we add the new widget to the child QSplitter
                    else:
                        splitter = child
                        index = child.indexOf(widget)
                        break

        new_widget = self._createWidget()
        index_offset = 1 if direction in [SplitDirection.BELOW, SplitDirection.RIGHT] else 0
        splitter.insertWidget(index + index_offset, new_widget)

    @pyqtSlot()
    def undock(self):
        # Position and resize the new window sensibly
        size = self._main_window.frameGeometry()
        self.resize(size.width() - 150, size.height() - 150)
        center = self._main_window.frameGeometry().translated(50, 50).center()
        self.move(center - self.frameGeometry().center())

        # Undock
        self.setWindowFlags(Qt.Window)
        self.show()

    def closeEvent(self, event):
        if self.tab_index == -1:
            self.setWindowFlags(Qt.Widget)
            self._main_window.dock(self)
            event.ignore()

    @property
    def tab_index(self):
        return self._tab_widget.indexOf(self)


class ViewWidget(QStackedWidget):
    # Emitted when the user requests the current widget to be split in a certain
    # direction.
    split_sgn = pyqtSignal(object, SplitDirection)
    # Emitted when the list of available views is updated. Mainly used for
    # testing.
    views_updated_sgn = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()

        self._main_window = main_window
        self._views = { }
        self._plots = { }
        self._error_plots = { }
        self._max_points = 5000
        self._xs = SimpleSequence(max_len=self._max_points)
        self._ys = defaultdict(lambda: SimpleSequence(max_len=self._max_points))
        self._errors = defaultdict(lambda: SimpleSequence(max_len=self._max_points))
        self._annotations = { }
        self._current_view = None

        self._main_window.registerPlotWidget(self)

        self.initUI()
        self.initConnections()
        self.reset()

    def initUI(self):
        # Create view selector widget
        label = QLabel("Select view:")
        self.view_picker = QComboBox()
        self.view_picker.setEnabled(False)
        self.view_picker.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.delete_btn = QPushButton(self.style().standardIcon(QStyle.SP_DialogCloseButton), "")
        self.delete_btn.setToolTip("Delete this plot")
        self._show_hidden_views_cb = QCheckBox("Show hidden views")

        layout = QGridLayout()
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.addWidget(label, 1, 1)
        layout.addWidget(self.view_picker, 1, 2)
        layout.addWidget(self.delete_btn, 1, 3)
        layout.addWidget(self._show_hidden_views_cb, 2, 1)
        layout.setColumnStretch(4, 1)
        layout.setRowStretch(3, 1)

        view_selection_widget = QWidget()
        view_selection_widget.setLayout(layout)

        # Create plot widgets
        self._image_view = ImageViewF(has_roi=True, hide_axis=False)
        for roi in self._image_view.rois:
            roi.setLocked(False)
        # Note: this QHBoxLayout is currently unused but should be kept, it's a
        # placeholder for future widgets.
        image_widget_layout = QHBoxLayout()
        image_widget_layout.addWidget(self._image_view)
        self._image_widget = QWidget()
        self._image_widget.setLayout(image_widget_layout)

        self._plot_widget = PlotWidgetF()
        self._legend = self._plot_widget.addLegend(offset=(-30, -30))
        self._plot_widget.showLegend()

        # Create menu items
        self._back_action = QAction("Back")
        self._reset_action = QAction("Reset")
        self._split_above_action = QAction("Above")
        self._split_below_action = QAction("Below")
        self._split_left_action = QAction("Left")
        self._split_right_action = QAction("Right")
        split_menu = QMenu("Split frame")
        split_menu.addAction(self._split_above_action)
        split_menu.addAction(self._split_below_action)
        split_menu.addAction(self._split_left_action)
        split_menu.addAction(self._split_right_action)
        for action in [split_menu, self._back_action, self._reset_action]:
            self._image_view.addMenuItem(action)
            self._plot_widget.addMenuItem(action)

        self.addWidget(view_selection_widget)
        self.addWidget(self._image_widget)
        self.addWidget(self._plot_widget)

    def initConnections(self):
        self.view_picker.currentTextChanged.connect(self._onViewChanged)
        self._back_action.triggered.connect(self._onBackAction)
        self._reset_action.triggered.connect(self.reset)
        self.delete_btn.clicked.connect(self.delete)
        self._show_hidden_views_cb.stateChanged.connect(lambda: self.updateAvailableViews())

        split = lambda direction: self.split_sgn.emit(self, direction)
        self._split_above_action.triggered.connect(lambda: split(SplitDirection.ABOVE))
        self._split_below_action.triggered.connect(lambda: split(SplitDirection.BELOW))
        self._split_left_action.triggered.connect(lambda: split(SplitDirection.LEFT))
        self._split_right_action.triggered.connect(lambda: split(SplitDirection.RIGHT))

    def updateImage(self, **kwargs):
        """
        This is a wrapper function for _SpecialAnalysisBase to work with.
        Specifically, this gets called for auto-leveling images.
        """
        self._image_view.updateImage(**kwargs)

    @pyqtSlot()
    def delete(self):
        self.hide()
        self._main_window.unregisterPlotWidget(self)
        self.deleteLater()

    def _clearData(self):
        self._xs.reset()

        for label in list(self._ys.keys()):
            self._deletePlot(label, self._ys, self._plots)
        for label in list(self._errors.keys()):
            self._deletePlot(label, self._errors, self._error_plots)

        # Resetting the color list makes the colors more consistent when views
        # are changed or reset.
        self._next_color = itertools.cycle(["k", "n", "r", "o", "y", "c", "b",
                                            "s", "g", "p", "d", "w", "i"])

    @pyqtSlot(bool)
    def _onBackAction(self, _):
        self.setCurrentIndex(0)
        self.view_picker.setCurrentIndex(0)
        self._current_view = None
        self.reset()

    def _currentPlotWidget(self):
        if self._views[self._current_view].output == ViewOutput.IMAGE:
            return self._image_view
        else:
            return self._plot_widget

    @pyqtSlot(str)
    def _onViewChanged(self, view_name):
        # If we are back at the view selection widget, do nothing
        if view_name == "":
            return
        else:
            self._current_view = view_name

        view = self._views[view_name]
        if view.output == ViewOutput.IMAGE:
            self.setCurrentWidget(self._image_widget)
        else:
            if view.output in [ViewOutput.SCALAR, ViewOutput.POINTS, ViewOutput.COMPUTE]:
                plot_factory = self._scatter_plot_factory
            elif view.output == ViewOutput.VECTOR:
                plot_factory = self._curve_plot_factory
            else:
                logger.error(f"Unsupported view type: {view.output}")

            self._plots = defaultdict(plot_factory)
            self.setCurrentWidget(self._plot_widget)

        self._currentPlotWidget().setTitle(view_name)

    def _scatter_plot_factory(self):
        color = next(self._next_color)
        return self._plot_widget.plotScatter(pen=FColor.mkPen(color),
                                             brush=FColor.mkBrush(color),
                                             size=7)

    def _curve_plot_factory(self):
        return self._plot_widget.plotCurve(pen=FColor.mkPen(next(self._next_color)))

    def _deletePlot(self, label, data_dict, plot_dict):
        del data_dict[label]
        self._plot_widget.removeItem(plot_dict[label])
        del plot_dict[label]

    @pyqtSlot(bool)
    def reset(self, _=None):
        self._clearData()
        self._plot_widget.reset()
        self._image_view.reset()
        self._max_points = 5000
        self._resize_buffers()

    def _resize_buffers(self):
        for buf in [*self._ys.values(), *self._errors.values(), self._xs]:
            if buf.capacity() != self._max_points:
                buf.resize(self._max_points)

    def downsample(self, x, y):
        return lttbc.downsample(x, y, 2000)

    def updateF(self, all_data):
        if self._current_view not in all_data:
            return

        data_list = all_data[self._current_view]
        for data in data_list:
            is_xarray = isinstance(data, xr.DataArray)
            is_ndarray = isinstance(data, np.ndarray)
            view_type = self._views[self._current_view].output

            def handle_rich_output():
                series_errors = data.attrs["series_errors"]
                y_series_labels = data.attrs["y_series_labels"]

                if len(data) == 1:
                    y_series_labels.append("y0")

                # Resize buffers if necessary
                if max_points := data.attrs.get("max_points"):
                    self._max_points = max_points
                    self._resize_buffers()

                # Append new scalar data
                if data.ndim == 1:
                    if len(data) == 1:
                        label = y_series_labels[0]
                        self._xs.append(len(self._xs))
                        self._ys[label].append(data.values[0])
                    else:
                        for i, value in enumerate(data.values[1:]):
                            label = y_series_labels[i]

                            # If the label is new, initialize it with NaNs for the
                            # previous elements so the label data is the right size to
                            # match self._xs.
                            if label not in self._ys and len(self._xs) > 0:
                                self._ys[label].extend(np.full((len(self._xs), ), np.nan))
                            self._ys[label].append(value)

                            if label in series_errors:
                                if label not in self._errors and len(self._xs) > 0:
                                    self._errors[label].extend(np.full((len(self._xs), ), np.nan))
                                self._errors[label].append(series_errors[label])

                        self._xs.append(data.values[0])

                # If we're dealing with Vector data
                elif data.ndim == 2:
                    # If only a single vector has been passed, then we assume
                    # it's the Y axis data and generate the X axis.
                    if data.values.shape[0] == 1:
                        y_data = data.values[0]
                        x_data = np.arange(len(y_data))
                        x_data, y_data = self.downsample(x_data, y_data)
                        self._xs.extend(x_data)
                        self._ys[y_series_labels[0]].extend(y_data)

                    # Otherwise, we treat the first vector slice as the X axis,
                    # and the rest as Y axis series.
                    else:
                        self._xs.extend(data.values[0])

                        for i, vector in enumerate(data.values[1:]):
                            label = y_series_labels[i]
                            self._ys[label].extend(vector)

                            if label in series_errors:
                                self._errors[label].extend(series_errors[label])
                else:
                    logger.error(f"Cannot handle data with dimension: {data.ndim}")
                    return

                # Remove old plots
                for label in list(self._ys.keys()):
                    if label not in y_series_labels:
                        self._deletePlot(label, self._ys, self._plots)

                        if label in self._errors:
                            self._deletePlot(label, self._errors, self._error_plots)

                    if label in self._errors and label not in series_errors:
                        self._deletePlot(label, self._errors, self._error_plots)

            # Update stored data
            if view_type == ViewOutput.SCALAR:
                if np.isscalar(data):
                    self._xs.append(len(self._xs))
                    self._ys["y0"].append(data)
                elif is_xarray:
                    handle_rich_output()
                else:
                    logger.error(f"Cannot handle Scalar data of type: {type(data)}")
                    return
            elif view_type == ViewOutput.VECTOR:
                self._clearData()

                if is_ndarray:
                    x_data = np.arange(len(data))
                    x_data, y_data = self.downsample(x_data, data)

                    self._xs.extend(x_data)
                    self._ys["y0"].extend(y_data)
                elif is_xarray:
                    handle_rich_output()
                else:
                    logger.error(f"Cannot handle Vector data of type: {type(data)}")
                    return
            elif view_type == ViewOutput.POINTS:
                self._xs.append(data[0][0])
                self._ys["y0"].append(data[0][1])
            elif view_type == ViewOutput.COMPUTE:
                if is_xarray:
                    handle_rich_output()
                else:
                    logger.error(f"Only rich_output() is supported for View.Compute, cannot handle: {type(data)}")
                    return

            view = self._views[self._current_view]
            is_image = view_type == ViewOutput.IMAGE

            # Check if there are annotations on this view
            if view.annotations is not None:
                for idx, roi_name in enumerate(view.annotations):
                    if roi_name not in self._annotations:
                        roi = self._image_view.rois[idx + 1] if is_image else LinearROI(self._plot_widget)
                        if not is_image:
                            self._plot_widget.addItem(roi)

                        self._annotations[roi_name] = roi

                        # Update parameters from context
                        metro_roi = self._main_window.context.parameters[roi_name]
                        roi.configureFromMetroROI(metro_roi)

                        roi.setLabel(roi_name)
                        roi.sigRegionChanged.connect(self._main_window.onRoiChanged)
                        roi.show()

            # Hide unused annotations
            for roi_name in list(self._annotations.keys()):
                if roi_name not in view.annotations:
                    roi = self._annotations[roi_name]
                    roi.sigRegionChanged.disconnect()
                    roi.hide()
                    del self._annotations[roi_name]

                    if not is_image:
                        self._plot_widget.removeItem(roi)
                        roi.deleteLater()

            # Update the plots
            if view_type == ViewOutput.IMAGE:
                self._image_view.setAspectLocked(not view.aspect_unlocked)

                if not is_ndarray:
                    logger.error(f"Cannot handle Image data of type: {type(data)}")
                    return
                elif data.ndim != 2:
                    logger.error(f"Image data has wrong number of dimensions: {data.ndim} (expected 2)")
                    return

                self._image_view.setImage(data)
            else:
                self._legend.clear()
                for label, ys_data in self._ys.items():
                    self._legend.addItem(self._plots[label], label)
                    self._plots[label].setData(self._xs.data(), ys_data.data())

                    if label in self._errors:
                        error_data = self._errors[label].data()

                        if label not in self._error_plots:
                            self._error_plots[label] = self._plot_widget.plotStatisticsBar(pen=self._plots[label]._pen, beam=1)

                        self._error_plots[label].setData(self._xs.data(), ys_data,
                                                         y_min=ys_data - error_data, y_max=ys_data + error_data)

                if is_xarray:
                    if "xlabel" in data.attrs:
                        self._plot_widget.setLabel("bottom", data.attrs["xlabel"])
                    if "ylabel" in data.attrs:
                        self._plot_widget.setLabel("left", data.attrs["ylabel"])
                    if "title" in data.attrs:
                        self._plot_widget.setTitle(data.attrs["title"])

    def setViews(self, views):
        # Always update the view indexes
        old_views = self._views
        self._views = views

        # But we only update the options displayed if any views have been
        # added/removed.
        if self._views.keys() != old_views.keys():
            self.updateAvailableViews()
            self.views_updated_sgn.emit()

    def updateAvailableViews(self):
        self.view_picker.blockSignals(True)
        self.view_picker.clear()
        self.view_picker.addItem("")

        for view in sort_views(self._views.keys()):
            name = view.split("#", maxsplit=1)[1]
            is_hidden_view = name.startswith("_")

            if not self.show_hidden_views and is_hidden_view:
                    continue

            self.view_picker.addItem(view)

            if self.show_hidden_views and is_hidden_view:
                self.view_picker.setItemData(self.view_picker.count() - 1,
                                             QBrush(Qt.darkGray), Qt.ForegroundRole)


        if self._current_view not in self._views:
            self._onBackAction(None)
        else:
            self.view_picker.setCurrentText(self._current_view)

        self.view_picker.setEnabled(len(self._views) != 0)
        self.view_picker.blockSignals(False)

    @property
    def show_hidden_views(self):
        return self._show_hidden_views_cb.checkState() == Qt.Checked


class PathAutocompleter(QsciAbstractAPIs):
    """
    This class provides autocompletion for extra-metro paths.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._paths = defaultdict(list)

        # Populate the list of possible extra-foam paths. We need to create an
        # instance of ProcessedData because the variables (and hence their
        # types) are only initialized upon creation.
        model = ProcessedData(1)

        # Helper function to recursively add all slots from the model to the
        # autocomplete list.
        for slot, _ in traverse_slots(model):
            self._paths["foam"].append(slot)

    def updateAutoCompletionList(self, context, completions):
        """
        Overload of inherited method, this returns the completion list for a
        context.
        """
        path_type = context[0]

        if path_type in self._paths and len(context) > 1:
            return [path for path in self._paths[path_type]
                    if path.lower().startswith(context[1].lower())]
        else:
            return []

    def callTips(*_):
        """
        Overload of inherited method. We don't use it, but Qt will complain if
        it isn't implemented.
        """
        return []

    def setViewPaths(self, views):
        """
        Update the view paths.
        """
        self._paths["view"] = [view.split("#")[1] for view in views.keys()]

    def setDataPaths(self, path_data):
        """
        Update the data paths ('internal', 'karabo', etc). Note that this is
        done separately from the view paths because we get the data paths from a
        different source, the worker thread. i.e. the correlator needs to be
        running for these paths to be known, whereas the view paths may be
        obtained at any time from extra-metro.
        """
        # We ignore the foam paths because they have already been set during
        # initialization (and they never change during runtime anyway).
        paths = [path for path in path_data.keys() if not path.startswith("foam#")]

        path_types = set(path.split("#")[0] for path in paths)
        for path_type in path_types:
            self._paths[path_type] = [path.split("#")[1] for path in paths
                                      if path.startswith(f"{path_type}#")]


class PathLexer(QsciLexerPython):
    def autoCompletionWordSeparators(self):
        """
        This class is only subclassed so we can override this function, which
        makes the lexer treat tokens separated by hashes as valid for
        autocompletion. By default it only supports periods.
        """
        return ["#"]


class CorrelatorCtrlWidget(_BaseAnalysisCtrlWidgetS):
    open_file_sgn = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.open_btn = QPushButton("Open context")
        self.save_btn = QPushButton("Save context")
        self.reload_btn = QPushButton("Reload context")
        self._path_label = QLabel("")

        menu = QMenu()
        menu.addSeparator()
        self.open_btn.setMenu(menu)

        self.setRecentFiles([])

        self._paths_treewidget = QTreeWidget()
        self._paths_treewidget.setSelectionMode(QAbstractItemView.NoSelection)
        self._paths_treewidget.setColumnCount(4)
        self._paths_treewidget.setColumnHidden(3, True)
        self._paths_treewidget.setHeaderLabels(["Data paths", "Type", "Shape", "Path"])

        for path_type in ["internal", "karabo", "view", "foam"]:
            item = QTreeWidgetItem([path_type, "", "", path_type])
            self._paths_treewidget.addTopLevelItem(item)

        self.layout().addRow(self.open_btn, self._path_label)
        self.layout().addRow(self.save_btn)
        self.layout().addRow(self.reload_btn)
        self.layout().addRow(self._paths_treewidget)

    def onOpenFile(self):
        path = QFileDialog.getOpenFileName(self, "Open context", filter="Python (*.py)")[0]
        if len(path) > 0:
            self.open_file_sgn.emit(path)

    def setRecentFiles(self, recent_files):
        menu = self.open_btn.menu()
        menu.clear()

        for f in recent_files:
            action = menu.addAction(f)
            action.triggered.connect(lambda _, f=f: self.open_file_sgn.emit(f))

        menu.addSeparator()
        open_action = menu.addAction("Open other file...")
        open_action.triggered.connect(self.onOpenFile)

    @pyqtSlot(dict)
    def setDataPaths(self, paths):
        for path in sorted(paths.keys()):
            path_data = paths[path]

            is_array = path_data.type in [np.ndarray.__name__, xr.DataArray.__name__]
            path_type, item_name = path.split("#")
            item_type = f"{path_data.type}<{path_data.dtype}>" if is_array else path_data.type
            item_shape = "" if path_data.shape is None else str(path_data.shape)

            if item := self._findPathItem(path):
                if item.data(0, Qt.UserRole) != path_data:
                    item.setText(1, item_type)
                    item.setText(2, item_shape)
                    item.setData(0, Qt.UserRole, path_data)

                continue

            if path_type == "foam" and "." in path:
                parent_path = ".".join(path.split(".")[:-1])
                item_name = path.split(".")[-1]
            else:
                parent_path = path_type

            parent = self._findPathItem(parent_path)
            child = self._createPathItem(path, path_data,
                                         item_name=item_name,
                                         item_type=item_type,
                                         item_shape=item_shape)
            parent.addChild(child)

        # For paths set from data, Karabo paths are the only ones who might
        # disappear and need to be GC'd.
        self._gcPaths(self._findPathItem("karabo"), list(paths.keys()))

    @pyqtSlot(dict)
    def setViewPaths(self, views):
        parent = self._findPathItem("view")
        parent.takeChildren()

        for path in sort_views(views.keys()):
            if not self._findPathItem(path):
                item = self._createPathItem(path, views[path])
                parent.addChild(item)

    def _createPathItem(self, path, data, item_name=None, item_type="", item_shape=""):
        if item_name is None:
            item_name = path.split("#")[1]

        item = QTreeWidgetItem([item_name, item_type, item_shape, path])
        item.setData(0, Qt.UserRole, data)
        return item

    def _gcPaths(self, item, paths):
        old_items = []

        for i in range(item.childCount()):
            child = item.child(i)
            if child.text(3) not in paths:
                old_items.append(child)
            else:
                self._gcPaths(child, paths)

        for item in old_items:
            item.parent().removeChild(item)

    def _findPathItem(self, path):
        results = self._paths_treewidget.findItems(path, Qt.MatchExactly | Qt.MatchRecursive, 3)
        return results[0] if len(results) == 1 else None

    @property
    def context_path(self):
        return self._path_label.text()

    @context_path.setter
    def context_path(self, path):
        self._path_label.setText(path)


class RoiTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cstmeta.ParentNodeProvider, )

    def __init__(self, roi_cls, roi_name, roi_args):
        self._roi_cls_name = roi_cls.__name__
        self._roi_name = roi_name
        self._roi_args = roi_args

    def gen_number_node(self, value):
        # If this is a numpy type, np.float64 etc, then we need to convert it to
        # a native Python ctype.
        if hasattr(value, "dtype"):
            value = value.item()

        if isinstance(value, int):
            node_cls = cst.Integer
        elif isinstance(value, float):
            node_cls = cst.Float
        else:
            raise RuntimeError(f"Unsupported numeric type: {type(value)}")

        number_node = node_cls(str(np.abs(value)))

        if value < 0:
            return cst.UnaryOperation(cst.Minus(), number_node)
        else:
            return number_node

    def visit_Call(self, call):
        """
        On the way down the CST, filter for the call to 'parameters()'.
        """
        return m.matches(call,
                         m.Call(
                             func=m.Name("parameters"),
                             args=[m.ZeroOrMore(),
                                   m.AtLeastN(n=1,
                                              matcher=m.Arg(
                                                  value=m.Call(func=m.Name(self._roi_cls_name))
                                              )),
                                   m.ZeroOrMore()
                                   ]
                         ))

    def leave_Call(self, original_node, updated_node):
        """
        On the way back up, modify the ROI calls.
        """
        if m.matches(updated_node.func, m.Name(self._roi_cls_name)):
            parent = self.get_metadata(cstmeta.ParentNodeProvider, updated_node)

            # If this object is the value for a keyword argument of the same
            # name, it's the one we're looking for and we can update it.
            if m.matches(parent, m.Arg(keyword=m.Name(self._roi_name))):
                # Create the new int nodes
                new_ints = [self.gen_number_node(x) for x in self._roi_args]

                # If the constructor already has all arguments, then we just
                # update each arguments value individually. This preserves any
                # existing formatting/comments.
                if len(updated_node.args) == len(self._roi_args):
                    new_args = []
                    for arg, x in zip(updated_node.args, new_ints):
                        new_args.append(arg.with_changes(value=x))

                    return updated_node.with_changes(args=new_args)
                else:
                    # But if it doesn't have all the arguments, replace the
                    # argument list entirely.
                    return updated_node.with_changes(args=[cst.Arg(x) for x in new_ints])

        return updated_node


class Settings(QSettings):
    RECENT_FILES = ("recent_files", { })

    _all_settings = [RECENT_FILES]

    def beginGroup(self, setting):
        if setting in self._all_settings:
            super().beginGroup(setting[0])
        else:
            super().beginGroup(setting)

    def value(self, setting, default=None):
        if setting in self._all_settings:
            if default is None:
                default = setting[1]

            return self.value(setting[0], default)
        else:
            return super().value(setting, default)

    def setValue(self, setting, value):
        if setting in self._all_settings:
            self.setValue(setting[0], value)
        else:
            super().setValue(setting, value)

    def recentFilePaths(self):
        recent_files = self.value(Settings.RECENT_FILES)
        return sorted(recent_files.keys(),
                      key=lambda p: recent_files[p]["last_opened"],
                      reverse=True)

    def updateRecentFile(self, path, key, value):
        recent_files = self.value(Settings.RECENT_FILES)
        recent_files[path][key] = value
        self.setValue(Settings.RECENT_FILES, recent_files)


@create_special(CorrelatorCtrlWidget, CorrelatorProcessor)
class CorrelatorWindow(_SpecialAnalysisBase):
    icon = "cam_view.png"
    _title = "Correlator"
    _long_title = "Correlator"
    _client_support = ClientType.BOTH

    _initial_context = """\
    import numpy as np
    from scipy.signal import gausspulse

    from extra_foam.utils import rich_output, Series as S


    # This view won't work unless the suite is reading from EXtra-foam
    @View.Image
    def train_image(masked_train: "foam#image.masked_mean"):
        return masked_train

    @View.Scalar
    def scalar(tid: "internal#train_id"):
        return rich_output(tid,
                           title="Scalar",
                           xlabel="Index",
                           ylabel="Train ID")

    # We don't actually use the train id in this view, but we add it as an
    # argument so that the view gets executed anyway. Note that View's with
    # names beginning with an underscore are hidden by default, check the 'Show
    # hidden views' checkbox in a viewer frame to see them.
    @View.Vector
    def _vector(_: "internal#train_id"):
        # Example copied from: 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gausspulse.html
        t = np.linspace(-1, 1, 2 * 100, endpoint=False)
        i, q, e = gausspulse(t, fc=5, retquad=True, retenv=True)

        # Add some noise to make it more interesting
        noise_gen = lambda scale: scale * (np.random.rand(len(t)) - 0.5)
        i += noise_gen(0.2)
        q += noise_gen(0.1)
        e += noise_gen(0.05)

        return rich_output(t, 
                           y1=S(i, name="Real component"),
                           y2=S(q, name="Imaginary component"),
                           y3=S(e, name="Envelope"),
                           title="Gaussian modulated sinusoid",
                           xlabel="Amplitude",
                           ylabel="Time")
    """

    def __init__(self, topic):
        super().__init__(topic, with_dark=False)

        self._path = None
        self._total_tab_count = 1
        self._context_saved = True
        self._roi_last_changed_time = time.time()

        self.settings = Settings(osp.join(ROOT_PATH, "correlator.ini"), QSettings.IniFormat)

        self.initUI()
        self.initConnections()
        self.startWorker()

        # Initialize the context
        self._ctrl_widget_st.reload_btn.clicked.emit()
        center_window(self)

        self._ctrl_widget_st.setRecentFiles(self.settings.recentFilePaths())

    def initUI(self):
        cw = self.centralWidget()

        self._tab_widget = QTabWidget()

        self._editor = QsciScintilla()
        lexer = PathLexer(self._editor)
        font = QFont("Monospace", pointSize=12)
        lexer.setDefaultFont(font)
        self._completer = PathAutocompleter(lexer)

        self._editor.setLexer(lexer)
        self._editor.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAPIs)
        self._editor.setAutoCompletionThreshold(2)
        self._editor.setText(textwrap.dedent(self._initial_context))
        self._editor.setIndentationsUseTabs(False)
        self._editor.setTabWidth(4)
        self._editor.setAutoIndent(True)
        self._editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self._editor.textChanged.connect(self._onContextModified)

        self._editor.setMarginWidth(0, "0000")
        self._editor.setMarginLineNumbers(0, True)
        self._editor.setMarginWidth(1, "0000")
        self._editor.setMarginType(1, QsciScintilla.MarginType.SymbolMargin)

        self._tab_widget.addTab(self._editor, "Context")

        # Add fake tab to add create new tabs
        self._new_tab_btn = QToolButton()
        self._new_tab_btn.setText("+")
        self._tab_widget.addTab(QWidget(), "")
        self._tab_widget.tabBar().setTabButton(1, QTabBar.RightSide, self._new_tab_btn)
        self._tab_widget.setTabEnabled(1, False)

        # Create an initial tab
        self._addTab()
        self._tab_widget.setCurrentIndex(0)

        cw.addWidget(self._tab_widget)
        cw.setSizes([int(self._TOTAL_W / 4), int(3 * self._TOTAL_W / 4)])

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self._error_marker_nr = 0
        self._editor.markerDefine(QsciScintilla.MarkerSymbol.Circle, self._error_marker_nr)
        self._editor.setMarkerForegroundColor(FColor.mkColor("r"), self._error_marker_nr)
        self._editor.setMarkerBackgroundColor(FColor.mkColor("r"), self._error_marker_nr)

    def initConnections(self):
        ctrl = self._ctrl_widget_st
        worker = self._worker_st

        self._new_tab_btn.clicked.connect(self._addTab)

        ctrl.reload_btn.clicked.connect(
            lambda: self._worker_st.setContext(self._editor.text())
        )
        ctrl.save_btn.clicked.connect(self._saveContext)
        ctrl.open_file_sgn.connect(self._openContext)

        self._com_ctrl_st.hostname_changed_sgn.connect(self._onHostnameChanged)
        self._com_ctrl_st.port_changed_sgn.connect(self._onPortChanged)
        self._com_ctrl_st.client_type_changed_sgn.connect(self._onClientTypeChanged)

        worker.reloaded_sgn.connect(self._clearMarkers)
        worker.context_error_sgn.connect(self._onContextError)
        worker.updated_views_sgn.connect(self._onViewsUpdated)
        worker.updated_data_paths_sgn.connect(self._completer.setDataPaths)
        worker.updated_data_paths_sgn.connect(self._ctrl_widget_st.setDataPaths)

    def _onContextError(self, error):
        self._clearMarkers()

        # Find the error line number in the context file
        lineno = -1
        if isinstance(error, cst.ParserSyntaxError):
            lineno = error.raw_line
        elif isinstance(error, mpc_error.PathResolutionError):
            lineno = error.view.kernel.__code__.co_firstlineno
        elif isinstance(error, mpc_error.ViewDefError):
            lineno = error._find_context_frame("<ctx>")
        elif isinstance(error, mpc_error.ContextSyntaxError):
            lineno = error.orig_error.lineno
        elif isinstance(error, Exception):
            tb = sys.exc_info()[2]
            for frame in traceback.extract_tb(tb):
                if frame.filename == "<ctx>":
                    lineno = frame.lineno
                    break

        if lineno == -1:
            logger.error(f"Couldn't get line number from {type(error)} for display")
            return

        self._editor.markerAdd(lineno - 1, self._error_marker_nr)

    def _clearMarkers(self):
        self._editor.markerDeleteAll()

    @property
    def context(self):
        return self._worker_st._ctx

    def _onHostnameChanged(self, hostname):
        if self._path is not None:
            self.settings.updateRecentFile(self._path, "hostname", hostname)

    def _onPortChanged(self, port):
        if self._path is not None:
            self.settings.updateRecentFile(self._path, "port", port)

    def _onClientTypeChanged(self, client_type):
        if self._path is not None:
            self.settings.updateRecentFile(self._path, "client_type", client_type.value)

    @pyqtSlot()
    def _onContextModified(self):
        self._markContextSaved(False)

    @pyqtSlot(dict)
    def _onViewsUpdated(self, views):
        for widget in self._plot_widgets_st:
            widget.setViews(views)

        self._completer.setViewPaths(views)
        self._ctrl_widget_st.setViewPaths(views)

    def onRoiChanged(self, roi):
        if time.time() - self._roi_last_changed_time < 0.1:
            return

        # Get current source
        ctx = self._editor.text()
        cursor_pos = self._editor.getCursorPosition()
        first_visible_line = self._editor.firstVisibleLine()

        roi_name = roi.label()
        if isinstance(roi, RectROI):
            roi_cls = MetroRectROI
            roi_args = [int(x) for x in [*roi.pos(), *roi.size()]]
        elif isinstance(roi, LinearROI):
            roi_cls = MetroLinearROI
            roi_args = [np.around(x, 5) for x in roi.getRegion()]
        else:
            raise RuntimeError("Unsupported ROI type")

        # Modify the source as needed
        try:
            module = cstmeta.MetadataWrapper(cst.parse_module(ctx))
        except cst.ParserSyntaxError as e:
            logger.error(str(e))

            marker_exists = self._editor.markerFindNext(e.raw_line - 1, 1) == e.raw_line - 1
            if not marker_exists:
                self._onContextError(e)
                active_window = QApplication.activeWindow()
                QMessageBox.warning(active_window,
                                    "Syntax error in context",
                                    "The context file has a syntax error, updating the parameters in the context file will not work until it is fixed.")

            return
        else:
            self._clearMarkers()

        transformer = RoiTransformer(roi_cls, roi_name, roi_args)
        new_source = module.visit(transformer)

        # Set the new source and update the parameter
        self._editor.setText(new_source.code)
        self._editor.setCursorPosition(*cursor_pos)
        self._editor.setFirstVisibleLine(first_visible_line)

        self._worker_st.set_parameter(roi.label(),
                                      roi_cls(*roi_args))
        self._roi_last_changed_time = time.time()

    @pyqtSlot()
    def _addTab(self, splitter=None):
        index = self._tab_widget.count() - 1

        if splitter is None:
            splitter = UberSplitter(self)
            splitter.setWindowTitle(f"Tab {self._total_tab_count}")
            self._total_tab_count += 1

        # We set the names of the buttons to find them later in the tests
        undock_btn = QToolButton()
        undock_btn.setObjectName("undock_btn")
        undock_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        undock_btn.clicked.connect(lambda: self._undockTab(splitter))
        close_btn = QToolButton()
        close_btn.setObjectName("close_btn")
        close_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        close_btn.clicked.connect(lambda: self._removeTab(splitter))
        buttons_hbox = QHBoxLayout()
        buttons_hbox.addWidget(undock_btn)
        buttons_hbox.addWidget(close_btn)
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_hbox)

        # This is a filthy hack to get around some weirdness of PyQt. In a
        # nutshell, it seems that the type information of undock_btn and
        # close_btn are somehow lost after this function returns. Printing
        # `buttons_widget.children()` in this function prints a list of
        # [QHBoxLayout, QToolButton, QToolButton] as expected, but outside of
        # this function we instead get [QObject, QObject, QObject]. This is bad
        # because it makes testing later rather difficult when we try to find
        # the buttons based on their objectName's with QObject.findChild() (see
        # the testTabs() tests), because they are only accessible as
        # QObject's. For some reason this causes manually calling their signals
        # with .emit() to fail, and QTest.mouseClick() won't work because it
        # requires QWidget's and not QObject's (and there's no way to cast
        # them). But, assigning them to some variable that exists outside the
        # scope of this function magically preserves their type information,
        # which is what the two variables below are for. One might suspect that
        # it has something to do with reference counting... Tested on Python 3.9
        # and 3.10.
        self.__close_btn = close_btn
        self.__undock_btn = undock_btn

        self._tab_widget.insertTab(index, splitter, splitter.windowTitle())
        self._tab_widget.tabBar().setTabButton(index, QTabBar.RightSide, buttons_widget)
        self._tab_widget.setCurrentIndex(index)

    @pyqtSlot(object)
    def _removeTab(self, splitter):
        index = self._tab_widget.indexOf(splitter)
        self._tab_widget.removeTab(index)
        splitter.delete()

        self._ensureValidTab(index)

    @pyqtSlot()
    def _saveContext(self):
        path = self._ctrl_widget_st.context_path
        first_time_saved = len(path) == 0
        if first_time_saved:
            path = QFileDialog.getSaveFileName(self, "Save context", filter="Python (*.py)")[0]

        if len(path) > 0:
            if not path.endswith(".py"):
                path += ".py"

            with open(path, "w") as context_file:
                context_file.write(self._editor.text())
                context_file.flush()

            self._ctrl_widget_st.context_path = path
            self._markContextSaved(True)
            logger.info("Context saved")

            if first_time_saved:
                self._addToRecentFiles(path)

    @pyqtSlot(str)
    def _openContext(self, path):
        with open(path) as context_file:
            self._editor.setText(context_file.read())

        logger.info("Context opened")
        self._ctrl_widget_st.context_path = path
        self._ctrl_widget_st.reload_btn.clicked.emit()
        self._markContextSaved(True)

        self._addToRecentFiles(path)

        # Restore settings
        recent_files = self.settings.value(Settings.RECENT_FILES)
        self.setClient(ClientType(recent_files[path]["client_type"]))
        self.setHostname(recent_files[path]["hostname"])
        self.setPort(recent_files[path]["port"])

    def _addToRecentFiles(self, path):
        com_ctrl = self._com_ctrl_st

        recent_files = self.settings.value(Settings.RECENT_FILES)
        if path not in recent_files:
            recent_files[path] = {
                "client_type": com_ctrl.selected_client.value,
                "hostname": com_ctrl.hostname,
                "port": com_ctrl.port
            }

        # Update last opened time
        recent_files[path]["last_opened"] = int(time.time())

        # Prune old paths
        self.settings.setValue(Settings.RECENT_FILES, recent_files)
        ordered_paths = self.settings.recentFilePaths()
        for p in ordered_paths[10:]:
            del recent_files[p]
        self.settings.setValue(Settings.RECENT_FILES, recent_files)

        # Update GUI
        self._ctrl_widget_st.setRecentFiles(self.settings.recentFilePaths())

        self._path = path

    def _markContextSaved(self, saved):
        self._context_saved = saved

        color = "k" if saved else "r"
        self._tab_widget.tabBar().setTabTextColor(0, FColor.mkColor(color))

    @pyqtSlot()
    def _undockTab(self, splitter):
        index = self._tab_widget.indexOf(splitter)
        self._tab_widget.removeTab(index)
        splitter.undock()
        self._ensureValidTab(index)

    def _ensureValidTab(self, index):
        # If we removed the last tab, move focus back to the previous one so
        # that the fake-tab-creator tab is not selected.
        if index == self._tab_widget.count() - 1:
            self._tab_widget.setCurrentIndex(index - 1)

    def dock(self, splitter):
        self._addTab(splitter)

    def closeEvent(self, event):
        if not self._context_saved:
            dialog = QMessageBox(QMessageBox.Warning,
                                 "Warning - unsaved changes",
                                 "There are unsaved changes to the context, do you want to save before exiting?",
                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            result = dialog.exec()

            if result == QMessageBox.Yes:
                self._saveContext()
            elif result == QMessageBox.Cancel:
                event.ignore()
                return

        self.settings.sync()
        self._worker_st.close()
        super().closeEvent(event)
