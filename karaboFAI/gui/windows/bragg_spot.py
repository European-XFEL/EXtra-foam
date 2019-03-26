"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

BraggSpotsWindow.

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from collections import deque, OrderedDict

import numpy as np
from scipy import ndimage

from ..pyqtgraph import (
    GraphicsLayoutWidget, ImageItem, LineSegmentROI, mkBrush, mkPen, QtCore,
    QtGui, RectROI, ScatterPlotItem
)
from ..pyqtgraph import parametertree as ptree
from ..pyqtgraph.dockarea import DockArea, Dock

from .base_window import AbstractWindow
from ...logger import logger
from ..misc_widgets import lookupTableFactory, make_pen
from ...config import config, OpLaserMode


# class PlotWindow(AbstractWindow):
#     """QMainWindow consists of a GraphicsLayoutWidget and a ParameterTree.
#
#     DEPRECATED!
#     """
#     available_modes = OrderedDict({
#         OpLaserMode.: "Laser-on/off pulses in the same train",
#         OpLaserMode.EVEN_ON: "Laser-on/off pulses in even/odd train",
#         OpLaserMode.ODD_ON: "Laser-on/off pulses in odd/even train"
#     })
#
#     def __init__(self, *args, **kwargs):
#         """Initialization."""
#         super().__init__(*args, **kwargs)
#
#         self._gl_widget = GraphicsLayoutWidget()
#         self._ctrl_widget = None
#
#         self._plot_items = []  # bookkeeping PlotItem objects
#         self._image_items = []  # bookkeeping ImageItem objects
#
#         # -------------------------------------------------------------
#         # define parameter tree
#         # -------------------------------------------------------------
#
#         self._ptree = ptree.ParameterTree(showHeader=True)
#
#         # parameters groups
#         self._exp_params = ptree.Parameter.create(
#             name='Experimental setups', type='group')
#         self._pro_params = ptree.Parameter.create(
#             name='Data processing parameters', type='group')
#         self._vis_params = ptree.Parameter.create(
#             name='Visualization options', type='group')
#         self._act_params = ptree.Parameter.create(
#             name='Actions', type='group')
#         self._ana_params = ptree.Parameter.create(
#             name='Analysis options', type='group')
#         self._ins_params = ptree.Parameter.create(
#             name='Instruction', type='group')
#
#         # -------------------------------------------------------------
#         # available Parameters (shared parameters and actions)
#         #
#         # shared parameters are updated by signal-slot
#         # Note:
#         # 1. shared parameters should end with '_sp'
#         # 2. we have all the slots and shared parameters here while only
#         #    connect the signal to slot in the concrete class.
#         # -------------------------------------------------------------
#
#         self.mask_range_sp = None
#         self.mask_range_param = ptree.Parameter.create(
#             name='Mask range', type='str', readonly=True
#         )
#
#         self.laser_mode_sp = None
#         self.optical_laser_mode_param = ptree.Parameter.create(
#             name='Optical laser mode', type='str', readonly=True
#         )
#
#         self.on_pulse_ids_sp = None
#         self.laser_on_pulse_ids_param = ptree.Parameter.create(
#             name='Laser-on pulse ID(s)', type='str', readonly=True
#         )
#
#         self.off_pulse_ids_sp = None
#         self.laser_off_pulse_ids_param = ptree.Parameter.create(
#             name='Laser-off pulse ID(s)', type='str', readonly=True
#         )
#
#         self.normalization_range_sp = None
#         self.normalization_range_param = ptree.Parameter.create(
#             name="Normalization range", type='str', readonly=True
#         )
#
#         self.integration_range_sp = None
#         self.integration_range_param = ptree.Parameter.create(
#             name="Diff integration range", type='str', readonly=True
#         )
#
#         self.moving_avg_window_sp = None
#         self.moving_avg_window_param = ptree.Parameter.create(
#             name='M.A. window size', type='int', readonly=True
#         )
#
#         self.reset_action_param = ptree.Parameter.create(
#             name='Clear history', type='action'
#         )
#         self.reset_action_param.sigActivated.connect(self._reset)
#
#         # this method inject parameters into the parameter tree
#         self.updateParameterTree()
#
#     def initUI(self):
#         """Override."""
#         self.initCtrlUI()
#         self.initPlotUI()
#
#         layout = QtGui.QHBoxLayout()
#         if self._ctrl_widget is not None:
#             layout.addWidget(self._ctrl_widget)
#         layout.addWidget(self._gl_widget)
#         self._cw.setLayout(layout)
#
#     def update(self):
#         """Update plots.
#
#         This method is called by the main GUI.
#         """
#         raise NotImplementedError
#
#     def clear(self):
#         """Clear plots.
#
#         This method is called by the main GUI.
#         """
#         for item in self._plot_items:
#             item.clear()
#         for item in self._image_items:
#             item.clear()
#
#     @QtCore.pyqtSlot(object, list, list)
#     def onOffPulseIdChanged(self, mode, on_pulse_ids, off_pulse_ids):
#         self.laser_mode_sp = mode
#         self.on_pulse_ids_sp = on_pulse_ids
#         self.off_pulse_ids_sp = off_pulse_ids
#
#     @QtCore.pyqtSlot(float, float)
#     def onMaskRangeChanged(self, lb, ub):
#         self.mask_range_sp = (lb, ub)
#
#         # then update the parameter tree
#         self._pro_params.child('Mask range').setValue(
#             '{}, {}'.format(lb, ub))
#
#     @QtCore.pyqtSlot(float, float)
#     def onNormalizationRangeChanged(self, lb, ub):
#         self.normalization_range_sp = (lb, ub)
#
#         # then update the parameter tree
#         self._pro_params.child('Normalization range').setValue(
#             '{}, {}'.format(lb, ub))
#
#     @QtCore.pyqtSlot(float, float)
#     def onDiffIntegrationRangeChanged(self, lb, ub):
#         self.integration_range_sp = (lb, ub)
#
#         # then update the parameter tree
#         self._pro_params.child("Diff integration range").setValue(
#             '{}, {}'.format(lb, ub))
#
#     @QtCore.pyqtSlot(int)
#     def onMAWindowSizeChanged(self, value):
#         self.moving_avg_window_sp = value
#
#         # then update the parameter tree
#         self._pro_params.child('M.A. window size').setValue(str(value))
#
#     def updateParameterTree(self):
#         """Update the parameter tree.
#
#         In this method, one should and only should have codes like
#
#         self._exp_params.addChildren(...)
#         self._pro_params.addChildren(...)
#         self._vis_params.addChildren(...)
#         self._act_params.addChildren(...)
#
#         params = ptree.Parameter.create(name='params', type='group',
#                                         children=[self._exp_params,
#                                                   self._pro_params,
#                                                   self._vis_params,
#                                                   self._act_params])
#
#         self._ptree.setParameters(params, showTop=False)
#
#         Here '...' is a list of Parameter instances or dictionaries which
#         can be used to instantiate Parameter instances.
#         """
#         pass
#
#     def _reset(self):
#         """Reset all internal states/histories."""
#         pass
#
#
# class BraggSpotsWindow(PlotWindow):
#     """ BraggSpotsClass: Compute moving average of the center of mass
#
#     This window is used to visualize the moving average of the position
#     of the centre of mass for the user selected region of interest.
#     User can drag the scalable ROI around the Bragg spot in the image
#     on top left corner of the window. A second ROI is also provided for
#     background subtraction. Two plots on top right corner shows the X
#     and Y coordinates of the centre of mass for the selected region as
#     a function of On and Off pulseIds provided by the users from
#     control panel in the main window. Botton left images are the zoomed-
#     in images of the selcted regions. Bottom right plot analyse the
#     pulsed averaged X and Y coordinates of centre of mass with trainIds.
#
#     There is also an option for profile analysis of the image.
#     By checking in the option of "Profile Analysis", use
#     can click anywhere in the image and two histograms will appear in
#     the bottom that provides profile of image along the horizontal and
#     vertical line segments passing through the position where mouse
#     click event happened.
#
#     Another option "Normalized Intensity Plot" when checked in, replaces
#     the Moving average plot on top right corner with the normalized
#     intensity plot of the region of interest.
#         I = \\sum (ROI_Bragg - ROI_background)/(\\sum ROI_background)
#
#     Author : Ebad Kamil
#     Email  : ebad.kamil@xfel.eu
#     """
#     instructions = (
#          "Green ROI: Place it around Bragg peak.\n\n"
#          "White ROI: Place it around Background.\n\n"
#          "Scale the Green ROI using handle on top right corner.\n\n"
#          "To analyse the profile of image check the Profile "
#          "analysis box and then click on the image on top-left corner.\n\n"
#          "Always Clear History when ROIs positions are changed or parameters "
#          "in the control panel in the main-gui are modified."
#          )
#
#     def __init__(self, data, *, parent=None):
#         """Initialization."""
#         super().__init__(data, parent=parent)
#
#         # -------------------------------------------------------------
#         # connect signal and slot
#         # -------------------------------------------------------------
#         self.parent().pump_probe_ctrl_widget.on_off_pulse_ids_sgn.connect(
#             self.onOffPulseIdChanged)
#         self.parent().pump_probe_ctrl_widget.moving_avg_window_sgn.connect(
#             self.onMAWindowSizeChanged)
#
#         # tell MainGUI to emit signals in order to update shared parameters
#         self.parent().updateSharedParameters()
#
#         self.setGeometry(100, 100, 1600, 1000)
#         self._docker_area = DockArea()
#         self._image_gl_widget = GraphicsLayoutWidget()
#         self._comxy_gl_widget = GraphicsLayoutWidget()
#         self._history_gl_widget = GraphicsLayoutWidget()
#         self._intensity_gl_widget = GraphicsLayoutWidget()
#
#         self._rois = []  # bookkeeping Region of interests.
#         self._hist_train_on_id = []
#         self._hist_train_off_id = []
#         self._hist_com_on = []
#         self._hist_com_off = []
#
#         self._profile_plot_items = []
#         self._profile_line_rois = []
#
#         self._on_train_received = False
#         self._off_train_received = False
#
#         self._drop_last_on_pulse = False
#
#         self._on_pulses_ma = None
#         self._off_pulses_ma = None
#
#         self._on_pulses_hist = deque()
#         self._off_pulses_hist = deque()
#
#         self.initUI()
#         self.update()
#         logger.info("Open COM Analysis Window")
#
#     def updateParameterTree(self):
#         """Override."""
#         self._exp_params.addChildren([
#             self.optical_laser_mode_param,
#             self.laser_on_pulse_ids_param,
#             self.laser_off_pulse_ids_param,
#         ])
#
#         self._pro_params.addChildren([
#             self.moving_avg_window_param,
#             self.mask_range_param
#
#         ])
#
#         self._ana_params.addChildren([
#            {'name': 'Profile Analysis', 'type': 'bool', 'value': False},
#         ])
#
#         self._act_params.addChildren([
#             self.reset_action_param
#         ])
#
#         self._ins_params.addChildren([
#            {'name': 'Instructions', 'type': 'text', 'readonly': True,
#             'value': self.instructions}
#         ])
#
#         params = ptree.Parameter.create(name='params', type='group',
#                                         children=[self._exp_params,
#                                                   self._pro_params,
#                                                   self._ana_params,
#                                                   self._act_params,
#                                                   self._ins_params])
#         # Profile check button needed to avoid clash while moving
#         # brad and background region of interests. Click based.
#         self._ana_params.child('Profile Analysis').sigStateChanged.connect(
#             self._profile)
#
#         self._ptree.setParameters(params, showTop=False)
#
#     def initUI(self):
#         """Override"""
#         self.initCtrlUI()
#         self.initPlotUI()
#         ptree_dock = Dock("Experimental parameters")
#         self._docker_area.addDock(ptree_dock, 'left')
#         ptree_dock.addWidget(self._ctrl_widget)
#
#         image_dock = Dock("")
#         self._docker_area.addDock(image_dock, 'right')
#         image_dock.addWidget(self._image_gl_widget)
#
#         intensity_dock = Dock("Normalized Intensity")
#         self._docker_area.addDock(intensity_dock, 'right', image_dock)
#         intensity_dock.addWidget(self._intensity_gl_widget)
#
#         comxy_dock = Dock("Moving average of centre of Mass")
#         self._docker_area.addDock(comxy_dock, 'above', intensity_dock)
#         comxy_dock.addWidget(self._comxy_gl_widget)
#
#         history_dock = Dock("Pulse averaged centre of mass")
#         self._docker_area.addDock(history_dock, 'bottom', comxy_dock)
#         history_dock.addWidget(self._history_gl_widget)
#
#         layout = QtGui.QHBoxLayout()
#         layout.addWidget(self._docker_area)
#         self._cw.setLayout(layout)
#
#     def initCtrlUI(self):
#         """Override"""
#         self._ctrl_widget = QtGui.QWidget()
#         layout = QtGui.QVBoxLayout()
#         layout.addWidget(self._ptree)
#         self._ctrl_widget.setLayout(layout)
#
#     def initPlotUI(self):
#         img = ImageItem(border='w')
#         img.setLookupTable(lookupTableFactory[config['COLOR_MAP']])
#         self._image_items.append(img)
#         self._main_vb = self._image_gl_widget.addPlot(
#             row=0, col=0, rowspan=2, colspan=2,
#             lockAspect=True, enableMouse=False)
#         self._main_vb.addItem(img)
#
#         data = self._data.get()
#         if data.empty():
#             # Define First Region of interests.Around Bragg Data
#             roi = RectROI([config['CENTER_X'], config['CENTER_Y']], [50, 50],
#                           pen=mkPen((0, 255, 0), width=2))
#             self._rois.append(roi)
#             # Define Second Region of interests.Around Background
#             roi = RectROI([config['CENTER_X']-100, config['CENTER_Y']-100],
#                           [50, 50], pen=mkPen((255, 255, 255), width=2))
#             self._rois.append(roi)
#         else:
#             img_width, img_height = data.image_mean.shape
#             # Define First Region of interests.Around Bragg Data
#             # Max Bounds for region of interest defined
#             roi = RectROI([int(img_width/2), int(img_height/2)], [50, 50],
#                           maxBounds=QtCore.QRectF(0, 0, img_height, img_width),
#                           pen=mkPen((0, 255, 0), width=2))
#             self._rois.append(roi)
#             # Define Second Region of interests.Around Background
#             roi = RectROI([int(img_width/2)-100, int(img_height/2)-100],
#                           [50, 50],
#                           maxBounds=QtCore.QRectF(0, 0, img_height, img_width),
#                           pen=mkPen((255, 255, 255), width=2))
#             self._rois.append(roi)
#
#         for roi in self._rois:
#             self._main_vb.addItem(roi)
#
#         for handle in self._rois[1].getHandles():
#             self._rois[1].removeHandle(handle)
#
#         # View Boxes vb1 and vb2 in lower left panels for images in
#         # selected ROIs
#         vb1 = self._image_gl_widget.addViewBox(row=2, col=0, rowspan=2, colspan=1,
#                                          lockAspect=True, enableMouse=False)
#         img1 = ImageItem()
#         img1.setLookupTable(lookupTableFactory[config['COLOR_MAP']])
#         vb1.addItem(img1)
#         self._image_items.append(img1)
#
#         vb2 = self._image_gl_widget.addViewBox(row=2, col=1, rowspan=2, colspan=1,
#                                          lockAspect=True, enableMouse=False)
#         img2 = ImageItem(border='w')
#         img2.setLookupTable(lookupTableFactory[config['COLOR_MAP']])
#         vb2.addItem(img2)
#         self._image_items.append(img2)
#
#         # Plot regions for COM moving averages and history over
#         # different trains
#         p = self._comxy_gl_widget.addPlot(
#             row=0, col=0, rowspan=1, colspan=1, lockAspect=True)
#         self._plot_items.append(p)
#         p.setLabel('left',  "<span style='text-decoration: overline'>R</span>\
#             <sub>x</sub>")
#
#         p = self._comxy_gl_widget.addPlot(
#             row=1, col=0, rowspan=1, colspan=1, lockAspect=True)
#         self._plot_items.append(p)
#         p.setLabel('left', "<span style='text-decoration: overline'>R</span>\
#             <sub>y</sub>")
#         p.setLabel('bottom', "Pulse ids")
#
#         p = self._intensity_gl_widget.addPlot(
#                 row=0, col=2, rowspan=2, colspan=2, lockAspect=True)
#         self._plot_items.append(p)
#         p.setLabel('left', "Intensity")
#         p.setLabel('bottom', "Pulse ids")
#
#
#         p = self._history_gl_widget.addPlot(
#             row=0, col=0, rowspan=1, colspan=1, lockAspect=True)
#         self._plot_items.append(p)
#         p.setLabel('left', '&lt;<span style="text-decoration:\
#             overline">R</span><sub>x</sub>&gt;<sub>pulse-avg</sub>')
#         p.setTitle(' ')
#
#         p = self._history_gl_widget.addPlot(
#             row=1, col=0, rowspan=1, colspan=1, lockAspect=True)
#         self._plot_items.append(p)
#         p.setLabel('left',  "&lt;<span style='text-decoration:\
#             overline'>R</span><sub>y</sub>&gt;<sub>pulse-avg</sub>")
#         p.setLabel('bottom', "Train ID")
#         p.setTitle(' ')
#
#     def _update(self, data):
#         self.mask_range_sp = (0, 2500)
#
#         # Same logic as LaserOnOffWindow.
#         available_modes = list(self.available_modes.keys())
#         if self.laser_mode_sp == available_modes[0]:
#             self._on_train_received = True
#             self._off_train_received = True
#         else:
#
#             if self.laser_mode_sp == available_modes[1]:
#                 flag = 0
#             elif self.laser_mode_sp == available_modes[2]:
#                 flag = 1
#             else:
#                 raise ValueError("Unknown laser mode!")
#
#             if self._on_train_received:
#                 if data.tid % 2 == 1 ^ flag:
#                     self._off_train_received = True
#                 else:
#                     self._drop_last_on_pulse = True
#             else:
#                 if data.tid % 2 == flag:
#                     self._on_train_received = True
#
#         # slices dictionary is used to store array region selected by
#         # two ROIs around Bragg data and background
#         slices = OrderedDict([('bragg_data', None), ('background_data', None)])
#
#         com_on = None
#         com_off = None
#         if self._on_train_received:
#
#             if self.laser_mode_sp == available_modes[0] or \
#                     not self._off_train_received:
#
#                 this_on_pulses = []
#                 # Collects centre of mass for each pulse in
#                 # this_on_pulses list
#                 for pid in self.on_pulse_ids_sp:
#                     if pid >= data.images.shape[0]:
#                         logger.error("Pulse ID {} out of range (0 - {})!".
#                                      format(pid, data.images.shape[0] - 1))
#                         continue
#
#                     for index, key in enumerate(slices):
#                         # slices of regions selected by two ROIs.
#                         # One around Bragg spot and one around background
#                         # key : bragg_data stores array region around
#                         #       bragg spot ROI
#                         # key : background_data stores array region
#                         #       around background ROI
#
#                         slices[key] = self._rois[index].getArrayRegion(
#                             data.images[pid], self._image_items[0])
#                         # convert nan to -inf so that we can clip
#                         # negatives and values above mask range.
#                         # np.clip will not clip NaNs.
#                         (slices[key])[np.isnan(slices[key])] = - np.inf
#                         np.clip(slices[key], self.mask_range_sp[0],
#                                 self.mask_range_sp[1], out=slices[key])
#                         # clip to restrict between mask values 0-2500
#
#                     # background subtraction from Bragg_data.
#                     # Resulting image to be used for COM evaluation.
#                     mass_from_data = (
#                         slices['bragg_data'] - slices['background_data'])
#                     np.clip(mass_from_data, self.mask_range_sp[0],
#                             self.mask_range_sp[1], out=mass_from_data)
#                     # normalization = \sum ROI_background
#                     # Ńormalized intensity:
#                     # \sum (ROI_bragg - ROI_background)/ normalization
#                     normalization = np.sum(slices['background_data'])
#                     intensity = np.sum(mass_from_data)/normalization
#                     # Centre of mass
#                     mass = ndimage.measurements.center_of_mass(mass_from_data)
#
#                     this_on_pulses.append(np.append(np.array(mass), intensity))
#
#                 this_on_pulses = np.array(this_on_pulses)
#                 # Same logic as LaserOnOffWindow. Running averages over
#                 # trains.
#                 if self._drop_last_on_pulse:
#                     length = len(self._on_pulses_hist)
#                     self._on_pulses_ma += (
#                         (this_on_pulses - self._on_pulses_hist.pop()) / length
#                         )
#                     self._drop_last_on_pulse = False
#                 else:
#                     if self._on_pulses_ma is None:
#                         self._on_pulses_ma = np.copy(this_on_pulses)
#                     elif len(self._on_pulses_hist) < self.moving_avg_window_sp:
#                         self._on_pulses_ma += \
#                             (this_on_pulses - self._on_pulses_ma) \
#                             / (len(self._on_pulses_hist) + 1)
#                     elif len(self._on_pulses_hist) == self.moving_avg_window_sp:
#                         self._on_pulses_ma += \
#                             (this_on_pulses - self._on_pulses_hist.popleft()) \
#                             / self.moving_avg_window_sp
#                     else:
#                         raise ValueError
#
#                 self._on_pulses_hist.append(this_on_pulses)
#
#             com_on = self._on_pulses_ma
#
#             # This part at the moment makes no physical sense.
#             # Atleast to me. To be discussed with Dmitry. I added it
#             # here for some kind of history book keeping
#             if com_on.size != 0:
#                 self._hist_train_on_id.append(data.tid)
#                 self._hist_com_on.append(np.mean(np.array(com_on), axis=0))
#
#         if self._off_train_received:
#
#             this_off_pulses = []
#             for pid in self.off_pulse_ids_sp:
#                 if pid > data.images.shape[0]-1:
#                     logger.error("Pulse ID {} out of range (0 - {})!".
#                                  format(pid, data.images.shape[0] - 1))
#                     continue
#
#                 for index, key in enumerate(slices):
#                     # slices of regions selected by two ROIs.
#                     # One around bragg spot and one around background
#                     # key : bragg_data stores array region around Bragg
#                     #       spot ROI
#                     # key : background stores array region around
#                     #       background ROI
#                     slices[key] = self._rois[index].getArrayRegion(
#                         data.images[pid], self._image_items[0])
#                     # convert nan to -inf so that we can clip
#                     # negatives and values above mask range.
#                     # np.clip will not clip NaNs.
#                     (slices[key])[np.isnan(slices[key])] = - np.inf
#
#                     np.clip(slices[key], self.mask_range_sp[0],
#                             self.mask_range_sp[1], out=slices[key])
#                     # clip to restrict between mask values 0-2500
#
#                 # background subtraction from Bragg_data. Resulting image
#                 # to be used for COM evaluation.
#                 mass_from_data = (
#                     slices['bragg_data'] - slices['background_data'])
#
#                 np.clip(mass_from_data, self.mask_range_sp[0],
#                         self.mask_range_sp[1], out=mass_from_data)
#                 # normalization = \sum ROI_background
#                 # Ńormalized intensity:
#                 # \sum (ROI_bragg - ROI_background)/ normalization
#                 normalization = np.sum(slices['background_data'])
#                 intensity = np.sum(mass_from_data)/normalization
#
#                 # Centre of mass
#                 mass = ndimage.measurements.center_of_mass(mass_from_data)
#
#                 this_off_pulses.append(np.append(np.array(mass), intensity))
#
#             this_off_pulses = np.array(this_off_pulses)
#             self._off_pulses_hist.append(this_off_pulses)
#             # Same logic as LaserOnOffWindow. Running averages over
#             # trains.
#             if self._off_pulses_ma is None:
#                 self._off_pulses_ma = np.copy(this_off_pulses)
#             elif len(self._off_pulses_hist) <= self.moving_avg_window_sp:
#                 self._off_pulses_ma += \
#                     (this_off_pulses - self._off_pulses_ma) \
#                     / len(self._off_pulses_hist)
#             elif len(self._off_pulses_hist) == self.moving_avg_window_sp + 1:
#                 self._off_pulses_ma += \
#                     (this_off_pulses - self._off_pulses_hist.popleft()) \
#                     / self.moving_avg_window_sp
#             else:
#                 raise ValueError
#
#             com_off = self._off_pulses_ma
#
#             # This part at the moment makes no physical sense. Atleast to me.
#             # To be discussed with Dmitry. I added it here for some kind of
#             # history book keeping
#             if com_off.size != 0:
#                 self._hist_train_off_id.append(data.tid)
#                 self._hist_com_off.append(np.mean(np.array(com_off), axis=0))
#
#             self._on_train_received = False
#             self._off_train_received = False
#
#         if hasattr(com_on, 'size') is True and com_on.size == 0:
#             com_on = None
#         if hasattr(com_off, 'size') is True and com_off.size == 0:
#             com_off = None
#
#         return com_on, com_off
#
#     def update(self):
#         data = self._data.get()
#         if data.empty():
#             return
#         self._main_vb.setMouseEnabled(x=False, y=False)
#         self._image_items[0].setImage(np.flip(data.image_mean, axis=0),
#                                       autoLevels=False,
#                                       levels=(0, data.image_mean.max()))
#         # Size of two region of interests should stay same.
#         # Important when Backgorund has to be subtracted from Bragg data
#         # TODO: Size of ROI should not be independent
#         size_bragg = (self._rois[0]).size()
#         self._rois[1].setSize(size_bragg)
#
#         # Profile analysis (Histogram) along a line
#         # Horizontal and vertical line region of interests
#         # Histograms along these lines plotted in the bottom panel
#         if self._ana_params.child('Profile Analysis').value():
#
#             if len(self._profile_line_rois) > 0:
#                 for line in self._profile_line_rois:
#                     index = self._profile_line_rois.index(line)
#
#                     slice_hist = line.getArrayRegion(
#                         data.image_mean, self._image_items[0])
#                     y, x = np.histogram(slice_hist, bins=np.linspace(
#                         slice_hist.min(), slice_hist.max(), 50))
#                     self._profile_plot_items[index].plot(
#                         x, y, stepMode=True, fillLevel=0,
#                         brush=(255, 0, 255, 150))
#
#         # Plot average image around two region of interests.
#         # Selected Bragg region and Background
#         for index, roi in enumerate(self._rois):
#             self._image_items[index+1].setImage(roi.getArrayRegion(
#                 np.flip(data.image_mean, axis=0),
#                 self._image_items[0]), levels=(0, data.image_mean.max()))
#         # com_on and com_off are of shape (num_pulses,3)
#         # contains (pulse_index, com_x, com_y, normalized intensity)
#         t0 = time.perf_counter()
#
#         com_on, com_off = self._update(data)
#
#         logger.debug("Time for centre of mass evaluation: {:.1f} ms\n"
#                      .format(1000 * (time.perf_counter() - t0)))
#
#         # plot COM X, COM Y and normalised intensity as a function of
#         # pulseIds
#
#         for index, p in enumerate(self._plot_items[:-2]):
#             p.addLegend()
#             if index == 0 or index == 2:
#                 p.setTitle(' TrainId :: {}'.format(data.tid))
#             if com_on is not None:
#                 p.plot(self.on_pulse_ids_sp[0:com_on.shape[0]],
#                        com_on[:, index], name='On', pen=make_pen("green"),
#                        symbol='o', symbolBrush=mkBrush(0, 255, 0, 255))
#             if com_off is not None:
#                 p.plot(self.off_pulse_ids_sp[0:com_off.shape[0]],
#                        com_off[:, index], name="Off",
#                        pen=make_pen("purple"), symbol='o',
#                        symbolBrush=mkBrush(255, 0, 255, 255))
#
#         for idx, p in enumerate(self._plot_items[-2:]):
#             p.clear()
#             if self._hist_com_off:
#                 s = ScatterPlotItem(size=10,
#                                     pen=mkPen(None),
#                                     brush=mkBrush(120, 255, 255, 255))
#                 s.addPoints([{'pos': (i, v), 'data': 1} for i, v in
#                              zip(self._hist_train_off_id,
#                                  np.array(self._hist_com_off)[:, idx])])
#
#                 p.addItem(s)
#                 p.plot(self._hist_train_off_id,
#                        np.array(self._hist_com_off)[:, idx],
#                        pen=make_pen("purple"), name='Off')
#             if self._hist_com_on:
#                 s = ScatterPlotItem(size=10,
#                                     pen=mkPen(None),
#                                     brush=mkBrush(240, 255, 255, 255))
#                 s.addPoints([{'pos': (i, v), 'data': 1} for i, v in
#                              zip(self._hist_train_on_id,
#                                  np.array(self._hist_com_on)[:, idx])])
#
#                 p.addItem(s)
#
#                 p.plot(self._hist_train_on_id,
#                        np.array(self._hist_com_on)[:, idx],
#                        pen=make_pen("green"), name='On')
#                 p.addLegend()
#
#
#     # Profile state change triggers this function
#     # If profile is checked, adds bottom panels to plot histograms.
#     def _profile(self):
#         if self._ana_params.child('Profile Analysis').value():
#
#             self._image_gl_widget.ci.layout.setRowStretchFactor(0, 2)
#             self._image_gl_widget.ci.layout.setRowStretchFactor(1, 2)
#             self._image_gl_widget.ci.layout.setRowStretchFactor(2, 2)
#             self._image_gl_widget.ci.layout.setRowStretchFactor(3, 2)
#             profile_plot = self._image_gl_widget.addPlot(
#                 row=4, col=0, rowspan=3, colspan=1)
#
#             self._profile_plot_items.append(profile_plot)
#             profile_plot = self._image_gl_widget.addPlot(
#                 row=4, col=1, rowspan=3, colspan=1)
#             self._image_gl_widget.ci.layout.setRowStretchFactor(4, 2)
#
#             self._profile_plot_items.append(profile_plot)
#
#             self._image_items[0].mouseClickEvent = self._click
#
#         else:
#
#             if len(self._profile_plot_items) > 0:
#                 for item in self._profile_plot_items:
#                     self._image_gl_widget.removeItem(item)
#                 self._profile_plot_items.clear()
#             if len(self._profile_line_rois) > 0:
#                 for line in self._profile_line_rois:
#                     self._main_vb.removeItem(line)
#                 self._profile_line_rois.clear()
#
#             self._image_gl_widget.ci.layout.setRowStretchFactor(0, 1)
#             self._image_gl_widget.ci.layout.setRowStretchFactor(1, 1)
#             self._image_gl_widget.ci.layout.setRowStretchFactor(2, 1)
#             self._image_gl_widget.ci.layout.setRowStretchFactor(3, 1)
#
#
#     # Mouse click on image in top left panel creates two line
#     # region of interests. One horizontal and one vertical.
#     def _click(self, event):
#         data = self._data.get()
#         if data.empty():
#             return
#
#         event.accept()
#
#         pos = event.pos()
#         x = int(pos.x())
#         y = int(pos.y())
#         x_pos, y_pos = data.image_mean.shape
#
#         if len(self._profile_line_rois) > 0:
#             for line in self._profile_line_rois:
#                 self._main_vb.removeItem(line)
#             self._profile_line_rois.clear()
#
#         line_roi = LineSegmentROI(
#             [[0, y], [y_pos, y]],
#             pen=mkPen((255, 255, 255), width=3), movable=False)
#         self._profile_line_rois.append(line_roi)
#
#         line_roi = LineSegmentROI(
#             [[x, 0], [x, x_pos]],
#             pen=mkPen((255, 255, 255), width=3), movable=False)
#         self._profile_line_rois.append(line_roi)
#         for line in self._profile_line_rois:
#             self._main_vb.addItem(line)
#
#         if self._ana_params.child('Profile Analysis').value():
#             for line in self._profile_line_rois:
#                 index = self._profile_line_rois.index(line)
#                 self._profile_plot_items[index].clear()
#
#                 slice_hist = line.getArrayRegion(
#                     data.image_mean, self._image_items[0])
#                 y, x = np.histogram(slice_hist, bins=np.linspace(
#                     slice_hist.min(), slice_hist.max(), 50))
#                 self._profile_plot_items[index].plot(
#                     x, y, stepMode=True, fillLevel=0,
#                     brush=(255, 0, 255, 150))
#
#     def clear(self):
#         """Override."""
#         for item in self._image_items:
#             item.clear()
#         for plot in self._plot_items[:-2]:
#             plot.clear()
#         if len(self._profile_plot_items) > 0:
#             for plot in self._profile_plot_items:
#                 plot.clear()
#
#     def _reset(self):
#         for plot in self._plot_items[-2:]:
#             plot.clear()
#
#         self._on_train_received = False
#         self._off_train_received = False
#         self._drop_last_on_pulse = False
#         self._on_pulses_ma = None
#         self._off_pulses_ma = None
#         self._on_pulses_hist.clear()
#         self._off_pulses_hist.clear()
#         # TODO: Fix hostory
#         self._hist_com_on.clear()
#         self._hist_com_off.clear()
#         self._hist_train_on_id.clear()
#         self._hist_train_off_id.clear()
