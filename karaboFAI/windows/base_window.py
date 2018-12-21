"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold base classes for windows.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
from weakref import WeakKeyDictionary

from ..widgets.pyqtgraph import GraphicsLayoutWidget, QtCore, QtGui
from ..widgets.pyqtgraph import parametertree as ptree
from ..widgets.pyqtgraph.dockarea import DockArea


class SingletonWindow:
    """SingletonWindow decorator.

    A singleton window is only allowed to have one instance.
    """
    def __init__(self, instance_type):
        self.instance = None
        self.instance_type = instance_type

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.instance_type(*args, **kwargs)
        else:
            if isinstance(self.instance, PlotWindow) \
                    or isinstance(self.instance, DockerWindow):
                self.instance.parent().registerPlotWindow(self.instance)
                self.instance.update()

        self.instance.show()
        return self.instance


class AbstractWindow(QtGui.QMainWindow):
    """Base class for various stand-alone windows.

    All the stand-alone windows should follow the interface defined
    in this abstract class.
    """
    title = ""

    def __init__(self, data, *, parent=None):
        """Initialization.

        :param Data4Visualization data: the data shared by widgets
            and windows.
        """
        super().__init__(parent=parent)
        self._data = data

        try:
            if self.title:
                title = parent.title + " - " + self.title
            else:
                title = parent.title

            self.setWindowTitle(title)
        except AttributeError:
            # for unit test where parent is None
            self.setWindowTitle(self.title)

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        self.show()

    def initUI(self):
        """Initialization of UI.

        This method should call 'initCtrlUI' and 'initPlotUI'.
        """
        pass

    def initCtrlUI(self):
        """Initialization of ctrl UI.

        Initialization of the ctrl UI should take place in this method.
        """
        pass

    def initPlotUI(self):
        """Initialization of plot UI.

        Initialization of the plot UI should take place in this method.
        """
        pass


class DockerWindow(AbstractWindow):
    """QMainWindow displaying a single DockArea."""
    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)
        self.parent().registerPlotWindow(self)

        self._plot_widgets = WeakKeyDictionary()  # book-keeping opened windows

        self._docker_area = DockArea()

    def initUI(self):
        """Override."""
        self.initPlotUI()
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._docker_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cw.setLayout(layout)

    def update(self):
        """Update widgets.

        This method is called by the main GUI.
        """
        data = self._data.get()
        if data.empty():
            return

        for widget in self._plot_widgets:
            widget.update(data)

    def clear(self):
        """Clear widgets.

        This method is called by the main GUI.
        """
        for widget in self._plot_widgets:
            widget.clear()

    def registerPlotWidget(self, instance):
        self._plot_widgets[instance] = 1

    def unregisterPlotWidget(self, instance):
        del self._plot_widgets[instance]

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.parent().unregisterPlotWindow(self)


class PlotWindow(AbstractWindow):
    """QMainWindow consists of a GraphicsLayoutWidget and a ParameterTree."""

    available_modes = OrderedDict({
        "normal": "Laser-on/off pulses in the same train",
        "even/odd": "Laser-on/off pulses in even/odd train",
        "odd/even": "Laser-on/off pulses in odd/even train"
    })

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)
        self.parent().registerPlotWindow(self)

        self._gl_widget = GraphicsLayoutWidget()
        self._ctrl_widget = None

        self._plot_items = []  # bookkeeping PlotItem objects
        self._image_items = []  # bookkeeping ImageItem objects

        # -------------------------------------------------------------
        # define parameter tree
        # -------------------------------------------------------------

        self._ptree = ptree.ParameterTree(showHeader=True)

        # parameters are grouped into 4 groups
        self._exp_params = ptree.Parameter.create(
            name='Experimental setups', type='group')
        self._pro_params = ptree.Parameter.create(
            name='Data processing parameters', type='group')
        self._vis_params = ptree.Parameter.create(
            name='Visualization options', type='group')
        self._act_params = ptree.Parameter.create(
            name='Actions', type='group')
        self._ana_params = ptree.Parameter.create(
            name='Analysis options', type='group')
        self._ins_params = ptree.Parameter.create(
            name='General', type='group')

        # -------------------------------------------------------------
        # define slots' behaviors
        # -------------------------------------------------------------

        # shared parameters are updated by signal-slot
        # Note: shared parameters should end with '_sp'
        self.mask_range_sp = None
        self.diff_integration_range_sp = None
        self.normalization_range_sp = None
        self.ma_window_size_sp = None
        self.laser_mode_sp = None
        self.on_pulse_ids_sp = None
        self.off_pulse_ids_sp = None

        self.parent().mask_range_sgn.connect(self.onMaskRangeChanged)
        self.parent().on_off_pulse_ids_sgn.connect(self.onOffPulseIdChanged)
        self.parent().diff_integration_range_sgn.connect(
            self.onDiffIntegrationRangeChanged)
        self.parent().normalization_range_sgn.connect(
            self.onNormalizationRangeChanged)
        self.parent().ma_window_size_sgn.connect(self.onMAWindowSizeChanged)

        # -------------------------------------------------------------
        # available Parameters (shared parameters and actions)
        # -------------------------------------------------------------

        self.mask_range_param = ptree.Parameter.create(
            name='Mask range', type='str', readonly=True
        )
        self.optical_laser_mode_param = ptree.Parameter.create(
            name='Optical laser mode', type='str', readonly=True
        )
        self.laser_on_pulse_ids_param = ptree.Parameter.create(
            name='Laser-on pulse ID(s)', type='str', readonly=True
        )
        self.laser_off_pulse_ids_param = ptree.Parameter.create(
            name='Laser-off pulse ID(s)', type='str', readonly=True
        )
        self.normalization_range_param = ptree.Parameter.create(
            name="Normalization range", type='str', readonly=True
        )
        self.diff_integration_range_param = ptree.Parameter.create(
            name="Diff integration range", type='str', readonly=True
        )
        self.ma_window_size_param = ptree.Parameter.create(
            name='M.A. window size', type='int', readonly=True
        )
        self.reset_action_param = ptree.Parameter.create(
            name='Clear history', type='action'
        )
        self.reset_action_param.sigActivated.connect(self._reset)

        # this method inject parameters into the parameter tree
        self.updateParameterTree()

        # tell MainGUI to emit signals in order to update shared parameters
        self.parent().updateSharedParameters()

    def initUI(self):
        """Override."""
        self.initCtrlUI()
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        if self._ctrl_widget is not None:
            layout.addWidget(self._ctrl_widget)
        layout.addWidget(self._gl_widget)
        self._cw.setLayout(layout)

    def update(self):
        """Update plots.

        This method is called by the main GUI.
        """
        raise NotImplementedError

    def clear(self):
        """Clear plots.

        This method is called by the main GUI.
        """
        for item in self._plot_items:
            item.clear()
        for item in self._image_items:
            item.clear()

    @QtCore.pyqtSlot(str, list, list)
    def onOffPulseIdChanged(self, mode, on_pulse_ids, off_pulse_ids):
        self.laser_mode_sp = mode
        self.on_pulse_ids_sp = on_pulse_ids
        self.off_pulse_ids_sp = off_pulse_ids
        # then update the parameter tree
        try:
            self._exp_params.child('Optical laser mode').setValue(
                self.available_modes[mode])
            self._exp_params.child('Laser-on pulse ID(s)').setValue(
                ', '.join([str(x) for x in on_pulse_ids]))
            self._exp_params.child('Laser-off pulse ID(s)').setValue(
                ', '.join([str(x) for x in off_pulse_ids]))
        except KeyError:
            pass

    @QtCore.pyqtSlot(float, float)
    def onMaskRangeChanged(self, lb, ub):
        self.mask_range_sp = (lb, ub)
        # then update the parameter tree
        try:
            self._pro_params.child('Mask range').setValue(
                '{}, {}'.format(lb, ub))
        except KeyError:
            pass

    @QtCore.pyqtSlot(float, float)
    def onNormalizationRangeChanged(self, lb, ub):
        self.normalization_range_sp = (lb, ub)
        # then update the parameter tree
        try:
            self._pro_params.child('Normalization range').setValue(
                '{}, {}'.format(lb, ub))
        except KeyError:
            pass

    @QtCore.pyqtSlot(float, float)
    def onDiffIntegrationRangeChanged(self, lb, ub):
        self.diff_integration_range_sp = (lb, ub)
        # then update the parameter tree
        try:
            self._pro_params.child("Diff integration range").setValue(
                '{}, {}'.format(lb, ub))
        except KeyError:
            pass

    @QtCore.pyqtSlot(int)
    def onMAWindowSizeChanged(self, value):
        self.ma_window_size_sp = value
        # then update the parameter tree
        try:
            self._pro_params.child('M.A. window size').setValue(str(value))
        except KeyError:
            pass

    def updateParameterTree(self):
        """Update the parameter tree.

        In this method, one should and only should have codes like

        self._exp_params.addChildren(...)
        self._pro_params.addChildren(...)
        self._vis_params.addChildren(...)
        self._act_params.addChildren(...)

        params = ptree.Parameter.create(name='params', type='group',
                                        children=[self._exp_params,
                                                  self._pro_params,
                                                  self._vis_params,
                                                  self._act_params])

        self._ptree.setParameters(params, showTop=False)

        Here '...' is a list of Parameter instances or dictionaries which
        can be used to instantiate Parameter instances.
        """
        pass

    def _reset(self):
        """Reset all internal states/histories."""
        pass

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.parent().unregisterPlotWindow(self)
