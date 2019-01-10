from collections import OrderedDict
import zmq

from ..config import config
from ..data_processing import DataSource
from ..file_server import FileServer
from ..helpers import parse_ids, parse_boundary, parse_table_widget
from ..logger import logger
from ..widgets.pyqtgraph import QtCore, QtGui
from ..widgets.misc_widgets import (
    CustomGroupBox, FixedWidthLineEdit, GuiLogger, InputDialogWithCheckBox
)


class ControlWidget(QtGui.QWidget):
    """Base class for the control widgets

    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent().registerControlWidget(self)

        self._ctrl_widget = None
        self._local_widgets_to_disable_during_daq = []

    def initUI(self):

        self.parent()._disabled_widgets_during_daq.extend(
            self._local_widgets_to_disable_during_daq)

        layout = QtGui.QHBoxLayout()

        if self._ctrl_widget is not None:
            layout.addWidget(self._ctrl_widget)
        self.setLayout(layout)

    def updateSharedParameters(self, log=False):
        """Update shared parameters for control widget.

        :params bool log: True for logging shared parameters and False
            for not.

        Returns bool: True if all shared parameters successfully parsed
            and emitted, otherwise False.

        Return True: If method not implemented in the inherited widget
                     class
        """

        if log:
            logger.info("--- No Shared parameters ---")
        return True


class AiSetUpWidget(ControlWidget):
    """Azimuthal integration set up class

    creates a widget for azimuthal integration parameters.
    """

    # *************************************************************
    # signals related to shared parameters
    # *************************************************************
    sample_distance_sgn = QtCore.pyqtSignal(float)
    center_coordinate_sgn = QtCore.pyqtSignal(int, int)  # (cx, cy)
    integration_method_sgn = QtCore.pyqtSignal(str)
    integration_range_sgn = QtCore.pyqtSignal(float, float)
    integration_points_sgn = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # *************************************************************
        # Azimuthal integration setup
        # *************************************************************

        self._ctrl_widget = CustomGroupBox("Azimuthal integration setup")

        w = 100
        self._sample_dist_le = FixedWidthLineEdit(w, str(config["DISTANCE"]))
        self._cx_le = FixedWidthLineEdit(w, str(config["CENTER_X"]))
        self._cy_le = FixedWidthLineEdit(w, str(config["CENTER_Y"]))
        self._itgt_method_cb = QtGui.QComboBox()
        self._itgt_method_cb.setFixedWidth(w)
        for method in config["INTEGRATION_METHODS"]:
            self._itgt_method_cb.addItem(method)
        self._itgt_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._itgt_points_le = FixedWidthLineEdit(
            w, str(config["INTEGRATION_POINTS"]))

        self._local_widgets_to_disable_during_daq = [
            self._sample_dist_le,
            self._cx_le,
            self._cy_le,
            self._itgt_method_cb,
            self._itgt_range_le,
            self._itgt_points_le,
        ]

        self._initCtrlUI()
        self.initUI()

    def _initCtrlUI(self):

        sample_dist_lb = QtGui.QLabel("Sample distance (m): ")
        cx = QtGui.QLabel("Cx (pixel): ")
        cy = QtGui.QLabel("Cy (pixel): ")
        itgt_method_lb = QtGui.QLabel("Integration method: ")
        itgt_points_lb = QtGui.QLabel("Integration points: ")
        itgt_range_lb = QtGui.QLabel("Integration range (1/A): ")

        layout = QtGui.QGridLayout()
        layout.addWidget(sample_dist_lb, 0, 0, 1, 1)
        layout.addWidget(self._sample_dist_le, 0, 1, 1, 1)
        layout.addWidget(cx, 1, 0, 1, 1)
        layout.addWidget(self._cx_le, 1, 1, 1, 1)
        layout.addWidget(cy, 2, 0, 1, 1)
        layout.addWidget(self._cy_le, 2, 1, 1, 1)
        layout.addWidget(itgt_method_lb, 4, 0, 1, 1)
        layout.addWidget(self._itgt_method_cb, 4, 1, 1, 1)
        layout.addWidget(itgt_points_lb, 5, 0, 1, 1)
        layout.addWidget(self._itgt_points_le, 5, 1, 1, 1)
        layout.addWidget(itgt_range_lb, 6, 0, 1, 1)
        layout.addWidget(self._itgt_range_le, 6, 1, 1, 1)

        self._ctrl_widget.setLayout(layout)

    def updateSharedParameters(self, log=False):
        """Override"""

        sample_distance = float(self._sample_dist_le.text().strip())
        if sample_distance <= 0:
            logger.error("<Sample distance>: Invalid input! Must be positive!")
            return False
        else:
            self.sample_distance_sgn.emit(sample_distance)

        center_x = int(self._cx_le.text().strip())
        center_y = int(self._cy_le.text().strip())
        self.center_coordinate_sgn.emit(center_x, center_y)

        integration_method = self._itgt_method_cb.currentText()
        self.integration_method_sgn.emit(integration_method)

        integration_points = int(self._itgt_points_le.text().strip())
        if integration_points <= 0:
            logger.error(
                "<Integration points>: Invalid input! Must be positive!")
            return False
        else:
            self.integration_points_sgn.emit(integration_points)

        try:
            integration_range = parse_boundary(self._itgt_range_le.text())
            self.integration_range_sgn.emit(*integration_range)
        except ValueError as e:
            logger.error("<Integration range>: " + str(e))
            return False

        if log:
            logger.info("--- Shared parameters ---")
            logger.info("<Sample distance (m)>: {}".format(sample_distance))
            logger.info("<Cx (pixel), Cy (pixel>: ({:d}, {:d})".
                        format(center_x, center_y))
            logger.info("<Cy (pixel)>: {:d}".format(center_y))
            logger.info("<Integration method>: '{}'".format(
                integration_method))
            logger.info("<Integration range (1/A)>: ({}, {})".
                        format(*integration_range))
            logger.info("<Number of integration points>: {}".
                        format(integration_points))

        return True


class GmtSetUpWidget(ControlWidget):
    """Geometry set up class

    creates a widget for Geometry parameters.
    """

    # *************************************************************
    # signals related to shared parameters
    # *************************************************************
    geometry_sgn = QtCore.pyqtSignal(str, list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # *************************************************************
        # Geometry setup
        # *************************************************************

        self._ctrl_widget = CustomGroupBox("Geometry setup")
        self._quad_positions_tb = QtGui.QTableWidget()
        self._geom_file_le = FixedWidthLineEdit(285, config["GEOMETRY_FILE"])

        self._local_widgets_to_disable_during_daq = [
            self._quad_positions_tb,
            self._geom_file_le,
        ]

        self._initCtrlUI()
        self.initUI()

    def _initCtrlUI(self):

        geom_file_lb = QtGui.QLabel("Geometry file:")
        quad_positions_lb = QtGui.QLabel("Quadrant positions:")

        self._initQuadTable()

        layout = QtGui.QGridLayout()
        layout.addWidget(geom_file_lb, 0, 0, 1, 3)
        layout.addWidget(self._geom_file_le, 1, 0, 1, 3)
        layout.addWidget(quad_positions_lb, 2, 0, 1, 2)
        layout.addWidget(self._quad_positions_tb, 3, 0, 1, 2)

        self._ctrl_widget.setLayout(layout)

    def _initQuadTable(self):
        n_row = 4
        n_col = 2
        widget = self._quad_positions_tb
        widget.setRowCount(n_row)
        widget.setColumnCount(n_col)
        try:
            for i in range(n_row):
                for j in range(n_col):
                    widget.setItem(i, j, QtGui.QTableWidgetItem(
                        str(config["QUAD_POSITIONS"][i][j])))
        except IndexError:
            pass

        widget.move(0, 0)
        widget.setHorizontalHeaderLabels(['x', 'y'])
        widget.setVerticalHeaderLabels(['1', '2', '3', '4'])
        widget.setColumnWidth(0, 80)
        widget.setColumnWidth(1, 80)

    def updateSharedParameters(self, log=False):
        """Override"""

        try:
            geom_file = self._geom_file_le.text()
            quad_positions = parse_table_widget(self._quad_positions_tb)
            self.geometry_sgn.emit(geom_file, quad_positions)
        except ValueError as e:
            logger.error("<Quadrant positions>: " + str(e))
            return False

        if log:
            logger.info("--- Shared parameters ---")
            logger.info("<Geometry file>: {}".format(geom_file))
            logger.info("<Quadrant positions>: [{}]".format(
                ", ".join(["[{}, {}]".format(p[0], p[1])
                           for p in quad_positions])))

        return True


class ExpSetUpWidget(ControlWidget):
    """Experiment set up class

    creates a widget for the Expreriment details.
    """

    available_modes = OrderedDict({
        "normal": "Laser-on/off pulses in the same train",
        "even/odd": "Laser-on/off pulses in even/odd train",
        "odd/even": "Laser-on/off pulses in odd/even train"
    })

    # *************************************************************
    # signals related to shared parameters
    # *************************************************************
    diff_integration_range_sgn = QtCore.pyqtSignal(float, float)
    normalization_range_sgn = QtCore.pyqtSignal(float, float)
    mask_range_sgn = QtCore.pyqtSignal(float, float)
    ma_window_size_sgn = QtCore.pyqtSignal(int)
    # (mode, on-pulse ids, off-pulse ids)
    on_off_pulse_ids_sgn = QtCore.pyqtSignal(str, list, list)
    photon_energy_sgn = QtCore.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # *************************************************************
        # Experiment setup
        # *************************************************************
        self._ctrl_widget = CustomGroupBox("Experiment setup")

        w = 100
        self._photon_energy_le = FixedWidthLineEdit(
            w, str(config["PHOTON_ENERGY"]))
        self._laser_mode_cb = QtGui.QComboBox()
        self._laser_mode_cb.setFixedWidth(w)
        self._laser_mode_cb.addItems(self.available_modes.keys())
        self._on_pulse_le = FixedWidthLineEdit(w, "0:8:2")
        self._off_pulse_le = FixedWidthLineEdit(w, "1:8:2")
        self._normalization_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._diff_integration_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._mask_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["MASK_RANGE"]]))
        self._ma_window_le = FixedWidthLineEdit(w, "9999")

        self._local_widgets_to_disable_during_daq = [
            self._photon_energy_le,
            self._laser_mode_cb,
            self._on_pulse_le,
            self._off_pulse_le,
            self._normalization_range_le,
            self._diff_integration_range_le,
            self._mask_range_le,
            self._ma_window_le,
        ]

        self._initCtrlUI()
        self.initUI()

    def _initCtrlUI(self):

        photon_energy_lb = QtGui.QLabel("Photon energy (keV): ")
        laser_mode_lb = QtGui.QLabel("Laser on/off mode: ")
        on_pulse_lb = QtGui.QLabel("On-pulse IDs: ")
        off_pulse_lb = QtGui.QLabel("Off-pulse IDs: ")
        normalization_range_lb = QtGui.QLabel("Normalization range (1/A): ")
        diff_integration_range_lb = QtGui.QLabel(
            "Diff integration range (1/A): ")
        mask_range_lb = QtGui.QLabel("Mask range: ")
        ma_window_lb = QtGui.QLabel("M.A. window size: ")

        layout = QtGui.QGridLayout()
        layout.addWidget(photon_energy_lb, 0, 0, 1, 1)
        layout.addWidget(self._photon_energy_le, 0, 1, 1, 1)
        layout.addWidget(laser_mode_lb, 1, 0, 1, 1)
        layout.addWidget(self._laser_mode_cb, 1, 1, 1, 1)
        layout.addWidget(on_pulse_lb, 2, 0, 1, 1)
        layout.addWidget(self._on_pulse_le, 2, 1, 1, 1)
        layout.addWidget(off_pulse_lb, 3, 0, 1, 1)
        layout.addWidget(self._off_pulse_le, 3, 1, 1, 1)
        layout.addWidget(normalization_range_lb, 4, 0, 1, 1)
        layout.addWidget(self._normalization_range_le, 4, 1, 1, 1)
        layout.addWidget(diff_integration_range_lb, 5, 0, 1, 1)
        layout.addWidget(self._diff_integration_range_le, 5, 1, 1, 1)
        layout.addWidget(mask_range_lb, 6, 0, 1, 1)
        layout.addWidget(self._mask_range_le, 6, 1, 1, 1)
        layout.addWidget(ma_window_lb, 7, 0, 1, 1)
        layout.addWidget(self._ma_window_le, 7, 1, 1, 1)

        self._ctrl_widget.setLayout(layout)

    def updateSharedParameters(self, log=False):
        """Override"""

        try:
            normalization_range = parse_boundary(
                self._normalization_range_le.text())
            self.normalization_range_sgn.emit(*normalization_range)
        except ValueError as e:
            logger.error("<Normalization range>: " + str(e))
            return False

        try:
            diff_integration_range = parse_boundary(
                self._diff_integration_range_le.text())
            self.diff_integration_range_sgn.emit(*diff_integration_range)
        except ValueError as e:
            logger.error("<Diff integration range>: " + str(e))
            return False
        try:
            mask_range = parse_boundary(self._mask_range_le.text())
            self.mask_range_sgn.emit(*mask_range)
        except ValueError as e:
            logger.error("<Mask range>: " + str(e))
            return False

        try:
            # check pulse ID only when laser on/off pulses are in the same
            # train (the "normal" mode)
            mode = self._laser_mode_cb.currentText()
            on_pulse_ids = parse_ids(self._on_pulse_le.text())
            off_pulse_ids = parse_ids(self._off_pulse_le.text())
            if mode == list(self.available_modes.keys())[0]:
                common = set(on_pulse_ids).intersection(off_pulse_ids)
                if common:
                    logger.error(
                        "Pulse IDs {} are found in both on- and off- pulses.".
                        format(','.join([str(v) for v in common])))
                    return False

            self.on_off_pulse_ids_sgn.emit(
                mode, on_pulse_ids, off_pulse_ids)
        except ValueError:
            logger.error("Invalid input! Enter on/off pulse IDs separated "
                         "by ',' and/or use the range operator ':'!")
            return False

        try:
            window_size = int(self._ma_window_le.text())
            if window_size < 1:
                logger.error("Moving average window width < 1!")
                return False
            self.ma_window_size_sgn.emit(window_size)
        except ValueError as e:
            logger.error("<Moving average window size>: " + str(e))
            return False

        photon_energy = float(self._photon_energy_le.text().strip())
        if photon_energy <= 0:
            logger.error("<Photon energy>: Invalid input! Must be positive!")
            return False
        else:
            self.photon_energy_sgn.emit(photon_energy)

        if log:
            logger.info("--- Shared parameters ---")
            logger.info("<Optical laser mode>: {}".format(mode))
            logger.info("<On-pulse IDs>: {}".format(on_pulse_ids))
            logger.info("<Off-pulse IDs>: {}".format(off_pulse_ids))
            logger.info("<Normalization range>: ({}, {})".
                        format(*normalization_range))
            logger.info("<Diff integration range>: ({}, {})".
                        format(*diff_integration_range))
            logger.info("<Moving average window size>: {}".
                        format(window_size))
            logger.info("<Photon energy (keV)>: {}".format(photon_energy))
            logger.info("<Mask range>: ({}, {})".format(*mask_range))

        return True


class DataSrcFileServerWidget(ControlWidget):
    """Data source and file server set up class

    creates a widget for the data source details and file server buttons.
    """

    # *************************************************************
    # signals related to shared parameters
    # *************************************************************

    server_tcp_sgn = QtCore.pyqtSignal(str, str)
    data_source_sgn = QtCore.pyqtSignal(object)
    pulse_range_sgn = QtCore.pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # *************************************************************
        # data source options
        # *************************************************************
        self._data_src_gp = CustomGroupBox("Data source")

        self._hostname_le = FixedWidthLineEdit(165, config["SERVER_ADDR"])
        self._port_le = FixedWidthLineEdit(70, str(config["SERVER_PORT"]))
        self._source_name_le = FixedWidthLineEdit(280, config["SOURCE_NAME"])
        self._pulse_range0_le = FixedWidthLineEdit(60, str(0))
        self._pulse_range1_le = FixedWidthLineEdit(60, str(2699))

        self._data_src_rbts = []
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@files"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@ZMQ bridge"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Assembled data@ZMQ bridge"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Processed data@ZMQ bridge"))
        self._data_src_rbts[int(config["SOURCE_TYPE"])].setChecked(True)

        # *************************************************************
        # file server
        # *************************************************************
        self._file_server = None
        self._file_server_widget = CustomGroupBox("Data stream server")
        self._server_start_btn = QtGui.QPushButton("Serve")
        self._server_start_btn.clicked.connect(self._onStartServeFile)
        self._server_terminate_btn = QtGui.QPushButton("Terminate")
        self._server_terminate_btn.setEnabled(False)
        self._server_terminate_btn.clicked.connect(
            self._onStopServeFile)

        self._pulse_range0_le.setEnabled(False)

        self._disabled_widgets_during_file_serving = [
            self._source_name_le,
        ]

        self._local_widgets_to_disable_during_daq = [
            self._hostname_le,
            self._port_le,
            self._source_name_le,
            self._pulse_range1_le,
        ]
        self._local_widgets_to_disable_during_daq.extend(self._data_src_rbts)

        self._initCtrlUI()
        self.initUI()

    @property
    def file_server(self):
        return self._file_server

    def _initCtrlUI(self):
        self._initDataSrcUI()
        self._initFileServerUI()

        self._ctrl_widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._data_src_gp, 2)
        layout.addWidget(self._file_server_widget, 1)
        self._ctrl_widget.setLayout(layout)

    def _initDataSrcUI(self):
        # *************************************************************
        # data source panel
        # *************************************************************
        hostname_lb = QtGui.QLabel("Hostname: ")
        self._hostname_le.setAlignment(QtCore.Qt.AlignCenter)
        port_lb = QtGui.QLabel("Port: ")
        self._port_le.setAlignment(QtCore.Qt.AlignCenter)
        source_name_lb = QtGui.QLabel("Source: ")
        self._source_name_le.setAlignment(QtCore.Qt.AlignCenter)
        pulse_range_lb = QtGui.QLabel("Pulse ID range: ")
        self._pulse_range0_le.setAlignment(QtCore.Qt.AlignCenter)
        self._pulse_range1_le.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        sub_layout1 = QtGui.QHBoxLayout()
        sub_layout1.addWidget(hostname_lb)
        sub_layout1.addWidget(self._hostname_le)
        sub_layout1.addWidget(port_lb)
        sub_layout1.addWidget(self._port_le)
        sub_layout2 = QtGui.QHBoxLayout()
        sub_layout2.addWidget(pulse_range_lb)
        sub_layout2.addWidget(self._pulse_range0_le)
        sub_layout2.addWidget(QtGui.QLabel(" to "))
        sub_layout2.addWidget(self._pulse_range1_le)
        sub_layout2.addStretch(2)
        sub_layout3 = QtGui.QHBoxLayout()
        sub_layout3.addWidget(source_name_lb)
        sub_layout3.addWidget(self._source_name_le)
        layout.addLayout(sub_layout1)
        layout.addLayout(sub_layout3)
        for btn in self._data_src_rbts:
            layout.addWidget(btn)
        layout.addLayout(sub_layout2)
        self._data_src_gp.setLayout(layout)

    def _initFileServerUI(self):
        layout = QtGui.QGridLayout()
        layout.addWidget(self._server_start_btn, 0, 0, 1, 1)
        layout.addWidget(self._server_terminate_btn, 0, 1, 1, 1)
        self._file_server_widget.setLayout(layout)

    def _onStartServeFile(self):
        """Actions taken before the start of file serving."""
        folder = self._source_name_le.text().strip()
        port = int(self._port_le.text().strip())
        # process can only be start once
        self._file_server = FileServer(folder, port)
        try:
            # TODO: signal the end of file serving
            self._file_server.start()
            logger.info("Start serving file in the folder {} through port {}"
                        .format(folder, port))
        except FileNotFoundError:
            logger.info("{} does not exist!".format(folder))
            return
        except zmq.error.ZMQError:
            logger.info("Port {} is already in use!".format(port))
            return

        self._server_terminate_btn.setEnabled(True)
        self._server_start_btn.setEnabled(False)
        for widget in self._disabled_widgets_during_file_serving:
            widget.setEnabled(False)

    def _onStopServeFile(self):
        """Actions taken before the end of file serving."""
        self._file_server.terminate()
        self._server_terminate_btn.setEnabled(False)
        self._server_start_btn.setEnabled(True)
        for widget in self._disabled_widgets_during_file_serving:
            widget.setEnabled(True)

    def updateSharedParameters(self, log=False):
        """Override"""

        if self._data_src_rbts[DataSource.CALIBRATED_FILE].isChecked() is True:
            data_source = DataSource.CALIBRATED_FILE
        elif self._data_src_rbts[DataSource.CALIBRATED].isChecked() is True:
            data_source = DataSource.CALIBRATED
        elif self._data_src_rbts[DataSource.ASSEMBLED].isChecked() is True:
            data_source = DataSource.ASSEMBLED
        else:
            data_source = DataSource.PROCESSED

        self.data_source_sgn.emit(data_source)

        pulse_range = (int(self._pulse_range0_le.text()),
                       int(self._pulse_range1_le.text()))
        if pulse_range[1] <= 0:
            logger.error("<Pulse range>: Invalid input!")
            return False
        else:
            self.pulse_range_sgn.emit(*pulse_range)

        server_hostname = self._hostname_le.text().strip()
        server_port = self._port_le.text().strip()
        self.server_tcp_sgn.emit(server_hostname, server_port)

        if log:
            logger.info("--- Shared parameters ---")
            logger.info("<Host name>, <Port>: {}, {}".
                        format(server_hostname, server_port))
            logger.info("<Pulse range>: ({}, {})".format(*pulse_range))

        return True
