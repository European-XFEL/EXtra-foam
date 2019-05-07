"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Services.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse

from PyQt5.QtWidgets import QApplication

from zmq.error import ZMQError

from . import __version__
from .config import config
from .logger import logger
from .gui import MainGUI, Mediator
from .offline import FileServer
from .pipeline import Bridge, Scheduler


class FaiServer:
    """FaiServer class.

    TODO: change the class name.

    It manages all services in karaboFAI: QApplication, Redis, Processors,
    etc.
    """
    __app = None

    def __init__(self, detector):
        """Initialization."""

        self.qt_app()

        # update global configuration
        config.load(detector)

        # a zmq bridge which acquires the data in another thread
        self._bridge = Bridge()

        # a data processing worker which processes the data in another thread
        self._scheduler = Scheduler()
        self._scheduler.connect_input(self._bridge)

        # a file server which streams data from files
        self._file_server = None
        self._port = None
        self._data_folder = None

        # -------------------------------------------------------------
        # mediator for connections
        # -------------------------------------------------------------
        mediator = Mediator()

        # with the bridge
        mediator.bridge_endpoint_sgn.connect(self._bridge.onEndpointChange)

        # with the scheduler
        mediator.source_type_change_sgn.connect(
            self._scheduler.onSourceTypeChange)
        mediator.detector_source_change_sgn.connect(
            self._scheduler.onDetectorSourceChange)
        mediator.xgm_source_change_sgn.connect(
            self._scheduler.onXgmSourceChange)
        mediator.mono_source_change_sgn.connect(
            self._scheduler.onMonoSourceChange)
        mediator.pp_ma_window_change_sgn.connect(
            self._scheduler.onPumpProbeMAWindowChange)
        mediator.reset_xas_sgn.connect(self._scheduler.onXasReset)
        mediator.energy_bins_change_sgn.connect(
            self._scheduler.onXasEnergyBinsChange)
        mediator.roi_region_change_sgn.connect(
            self._scheduler.onRoiRegionChange)
        mediator.roi_fom_change_sgn.connect(self._scheduler.onRoiFomChange)
        mediator.roi_hist_clear_sgn.connect(self._scheduler.onRoiHistClear)

        mediator.pp_abs_difference_sgn.connect(
            self._scheduler.onPpDifferenceTypeChange)
        mediator.pp_analysis_type_sgn.connect(
            self._scheduler.onPpAnalysisTypeChange)

        mediator.correlation_fom_change_sgn.connect(
            self._scheduler.onCorrelationFomChange)

        # with the file server
        mediator.start_file_server_sgn.connect(self.start_fileserver)
        mediator.stop_file_server_sgn.connect(self.stop_fileserver)
        mediator.port_change_sgn.connect(self.onFileServerPortChange)
        mediator.data_folder_change_sgn.connect(self.onFileServerDataFolderChange)

        # -------------------------------------------------------------
        # MainGUI for karaboFAI
        # -------------------------------------------------------------
        self._gui = MainGUI()
        self._gui.connectInput(self._scheduler)

        if config['REQUIRE_GEOMETRY']:
            self._gui.geometry_ctrl_widget.geometry_sgn.connect(
                self._scheduler.onGeometryChange)

        self._gui.ai_ctrl_widget.photon_energy_sgn.connect(
            self._scheduler.onPhotonEnergyChange)
        self._gui.ai_ctrl_widget.sample_distance_sgn.connect(
            self._scheduler.onSampleDistanceChange)
        self._gui.ai_ctrl_widget.integration_center_sgn.connect(
            self._scheduler.onIntegrationCenterChange)
        self._gui.ai_ctrl_widget.integration_method_sgn.connect(
            self._scheduler.onIntegrationMethodChange)
        self._gui.ai_ctrl_widget.integration_range_sgn.connect(
            self._scheduler.onIntegrationRangeChange)
        self._gui.ai_ctrl_widget.integration_points_sgn.connect(
            self._scheduler.onIntegrationPointsChange)
        self._gui.ai_ctrl_widget.ai_normalizer_sgn.connect(
            self._scheduler.onAiNormalizeChange)
        self._gui.ai_ctrl_widget.auc_x_range_sgn.connect(
            self._scheduler.onAucXRangeChange)
        self._gui.ai_ctrl_widget.fom_integration_range_sgn.connect(
            self._scheduler.onFomIntegrationRangeChange)
        self._gui.ai_ctrl_widget.pulsed_ai_cb.stateChanged.connect(
            self._scheduler.onPulsedAiStateChange)

        self._gui.analysis_ctrl_widget.pulse_id_range_sgn.connect(
            self._scheduler.onPulseIdRangeChange)

        self._gui.pump_probe_ctrl_widget.pp_pulse_ids_sgn.connect(
            self._scheduler.onPpPulseStateChange)
        self._gui.pump_probe_ctrl_widget.reset_btn.clicked.connect(
            self._scheduler.onPumpProbeReset)

        self._gui.correlation_ctrl_widget.correlation_param_change_sgn.connect(
            self._scheduler.onCorrelationParamChange)
        self._gui.correlation_ctrl_widget.clear_btn.clicked.connect(
            self._scheduler.onCorrelationReset)

        self._gui.start_bridge_sgn.connect(self._bridge.start)
        self._gui.stop_bridge_sgn.connect(self.stop_bridge)
        self._gui.closed_sgn.connect(self.stop_bridge)
        self._gui.closed_sgn.connect(self.stop_scheduler)

        self._bridge.started.connect(self._gui.onBridgeStarted)
        self._bridge.finished.connect(self._gui.onBridgeStopped)

        # -------------------------------------------------------------
        # logging from threads
        # -------------------------------------------------------------
        self._bridge.log_on_main_thread(self._gui)
        self._scheduler.log_on_main_thread(self._gui)

    def stop_bridge(self):
        self._bridge.requestInterruption()
        self._bridge.quit()
        self._bridge.wait()

    def stop_scheduler(self):
        self._scheduler.requestInterruption()
        self._scheduler.quit()
        self._scheduler.wait()

    def start_fileserver(self):
        folder = self._data_folder
        port = self._port

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
        except ZMQError:
            logger.info("Port {} is already in use!".format(port))
            return

        Mediator().file_server_started_sgn.emit()

    def stop_fileserver(self):
        if self._file_server is not None and self._file_server.is_alive():
            self._file_server.terminate()

        if self._file_server is not None:
            self._file_server.join()

        print("File server stopped!")
        Mediator().file_server_stopped_sgn.emit()

    @classmethod
    def qt_app(cls):
        if cls.__app is None:
            import sys
            cls.__app = QApplication(sys.argv)
        return cls.__app.instance()

    def start(self):
        try:
            # scheduler is always running
            self._scheduler.start()

            self.qt_app().exec_()
        finally:
            self.stop_fileserver()
            self.stop_bridge()
            self.stop_scheduler()

    def onFileServerPortChange(self, port):
        self._port = port

    def onFileServerDataFolderChange(self, path):
        self._data_folder = path


def application():
    parser = argparse.ArgumentParser(prog="karaboFAI")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=['AGIPD', 'LPD', 'JUNGFRAU', 'FASTCCD',
                                 'BASLERCAMERA'],
                        type=lambda s: s.upper())
    parser.add_argument('--debug', action='store_true',
                        help="Enable faulthandler")

    args = parser.parse_args()

    if args.debug:
        import faulthandler
        faulthandler.enable()
        logger.debug("'faulthandler enabled")

    detector = args.detector
    if detector == 'JUNGFRAU':
        detector = 'JungFrau'
    elif detector == 'FASTCCD':
        detector = 'FastCCD'
    elif detector == 'BASLERCAMERA':
        detector = 'BaslerCamera'
    else:
        detector = detector.upper()

    server = FaiServer(detector)

    server.start()


if __name__ == "__main__":

    application()
