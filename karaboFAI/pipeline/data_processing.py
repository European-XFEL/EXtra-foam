"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data processors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
import queue

from scipy import constants

from .image_assembler import ImageAssemblerFactory
from .data_model import ProcessedData
from .worker import Worker
from .data_processor import (
    RegionOfInterestProcessor, AzimuthalIntegrationProcessor,
    LaserOnOffProcessor, SampleDegradationProcessor, CorrelationProcessor
)
from ..config import config
from ..gui import QtCore
from ..logger import logger


class PipelineLauncher(Worker):
    """Facade class which managers different processing pipelines."""
    def __init__(self, in_queue, out_queue):
        """Initialization.

        :param Queue in_queue: a queue of data from the ZMQ bridge.
        :param Queue out_queue: a queue of processed data.
        """
        super().__init__()

        self._in_queue = in_queue
        self._out_queue = out_queue

        # whether to turn azimuthal integration on
        self._enable_ai = False

        self._image_assembler = ImageAssemblerFactory.create(config['DETECTOR'])

        self._roi_proc = RegionOfInterestProcessor(parent=self)
        self._ai_proc = AzimuthalIntegrationProcessor(parent=self)
        self._laser_on_off_proc = LaserOnOffProcessor(parent=self)
        self._sample_degradation_proc = SampleDegradationProcessor(parent=self)
        self._correlation_proc = CorrelationProcessor(parent=self)

    @QtCore.pyqtSlot(str)
    def onMessageReceived(self, msg):
        self.log(msg)

    @QtCore.pyqtSlot(str, list)
    def onGeometryChange(self, filename, quad_positions):
        success, info = self._image_assembler.load_geometry(
            filename, quad_positions)
        if not success:
            self.log(info)

    @QtCore.pyqtSlot(int, int)
    def onPulseIdRangeChange(self, lb, ub):
        self._image_assembler.pulse_id_range = (lb, ub)

    @QtCore.pyqtSlot(object)
    def onSourceTypeChange(self, value):
        self._image_assembler.source_type = value

    @QtCore.pyqtSlot(str)
    def onSourceNameChange(self, name):
        self._image_assembler.source_name = name

    @QtCore.pyqtSlot(object, list, list)
    def onOffPulseStateChange(self, mode, on_pulse_ids, off_pulse_ids):
        if mode != self._laser_on_off_proc.laser_mode:
            self._laser_on_off_proc.laser_mode = mode
            self._laser_on_off_proc.reset()
            ProcessedData.clear_onoff_hist()

        self._laser_on_off_proc.on_pulse_ids = on_pulse_ids
        self._laser_on_off_proc.off_pulse_ids = off_pulse_ids

    @QtCore.pyqtSlot(int)
    def onAbsDifferenceStateChange(self, state):
        self._laser_on_off_proc.abs_difference = state == QtCore.Qt.Checked

    @QtCore.pyqtSlot(int)
    def onMovingAverageWindowChange(self, value):
        self._laser_on_off_proc.moving_avg_window = value

    @QtCore.pyqtSlot(float, float)
    def onAucXRangeChange(self, lb, ub):
        self._laser_on_off_proc.auc_x_range = (lb, ub)
        self._sample_degradation_proc.auc_x_range = (lb, ub)
        self._correlation_proc.auc_x_range = (lb, ub)

    @QtCore.pyqtSlot(float, float)
    def onFomIntegrationRangeChange(self, lb, ub):
        self._laser_on_off_proc.fom_itgt_range = (lb, ub)
        self._sample_degradation_proc.fom_itgt_range = (lb, ub)
        self._correlation_proc.fom_itgt_range = (lb, ub)

    @QtCore.pyqtSlot(object)
    def onAiNormalizeChange(self, normalizer):
        self._laser_on_off_proc.normalizer = normalizer
        self._correlation_proc.normalizer = normalizer

    @QtCore.pyqtSlot(float)
    def onSampleDistanceChange(self, value):
        self._ai_proc.sample_distance = value

    @QtCore.pyqtSlot(int, int)
    def onPoniChange(self, cy, cx):
        self._ai_proc.poni = (cy, cx)

    @QtCore.pyqtSlot(str)
    def onIntegrationMethodChange(self, value):
        self._ai_proc.integration_method = value

    @QtCore.pyqtSlot(float, float)
    def onIntegrationRangeChange(self, lb, ub):
        self._ai_proc.integration_range = (lb, ub)

    @QtCore.pyqtSlot(int)
    def onIntegrationPointsChange(self, value):
        self._ai_proc.integration_points = value

    @QtCore.pyqtSlot(float)
    def onPhotonEnergyChange(self, photon_energy):
        """Compute photon wavelength (m) from photon energy (keV)."""
        # Plank-einstein relation (E=hv)
        HC_E = 1e-3 * constants.c * constants.h / constants.e
        self._ai_proc.wavelength = HC_E / photon_energy

    @QtCore.pyqtSlot()
    def onLaserOnOffClear(self):
        ProcessedData.clear_onoff_hist()
        self._laser_on_off_proc.reset()

    @QtCore.pyqtSlot(int)
    def onEnableAiStateChange(self, state):
        self._enable_ai = state == QtCore.Qt.Checked

    @QtCore.pyqtSlot()
    def onCorrelationClear(self):
        ProcessedData.clear_correlation_hist()

    @QtCore.pyqtSlot(int, str, str, float)
    def onCorrelationParamChange(self, idx, device_id, ppt, resolution):
        ProcessedData.add_correlator(idx, device_id, ppt, resolution)

    @QtCore.pyqtSlot(object)
    def onCorrelationFomChange(self, fom):
        if self._correlation_proc.fom_name != fom:
            self._correlation_proc.fom_name = fom
            ProcessedData.clear_correlation_hist()

    def update_roi1_region(self, activated, w, h, px, py):
        if activated:
            self._roi_proc.roi1 = (w, h, px, py)
        else:
            self._roi_proc.roi1 = None

    def update_roi2_region(self, activated, w, h, px, py):
        if activated:
            self._roi_proc.roi2 = (w, h, px, py)
        else:
            self._roi_proc.roi2 = None

    def update_roi_value_type(self, value):
        self._roi_proc.roi_value_type = value
        ProcessedData.clear_roi_hist()

    def clear_roi_hist(self):
        ProcessedData.clear_roi_hist()

    def register_processor(self, processor):
        processor.message_sgn.connect(self.onMessageReceived)

    def run(self):
        """Run the data processor."""
        self._running = True
        self.log("Data processor started!")
        while self._running:
            try:
                data = self._in_queue.get(timeout=config['TIMEOUT'])
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            processed_data = self._process(data)
            logger.debug("Time for data processing: {:.1f} ms in total!\n"
                         .format(1000 * (time.perf_counter() - t0)))

            while self._running:
                try:
                    self._out_queue.put(processed_data,
                                        timeout=config['TIMEOUT'])
                    break
                except queue.Full:
                    continue

            logger.debug("Size of in and out queues: {}, {}".format(
                self._in_queue.qsize(), self._out_queue.qsize()))

        self.log("Data processor stopped!")

    def _process(self, data):
        """Process data received from the bridge.

        data processing work flow:

        pre-processing -> retrieve ROI information -> perform azimuthal
        integration -> perform laser on-off analysis -> add correlation
        information
        """
        tid, assembled = self._image_assembler.assemble(data)

        if assembled is None:
            return ProcessedData(tid, assembled)

        processed_data = ProcessedData(tid, assembled)

        self._roi_proc.process(processed_data)

        if self._enable_ai:
            self._ai_proc.process(processed_data)
            self._sample_degradation_proc.process(processed_data)
            self._laser_on_off_proc.process(processed_data)

        # Process correlation after laser-on-off since correlation may
        # requires the laser-on-off result
        self._correlation_proc.process((processed_data, data))

        return processed_data
