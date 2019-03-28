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
    AzimuthalIntegrationProcessor, CorrelationProcessor, HeadProcessor,
    LaserOnOffProcessor, RegionOfInterestProcessor, SampleDegradationProcessor
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

        self._image_assembler = ImageAssemblerFactory.create(config['DETECTOR'])

        self._roi_proc = RegionOfInterestProcessor()
        self._correlation_proc = CorrelationProcessor()

        self._ai_proc = AzimuthalIntegrationProcessor()
        self._laser_on_off_proc = LaserOnOffProcessor()
        self._sample_degradation_proc = SampleDegradationProcessor()

        self._head = HeadProcessor()

    @QtCore.pyqtSlot(str)
    def onSourceNameChange(self, name):
        self._image_assembler.source_name = name

    @QtCore.pyqtSlot(object)
    def onSourceTypeChange(self, value):
        self._image_assembler.source_type = value

    @QtCore.pyqtSlot(str, list)
    def onGeometryChange(self, filename, quad_positions):
        success, info = self._image_assembler.load_geometry(
            filename, quad_positions)
        if not success:
            self.log(info)

    @QtCore.pyqtSlot(int, int)
    def onPulseIdRangeChange(self, lb, ub):
        self._image_assembler.pulse_id_range = (lb, ub)

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
        self._correlation_proc.normalizer = normalizer

    @QtCore.pyqtSlot(float)
    def onSampleDistanceChange(self, value):
        self._ai_proc.sample_distance = value

    @QtCore.pyqtSlot(int, int)
    def onIntegrationCenterChange(self, cx, cy):
        self._ai_proc.integration_center = (cx, cy)

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
        enabled = state == QtCore.Qt.Checked
        self._ai_proc.setEnabled(enabled)
        self._sample_degradation_proc.setEnabled(enabled)
        self._laser_on_off_proc.setEnabled(enabled)

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

    def update_roi_region(self, rank, activated, w, h, px, py):
        if activated:
            self._roi_proc.set_roi(rank, (w, h, px, py))
        else:
            self._roi_proc.set_roi(rank, None)

    def update_roi_fom(self, value):
        self._roi_proc.roi_fom = value
        ProcessedData.clear_roi_hist()

    def clear_roi_hist(self):
        ProcessedData.clear_roi_hist()

    def _build_graph(self):
        # TODO: define different graphs for different pipelines
        self._head.next = self._roi_proc
        self._roi_proc.next = self._ai_proc
        self._ai_proc.next = self._sample_degradation_proc
        self._sample_degradation_proc.next = self._laser_on_off_proc
        self._laser_on_off_proc.next = self._correlation_proc

    def run(self):
        """Run the data processor."""
        self._build_graph()
        self._running = True
        self.log("Data processor started!")
        while self._running:
            try:
                data = self._in_queue.get(timeout=config['TIMEOUT'])
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            processed_data = self._process(data)
            if processed_data is None:
                continue
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
        """Process data received from the bridge."""
        data, meta = data

        # get the train ID of the first metadata
        # Note: this is better than meta[src_name] because:
        #       1. For streaming AGIPD/LPD data from files, 'src_name' does
        #          not work;
        #       2. Prepare for the scenario where a 2D detector is not
        #          mandatory.
        tid = next(iter(meta.values()))["timestamp.tid"]

        try:
            assembled = self._image_assembler.assemble(data)
            processed_data = ProcessedData(tid, assembled)
        # Exception:
        #   - ValueError, IndexError, KeyError: raised by 'assemble'
        #   - ValueError, TypeError: raised by initialization of ProcessedData
        except (ValueError, IndexError, KeyError, TypeError) as e:
            self.log(f"Train ID: {tid}: " + repr(e))
            return None
        except Exception as e:
            self.log(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
            raise

        try:
            proc = self._head
            while True:
                if proc is None:
                    break

                if proc.isEnabled():
                    error_msg = proc.process(processed_data, data)
                    if error_msg:
                        self.log(f"Train ID: {tid}: " + error_msg)

                proc = proc.next
        except Exception as e:
            self.log(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
            raise

        return processed_data
