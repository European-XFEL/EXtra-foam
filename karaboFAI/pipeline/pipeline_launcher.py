"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data processors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from queue import Queue, Empty, Full

from scipy import constants

from .image_assembler import ImageAssemblerFactory
from .data_aggregator import DataAggregator
from .data_model import ProcessedData
from .worker import Worker
from .processors import (
    AzimuthalIntegrationProcessor, CorrelationProcessor,
    PumpProbeProcessor, RoiProcessor, XasProcessor
)
from .exceptions import AggregatingError, AssemblingError, ProcessingError
from ..config import config
from ..gui import QtCore
from ..helpers import profiler


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
        self._data_aggregator = DataAggregator()

        self._roi_proc = RoiProcessor()
        self._roi_proc.setEnabled(True)
        self._correlation_proc = CorrelationProcessor()
        self._correlation_proc.setEnabled(True)

        self._ai_proc = AzimuthalIntegrationProcessor()

        self._pp_proc = PumpProbeProcessor()

        self._xas_proc = XasProcessor()

        self._tasks = [
            self._roi_proc,
            self._xas_proc,
            self._ai_proc,
            self._pp_proc,
            self._correlation_proc
        ]

    @QtCore.pyqtSlot(str)
    def onDetectorSourceChange(self, src):
        self._image_assembler.source_name = src

    @QtCore.pyqtSlot(int)
    def onSourceTypeChange(self, value):
        self._image_assembler.source_type = value

    @QtCore.pyqtSlot(str)
    def onXgmSourceChange(self, name):
        self._data_aggregator.xgm_src = name

    @QtCore.pyqtSlot(str)
    def onMonoSourceChange(self, name):
        self._data_aggregator.mono_src = name

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
        if mode != self._pp_proc.mode:
            self._pp_proc.mode = mode
            self._pp_proc.reset()
            ProcessedData.clear_onoff_hist()

        self._pp_proc.on_pulse_ids = on_pulse_ids
        self._pp_proc.off_pulse_ids = off_pulse_ids

    @QtCore.pyqtSlot(int)
    def onAbsDifferenceStateChange(self, state):
        self._pp_proc.abs_difference = state == QtCore.Qt.Checked

    @QtCore.pyqtSlot(float, float)
    def onAucXRangeChange(self, lb, ub):
        self._ai_proc.auc_x_range = (lb, ub)

    @QtCore.pyqtSlot(object)
    def onAiNormalizeChange(self, normalizer):
        self._ai_proc.normalizer = normalizer

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

    @QtCore.pyqtSlot(float, float)
    def onFomIntegrationRangeChange(self, lb, ub):
        self._pp_proc.fom_itgt_range = (lb, ub)
        self._correlation_proc.fom_itgt_range = (lb, ub)

    @QtCore.pyqtSlot()
    def onLaserOnOffClear(self):
        ProcessedData.clear_onoff_hist()
        self._pp_proc.reset()

    @QtCore.pyqtSlot(int)
    def onEnableAiStateChange(self, state):
        enabled = state == QtCore.Qt.Checked
        self._ai_proc.setEnabled(enabled)
        self._pp_proc.setEnabled(enabled)

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

    @QtCore.pyqtSlot(int)
    def onPumpProbeMAWindowChange(self, n):
        self._pp_proc.ma_window = n

    @QtCore.pyqtSlot(int)
    def onXasStateToggle(self, state):
        enabled = state == QtCore.Qt.Checked
        self._xas_proc.setEnabled(enabled)

    @QtCore.pyqtSlot(int)
    def onXasEnergyBinsChange(self, n):
        self._xas_proc.n_bins = n

    @QtCore.pyqtSlot()
    def onXasClear(self):
        self._xas_proc.reset()

    def update_roi_region(self, rank, activated, w, h, px, py):
        if activated:
            self._roi_proc.set(rank, (w, h, px, py))
        else:
            self._roi_proc.set(rank, None)

    def update_roi_fom(self, value):
        self._roi_proc.roi_fom = value
        ProcessedData.clear_roi_hist()

    def clear_roi_hist(self):
        ProcessedData.clear_roi_hist()

    def run(self):
        """Run the data processor."""
        self._running = True
        self.log("Data processor started!")
        while self._running:
            try:
                data = self._in_queue.get(timeout=config['TIMEOUT'])
            except Empty:
                continue

            processed_data = self._process(data)
            if processed_data is None:
                continue

            while self._running:
                try:
                    self._out_queue.put(processed_data,
                                        timeout=config['TIMEOUT'])
                    break
                except Full:
                    continue

        self.log("Data processor stopped!")

    @profiler("Process Data (total)")
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
        except AssemblingError as e:
            self.log(f"Train ID: {tid}: " + repr(e))
            return None
        except Exception as e:
            self.log(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
            raise

        try:
            processed_data = ProcessedData(tid, assembled)
        except Exception as e:
            self.log(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
            raise

        try:
            self._data_aggregator.aggregate(processed_data, data)
        except AggregatingError as e:
            self.log(f"Train ID: {tid}: " + repr(e))
        except Exception as e:
            self.log(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
            raise

        for task in self._tasks:
            try:
                if task.isEnabled():
                    task.process(processed_data, data)
            except ProcessingError as e:
                self.log(f"Train ID: {tid}: " + repr(e))
            except Exception as e:
                self.log(f"Unexpected Exception: Train ID: {tid}: " + repr(e))
                raise

        return processed_data
