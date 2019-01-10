"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data processor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import queue

import numpy as np

from h5py import File
import fabio

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from ..widgets.pyqtgraph import QtCore
from .data_model import DataSource, ProcessedData
from ..config import config
from ..logger import logger
from .proc_utils import nanmean_axis0_para
from ..worker import Worker


class COMDataProcessor(Worker):
    """Class for data processing.

    Attributes:
        source_sp (DataSource): data source.
        pulse_range_sp (tuple): (min, max) pulse ID to be processed.
            (int, int)
        geom_sp (LPDGeometry): geometry.
        mask_range_sp (tuple): (min, max), the pixel value outside
            the range will be clipped to the corresponding edge.
        image_mask (numpy.ndarray): a 2D mask.
    """
    def __init__(self, in_queue, out_queue):
        """Initialization.

        :param Queue in_queue: a queue of data from the ZMQ bridge.
        :param Queue out_queue: a queue of processed data
        """
        super().__init__()

        self._in_queue = in_queue
        self._out_queue = out_queue

        self.image_mask = None

        # shared parameters are updated by signal-slot
        # Note: shared parameters should end with '_sp'

        self.source_sp = None
        self.pulse_range_sp = None
        self.geom_sp = None
        self.mask_range_sp = None

    @QtCore.pyqtSlot(str)
    def onImageMaskChanged(self, filename):
        try:
            self.image_mask = fabio.open(filename).data
            msg = "Image mask {} loaded!".format(filename)
        except (IOError, OSError) as e:
            msg = str(e)
        finally:
            self.log(msg)

    @QtCore.pyqtSlot(object)
    def onSourceChanged(self, value):
        self.source_sp = value

    @QtCore.pyqtSlot(str, list)
    def onGeometryChanged(self, filename, quad_positions):
        if config['TOPIC'] == 'FXE':
            with File(filename, 'r') as f:
                self.geom_sp = LPDGeometry.from_h5_file_and_quad_positions(
                    f, quad_positions)
        elif config['TOPIC'] == 'SPB':
            try:
                from karabo_data.geometry2 import AGIPD_1MGeometry
            except (ImportError, ModuleNotFoundError):
                logger.debug(
                    "You are not in the correct branch for SPB experiment!")
                raise

            self.geom_sp = AGIPD_1MGeometry.from_crystfel_geom(filename).snap()

    @QtCore.pyqtSlot(float, float)
    def onMaskRangeChanged(self, lb, ub):
        self.mask_range_sp = (lb, ub)

    @QtCore.pyqtSlot(float)
    def onPhotonEnergyChanged(self, photon_energy):
        pass

    @QtCore.pyqtSlot(int, int)
    def onPulseRangeChanged(self, lb, ub):
        self.pulse_range_sp = (lb, ub)

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

            if self.source_sp == DataSource.CALIBRATED_FILE:
                processed_data = self.process_calibrated_data(
                    data, from_file=True)
            elif self.source_sp == DataSource.CALIBRATED:
                processed_data = self.process_calibrated_data(data)
            elif self.source_sp == DataSource.PROCESSED:
                processed_data = data[0]
            else:
                raise ValueError("Unknown data source!")

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

    def process_assembled_data(self, assembled, tid):
        """Process assembled image data.

        :param numpy.ndarray assembled: assembled image data.
        :param int tid: train ID

        :return ProcessedData: processed data.
        """
        # This needs to be checked. Sometimes throws an error readonly
        # when trying to convert nan to -inf. Dirty hack -> to copy
        if config["TOPIC"] == 'JungFrau':
            assembled = np.copy(assembled)

        # pre-processing

        t0 = time.perf_counter()

        # original data contains 'nan', 'inf' and '-inf' pixels
        assembled_mean = nanmean_axis0_para(assembled,
                                            max_workers=8, chunk_size=20)

        # Convert 'nan' to '-inf' and it will later be converted to the
        # lower range of mask, which is usually 0.
        # We do not convert 'nan' to 0 because: if the lower range of
        # mask is a negative value, 0 will be converted to a value
        # between 0 and 255 later.
        assembled_mean[np.isnan(assembled_mean)] = -np.inf

        logger.debug("Time for pre-processing: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        # clip the value in the array
        np.clip(assembled_mean, self.mask_range_sp[0], self.mask_range_sp[1],
                out=assembled_mean)
        # now 'assembled_mean' contains only numerical values within
        # the mask range

        # Note: 'assembled' still contains 'inf' and '-inf', we only do
        #       the clip later when necessary in order not to waste
        #       computing power.

        data = ProcessedData(tid,
                             image=assembled,
                             image_mean=assembled_mean,
                             image_mask=self.image_mask)

        return data

    def process_calibrated_data(self, calibrated_data, *, from_file=False):
        """Process calibrated data.

        :param tuple calibrated_data: (data, metadata). See return of
            KaraboBridge.Client.next().
        :param bool from_file: True for data streamed from files and False
            for data from the online ZMQ bridge.

        :return ProcessedData: processed data.
        """
        data, metadata = calibrated_data

        t0 = time.perf_counter()

        if from_file is False:
            if len(metadata.items()) > 1:
                logger.debug("Found multiple data sources!")

            tid = metadata[config["SOURCE_NAME"]]["timestamp.tid"]
            # Data coming from bridge in case of JungFrau will have
            # different key. To be included
            modules_data = data[config["SOURCE_NAME"]]["image.data"]

            if config["TOPIC"] == "FXE":
                # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
                modules_data = np.moveaxis(np.moveaxis(modules_data, 3, 0), 3, 2)
        else:
            tid = next(iter(metadata.values()))["timestamp.tid"]

            try:
                if config["TOPIC"] == "FXE":
                    modules_data = stack_detector_data(
                        data, "image.data", only='LPD')
                elif config['TOPIC'] == 'SPB':
                    modules_data = stack_detector_data(
                        data, "image.data", only='AGIPD')
                elif config["TOPIC"] == 'JungFrau':
                    source = next(iter(metadata.values()))["source"]
                    # stack_detector data at the moment doesn't support
                    # JungFrau detector because of different naming
                    # convention for source types.
                    modules_data = data[source]['data.adc'][:,np.newaxis,:,:]
                    # Add new axis which mimics module_number which at
                    # the moment is only 1. Once we will have stack
                    # detector data for JungFrau we will have
                    # required shape anyway (num_pulses, modules, y,x)

            # To handle a bug when using the recent karabo_data on the
            # old data set:
            # 1. Missing "image.data" will raise KeyError!
            # 2. Different modules could have different shapes, e.g.
            #    a train with 32 pulses could has a module with shape
            #    (4, 256, 256), which means the data for some pulses
            #    were lost. It will raise ValueError!
            #
            # Note: we log the information in 'debug' since otherwise it
            #       will go to the log window and cause problems like
            #       segmentation fault.
            except (KeyError, ValueError) as e:
                logger.debug("Error in stacking detector data: " + str(e))
                return ProcessedData(tid)

        logger.debug("Time for moveaxis/stacking: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        if config["TOPIC"] == "FXE":
            expected_shape = (16, 256, 256)
        elif config['TOPIC'] == 'SPB':
            expected_shape = (16, 512, 128)
        elif config['TOPIC'] == 'JungFrau':
            expected_shape = (1, 512, 1024)

        if hasattr(modules_data, 'shape') is False \
                or modules_data.shape[-3:] != expected_shape:
            logger.debug("Error in modules data of train {}".format(tid))
            return ProcessedData(tid)

        t0 = time.perf_counter()
        if config["TOPIC"] == "FXE" or config["TOPIC"] == "SPB":
            assembled, centre = self.geom_sp.position_all_modules(modules_data)
        elif config["TOPIC"] == "JungFrau":
            # Just for the time-being to be consistent with other
            # detector types.
            # Will have some kind of assembly/stacking in case of 2 modules
            assembled = np.squeeze(modules_data, axis=1)

        # This is a bug in old version of karabo_data. The above function
        # could return a numpy.ndarray with shape (0, x, x)
        if assembled.shape[0] == 0:
            logger.debug("Bad shape {} in assembled image of train {}".
                         format(assembled.shape, tid))
            return ProcessedData(tid)

        assembled = assembled[
            self.pulse_range_sp[0]:self.pulse_range_sp[1] + 1]

        logger.debug("Time for assembling: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        return self.process_assembled_data(assembled, tid)
