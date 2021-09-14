import unittest
from threading import Thread
from unittest.mock import MagicMock, patch
import multiprocessing as mp

from . import _TestDataMixin
from extra_foam.pipeline.exceptions import ProcessingError, StopPipelineError
from extra_foam.pipeline.f_worker import TrainWorker, PulseWorker
from extra_foam.config import config, ExtensionType
from extra_foam.pipeline.f_zmq import FoamZmqClient

import numpy as np
from karabo_bridge import Client


@patch.dict(config._data, {"DETECTOR": "LPD"})
class TestWorker(_TestDataMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._pause_ev = mp.Event()
        cls._close_ev = mp.Event()

    @patch('extra_foam.ipc.ProcessLogger.debug')
    @patch('extra_foam.ipc.ProcessLogger.error')
    def testRunTasks(self, error, debug):
        for kls in (TrainWorker, PulseWorker):
            worker = kls(self._pause_ev, self._close_ev)
            for proc in worker._tasks:
                proc.update = MagicMock()
                proc.process = MagicMock()
            worker._run_tasks({})

            proc = worker._tasks[0]

            # test responses to different Exceptions

            proc.process.side_effect = ValueError()
            worker._run_tasks({})
            debug.assert_called_once()
            self.assertIn("Unexpected Exception", debug.call_args_list[0][0][0])
            debug.reset_mock()
            error.assert_called_once()
            error.reset_mock()

            proc.process.side_effect = ProcessingError()
            worker._run_tasks({})
            debug.assert_called_once()
            self.assertNotIn("Unexpected Exception", debug.call_args_list[0][0][0])
            debug.reset_mock()
            error.assert_called_once()
            error.reset_mock()

            proc.process.side_effect = StopPipelineError()
            with self.assertRaises(StopPipelineError):
                worker._run_tasks({})
            debug.reset_mock()
            error.reset_mock()

            # Check that the extensions are enabled appropriately
            extensions_enabled = kls == TrainWorker
            self.assertEqual(worker._extension != None, extensions_enabled)
            self.assertEqual(worker._detector_extension != None, extensions_enabled)

    @patch('extra_foam.ipc.ProcessLogger.debug')
    def testExtensions(self, _):
        worker = TrainWorker(self._pause_ev, self._close_ev)

        # Disable processors
        worker._run_tasks = MagicMock()

        # Generate mock data
        mock_data = self.simple_data(1, (10, 10))[0]
        detector, key = mock_data["catalog"].main_detector.split()

        # Mock the input and output pipes
        worker._input.start = MagicMock()
        worker._input.get = MagicMock(return_value=mock_data)
        worker._output = MagicMock()

        # Mock the database configuration for the extension ZmqOutQueue's
        extension_endpoint = "ipc://foam-extension"
        detector_extension_endpoint = "ipc://bridge-extension"
        worker._extension._meta.hget_all = MagicMock(return_value={ ExtensionType.ALL_OUTPUT.value: extension_endpoint })
        worker._detector_extension._meta.hget_all = MagicMock(return_value={ ExtensionType.DETECTOR_OUTPUT.value: detector_extension_endpoint })

        # Start worker
        self._pause_ev.set()
        worker_thread = Thread(target=worker.run)
        worker_thread.start()

        # Create clients
        bridge_client = Client(detector_extension_endpoint, timeout=1)
        foam_client = FoamZmqClient(extension_endpoint, timeout=1)

        # Test received detector data
        detector_data, _ = bridge_client.next()
        np.testing.assert_array_equal(detector_data[f"EF_{detector}"][key],
                                      mock_data["processed"].image.masked_mean)

        # Test received special suite data
        foam_data = foam_client.next()
        for key in foam_data:
            if key != "processed":
                self.assertEqual(foam_data[key], mock_data[key])
            else:
                # Just comparing the detector image is enough for the
                # ProcessedData object.
                np.testing.assert_array_equal(foam_data[key].image.masked_mean,
                                              mock_data[key].image.masked_mean)

        # Close worker
        self._close_ev.set()
        worker_thread.join(timeout=1)
