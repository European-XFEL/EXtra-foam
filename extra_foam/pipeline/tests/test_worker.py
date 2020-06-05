import unittest
from unittest.mock import MagicMock, patch
import multiprocessing as mp

from extra_foam.pipeline.exceptions import ProcessingError, StopPipelineError
from extra_foam.pipeline.f_worker import TrainWorker, PulseWorker
from extra_foam.config import config


@patch.dict(config._data, {"DETECTOR": "LPD"})
class TestWorker(unittest.TestCase):
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
