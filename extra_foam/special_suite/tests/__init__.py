import abc
import unittest

from extra_foam.gui.plot_widgets import TimedPlotWidgetF, TimedImageViewF

from extra_foam.special_suite import logger, mkQApp


class _SpecialSuiteWindowTestBase(unittest.TestCase):
    @staticmethod
    def data4visualization():
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        cls._win = cls._window_type("TEST")
        # We need to wait until the worker is waiting for notifications before
        # continuing, otherwise a race condition is possible that would cause
        # the worker to miss the notification to stop, which would cause the
        # tests to hang.
        cls._win._worker_st._waiting_st.wait()

    @classmethod
    def tearDownClass(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    def _check_update_plots(self):
        win = self._win
        worker = win._worker_st

        with self.assertLogs(logger, level="ERROR") as cm:
            logger.error("dummy")  # workaround

            win.updateWidgetsST()  # with empty data

            worker._output_st.put_pop(self.data4visualization())
            win.updateWidgetsST()
            for widget in win._plot_widgets_st:
                if isinstance(widget, TimedPlotWidgetF):
                    widget.refresh()
            for widget in win._image_views_st:
                if isinstance(widget, TimedImageViewF):
                    widget.refresh()

            self.assertEqual(1, len(cm.output))


class _SpecialSuiteProcessorTestBase:
    @abc.abstractmethod
    def _check_processed_data_structure(self, processed):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_reset(self, proc):
        raise NotImplementedError
