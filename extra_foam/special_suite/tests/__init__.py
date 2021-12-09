import abc
import logging

import pytest

from extra_foam.gui.plot_widgets import TimedPlotWidgetF, TimedImageViewF
from extra_foam.special_suite import special_suite_logger_name, logger


class _SpecialSuiteWindowTestBase:
    @staticmethod
    def data4visualization():
        raise NotImplementedError

    @pytest.fixture
    def check_update_plots(self, win, caplog):
        def _check_update_plots():
            worker = win._worker_st

            caplog.set_level(logging.ERROR, logger=special_suite_logger_name)
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

            assert 1 == len(caplog.messages)

        return _check_update_plots


class _SpecialSuiteProcessorTestBase:
    @abc.abstractmethod
    def _check_processed_data_structure(self, processed):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_reset(self, proc):
        raise NotImplementedError
