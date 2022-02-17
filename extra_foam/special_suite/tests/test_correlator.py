import time
import tempfile
import textwrap
from unittest.mock import patch, ANY

import pytest
import numpy as np
from metropc.client import ViewOutput
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTabBar, QToolButton

from . import _SpecialSuiteProcessorTestBase
from ...utils import rich_output, Series as S
from ..special_analysis_base import ClientType, _SpecialAnalysisBase
from ...pipeline.tests import _TestDataMixin
from ..correlator_w import CorrelatorWindow, SplitDirection, ViewWidget
from ..correlator_proc import CorrelatorProcessor, MetroEvent


window_type = CorrelatorWindow

class TestCorrelatorWindow:
    # Dummy data to use for testing plotting
    scalar_data = np.random.rand(10)
    vector_data = [np.random.rand(20) for _ in range(10)]

    @pytest.fixture
    def initial_context(self, win):
        """
        Initialize the suite with simple contexts for all major view types.
        """
        widget = win._tab_widget.widget(1).widget(0)

        # Set a new context
        ctx = """\
        @View.Compute
        def compute(tid: "internal#train_id"):
            return tid

        @View.Scalar
        def scalar(tid: "internal#train_id"):
            return tid

        @View.Vector
        def vector(tid: "internal#train_id"):
            return np.array([tid] * 10)

        @View.Image
        def image(_: "internal#train_id"):
            return np.random.rand(20, 20)
        """
        ctx = textwrap.dedent(ctx)
        win._editor.setText(ctx)
        # Mark it as saved so that we don't get a pop-up when the suite exits
        win._markContextSaved(True)
        win._ctrl_widget_st.reload_btn.clicked.emit()
        win._worker_st.waitUntil(MetroEvent.INDEX)

        # Wait for the views to get updated
        spy = QSignalSpy(widget.views_updated_sgn)
        spy.wait()

        return win._worker_st._ctx

    def testContextManagement(self, win):
        ctx = """\
        @View.Scalar
        def foo(tid: "internal#train_id"):
            return tid
        """
        ctx = textwrap.dedent(ctx)

        with tempfile.NamedTemporaryFile() as ctx_file:
            # Save the context to a file
            ctx_file.write(ctx.encode())
            ctx_file.flush()
            path = ctx_file.name

            # Helper function to read the current contents of the context file
            def saved_ctx():
                ctx_file.seek(0)
                return ctx_file.read().decode()

            with patch.object(QFileDialog, "getOpenFileName", return_value=(path, )):
                # Open the context file
                win._openContext()

                # Check the path and source is displayed correctly
                assert win._ctrl_widget_st._path_label.text() == path
                assert win._editor.text() == ctx

            # Change the context
            ctx += "\n# Hello world"
            win._editor.setText(ctx)

            # Trying to close without saved changes should cause a warning
            # message. Clicking the 'Cancel' button should do nothing.
            event = QCloseEvent()
            with patch.object(QMessageBox, "exec", return_value=QMessageBox.Cancel):
                win.closeEvent(event)
                assert not event.isAccepted()

            # The 'No' button should exit without saving
            with (patch.object(QMessageBox, "exec", return_value=QMessageBox.No),
                  patch.object(win._worker_st, "close") as worker_close,
                  # Annoyingly, we have to explicitly specify
                  # _SpecialAnalysisBase instead of using super() because the
                  # use of the create_special() decorator means that the type of
                  # CorrelatorWindow is actually a function.
                  patch.object(_SpecialAnalysisBase, "closeEvent") as window_close):
                win.closeEvent(event)

                worker_close.assert_called_once()
                window_close.assert_called_once()

                assert saved_ctx() != ctx

            # The 'Yes' button should save before exiting
            with (patch.object(QMessageBox, "exec", return_value=QMessageBox.Yes),
                  patch.object(win._worker_st, "close") as worker_close,
                  patch.object(_SpecialAnalysisBase, "closeEvent") as window_close):
                win.closeEvent(event)

                worker_close.assert_called_once()
                window_close.assert_called_once()

                assert saved_ctx() == ctx

            # Change the context again
            ctx += "\n # Spam and eggs"
            win._editor.setText(ctx)

            # Test the save button
            win._ctrl_widget_st.save_btn.clicked.emit()
            assert saved_ctx() == ctx

        # Test the reload button
        with patch.object(win._worker_st, "setContext") as worker_set_context:
            win._ctrl_widget_st.reload_btn.clicked.emit()
            worker_set_context.assert_called_once()

    def testTabs(self, win):
        tab_widget = win._tab_widget
        tab_bar = tab_widget.tabBar()

        # Helper function to find a QToolButton by name for a specific tab
        def get_tab_toolbutton(idx, name):
            tab_btn = tab_bar.tabButton(idx, QTabBar.RightSide)
            return tab_btn.findChild(QToolButton, name)

        new_tab_btn_idx = lambda: tab_bar.tabAt(win._new_tab_btn.pos())

        # We start with three tabs: one for the editor, one view tab, and a fake
        # tab for the button to add tabs.
        assert tab_widget.count() == 3

        # We should be able to delete the first view tab
        close_btn = get_tab_toolbutton(1, "close_btn")
        close_btn.clicked.emit()
        assert tab_widget.count() == 2

        # Add a tab
        win._new_tab_btn.clicked.emit()
        assert tab_widget.count() == 3
        # The new tab should be the current one
        assert tab_widget.currentIndex() == 1
        # The tab for the 'New Tab' button should be at the end
        assert new_tab_btn_idx() == 2

        # Now let's try undocking a tab
        splitter = tab_widget.widget(1)
        undock_btn = get_tab_toolbutton(1, "undock_btn")
        undock_btn.clicked.emit()
        # It should be removed from the tab widget
        assert tab_widget.count() == 2
        # The active tab should be the editor (not the tab for the 'New Tab' button)
        assert tab_widget.currentIndex() == 0
        # The widget should become its own window
        assert splitter.isWindow()

        # Now we dock it again
        splitter.closeEvent(QCloseEvent())
        # It should be added back to the tab widget
        assert not splitter.isWindow()
        assert tab_widget.indexOf(splitter) == 1
        # The 'New Tab' button tab should be at the end
        assert new_tab_btn_idx() == 2

    def testUberSplitter(self, win):
        # Note: we don't bother checking docking/undocking behaviour (even
        # though it's partly implemented in UberSplitter) because that's tested
        # in testTabs().
        splitter = win._tab_widget.widget(1)
        widgets = splitter._widgets

        # By default the top-level QSplitter is oriented vertically
        assert splitter.orientation() == Qt.Vertical

        # We start off with a single widget, so its delete button must be
        # disabled.
        assert len(widgets) == 1
        assert not widgets[0].delete_btn.isEnabled()

        # Split below.
        # New state:
        # [*,
        #  +]
        # Legend: '*' == existing widget, '+' == new widget
        splitter._split(widgets[0], SplitDirection.BELOW)
        assert len(widgets) == 2
        # Because we added a split in a vertical direction, it should have been
        # added to the top-level QSplitter.
        assert splitter.count() == 2
        # The original widget should be at the top
        assert splitter.indexOf(widgets[0]) == 0
        # And the new one at the bottom
        assert splitter.indexOf(widgets[-1]) == 1

        # Now we've got two widgets so the delete button for the first widget
        # should be enabled again.
        assert widgets[0].delete_btn.isEnabled()

        # Split above.
        # New state:
        # [+,
        #  *,
        #  *]
        splitter._split(widgets[0], SplitDirection.ABOVE)
        assert len(widgets) == 3
        # Again, because this is split vertically it should be added to the
        # top-level QSplitter.
        assert splitter.count() == 3
        # The new widget should be at the top
        assert splitter.indexOf(widgets[-1]) == 0

        # Split right.
        # New state:
        # [[*, +],
        #   *,
        #   *   ]
        splitter._split(splitter.widget(0), SplitDirection.RIGHT)
        assert len(widgets) == 4
        # Now the split is horizontal, the new widget should be added to a child
        # QSplitter, so the top-level QSplitter should remain at three children.
        assert splitter.count() == 3
        hsplitter = splitter.widget(0)
        assert hsplitter.count() == 2
        assert hsplitter.orientation() == Qt.Horizontal
        assert hsplitter.indexOf(widgets[-1]) == 1

        # Split left.
        # New state:
        # [[*, +, *],
        #      *,
        #      *   ]
        splitter._split(widgets[-1], SplitDirection.LEFT)
        assert len(widgets) == 5
        assert hsplitter.count() == 3
        assert hsplitter.indexOf(widgets[-1]) == 1

        # Delete the first widget
        # New state:
        # [[*, *, *],
        #      *   ]
        widgets[0].delete_btn.clicked.emit()
        # For now we can't do much more than this when it comes to testing that
        # the widget is actually deleted from the splitter, because
        # deleteLater() is used to delete the widgets, and that is only executed
        # by Qt in its event loop (which this test suite doesn't yet bother
        # creating).
        assert len(widgets) == 4
        # This won't work because the event loop isn't running:
        # assert splitter.count() == 2

        # Delete all but one widget
        while len(widgets) > 1:
            widgets[-1].delete_btn.clicked.emit()

        # Now that we're back to only one widget, its delete button should be
        # disabled again.
        assert not widgets[0].delete_btn.isEnabled()

    def testViewWidget(self, win, initial_context):
        widget = win._tab_widget.widget(1).widget(0)
        plot_widget = widget._plot_widget
        view_picker = widget.view_picker
        view_picker_widget = view_picker.parent()
        assert type(widget) == ViewWidget

        # Four views, plus an empty item
        assert view_picker.count() == 5, "Unexpected number of ViewWidget options"

        # The initial widget displayed should be the view picker
        assert widget.currentWidget() == view_picker_widget

        # Selecting the image view should show the image view widget, everything
        # else should show the plot widget.
        view_picker.setCurrentText("view#compute")
        assert widget.currentWidget() == plot_widget, "Wrong widget displayed for View.Compute"
        view_picker.setCurrentText("view#scalar")
        assert widget.currentWidget() == plot_widget, "Wrong widget displayed for View.Scalar"
        view_picker.setCurrentText("view#vector")
        assert widget.currentWidget() == plot_widget, "Wrong widget displayed for View.Vector"
        view_picker.setCurrentText("view#image")
        assert widget.currentWidget() == widget._image_widget, "Wrong widget displayed for View.Image"

        # Hitting the back button should get us back to the view picker
        widget._back_action.triggered.emit()
        assert widget.currentWidget() == view_picker_widget

    @pytest.mark.parametrize("view_type, output_data", [(ViewOutput.COMPUTE, scalar_data),
                                                        (ViewOutput.SCALAR, scalar_data),
                                                        (ViewOutput.VECTOR, vector_data)])
    def test1dPlotting(self, view_type, output_data, win, initial_context, caplog):
        """
        Test plotting Compute's, Scalar's, and Vector's. Note that we don't test
        Points views because those are a bit... strange... and I can't be
        bothered adding support for them when it's already possible to plot
        points with rich_output(). Image's are tested elsewhere because those
        are treated quite differently from 1D views.
        """
        widget = win._tab_widget.widget(1).widget(0)
        view_picker = widget.view_picker
        plot_widget = widget._plot_widget

        is_vector = view_type == ViewOutput.VECTOR
        generate_x_data = lambda: np.random.rand(len(output_data[0])) if is_vector else np.random.rand()
        view_name = f"view#{str(view_type).split('.')[1].lower()}"

        with patch.object(plot_widget, "setTitle") as setTitle:
            view_picker.setCurrentText(view_name)

            # The title should be set to the view name by default
            setTitle.assert_called_with(view_name)

        for train_output in output_data:
            widget.updateF({ view_name: [train_output] })

        # Compute views only support rich_output()
        if view_type == ViewOutput.COMPUTE:
            assert "Only rich_output() is supported" in caplog.messages[-1]
        else:
            # For all others, the 'y0' series should automatically be created
            assert "y0" in widget._ys

        # Resetting should clear everything
        widget._reset_action.triggered.emit()
        assert len(widget._xs) == 0
        assert len(widget._ys["y0"]) == 0

        # Now we try again with rich_output()
        max_points = len(output_data) // 2
        with (patch.object(plot_widget, "setLabel") as setLabel,
              patch.object(plot_widget, "setTitle") as setTitle):
            for y in output_data:
                output = rich_output(y, xlabel="Foo", ylabel="Bar", title="Baz",
                                     max_points=max_points)
                widget.updateF({ view_name: [output] })

            # The labels and title should have been set
            setLabel.assert_any_call("bottom", "Foo")
            setLabel.assert_any_call("left", "Bar")
            setTitle.assert_called_with("Baz")

            # There should be 'max_points' points
            assert len(widget._xs) == max_points
            assert len(widget._ys["y0"]) == max_points

        widget.reset()

        # And once more with multiple series
        for y in output_data:
            x = generate_x_data()
            output = rich_output(x,
                                 y1=S(y, name="Foo", error=0.1 * y),
                                 y2=S(y, name="Bar", error=0.2 * y))
            widget.updateF({ view_name: [output] })

        # For vectors, the plot data is replaced on every train, so what we
        # need to compare against is the length of the output per-train
        # rather than how many trains in total have been processed.
        output_len = len(output_data[0]) if is_vector else len(output_data)

        assert len(widget._xs) == output_len
        for series in ["Foo", "Bar"]:
            assert len(widget._ys[series]) == output_len
            assert len(widget._errors[series]) == output_len

            # Each series should have its own label
            assert series == widget._legend.getLabel(widget._plots[series]).text

        # Removing a series should also delete it from the plot
        x = generate_x_data()
        output = rich_output(x, y1=S(output_data[0], name="Foo"),
                             y2=S(output_data[0], name="Baz"))
        widget.updateF({ view_name: [output] })
        assert "Bar" not in widget._ys
        assert "Bar" not in widget._errors

        # If a name is not given, the series name should default to the keyword
        # argument.
        output = rich_output(x, y42=S(output_data[0]))
        widget.updateF({ view_name: [output] })
        assert "Foo" not in widget._ys
        assert "y42" in widget._ys

    def testImagePlotting(self, win, initial_context, caplog):
        widget = win._tab_widget.widget(1).widget(0)
        view_picker = widget.view_picker
        image_view = widget._image_view
        view_name = "view#image"

        with patch.object(image_view, "setTitle") as set_title:
            view_picker.setCurrentText(view_name)

            # The title should be set to the view name by default
            set_title.assert_called_with(view_name)

        # Sending data that isn't 2D should fail
        output = np.random.rand(10)
        widget.updateF({ view_name: [output] })
        assert "wrong number of dimensions" in caplog.messages[-1]

        output = np.random.rand(10, 10)
        with patch.object(image_view, "setImage") as set_image:
            widget.updateF({ view_name: [output] })
            set_image.assert_called_with(output)


class TestCorrelatorProcessor(_TestDataMixin, _SpecialSuiteProcessorTestBase):
    digitizer = "MID_EXP_FASTADC/ADC/DESTEST:channel_1.output"
    digitizer_property = "data.peaks"

    @pytest.fixture
    def proc(self):
        processor = CorrelatorProcessor(None, None)
        processor._pipeline.wait_till_ready()

        yield processor

        processor.close()

    def generateContext(self, with_raw=False):
        ctx = """\
        import numpy as np

        @View.Scalar
        def train_id(tid: "internal#train_id"):
            return tid

        @View.Scalar
        def xgm(data: "foam#pulse.xgm.intensity"):
            return data[0]
        """

        if with_raw:
            ctx += f"""\

        @View.Scalar
        def digitizer(data: "karabo#{self.digitizer}[{self.digitizer_property}]"):
            return np.nanmean(data)
        """

        return textwrap.dedent(ctx)

    def testProcess(self, proc):
        # Helper function to send data to the processor and get the outputs
        def send_trains(tids, with_digitizer=False):
            outputs = []
            digitizer_arg = [(self.digitizer, self.digitizer_property)] if with_digitizer else []

            # Stream all the inputs at once
            for tid in tids:
                data, _ = self.data_with_assembled(tid, (10, 10), with_xgm=True,
                                                   with_fast_data=digitizer_arg)
                outputs.append(proc.process(data))
                # Wait for a realistic amount of time to let the results
                # propagate. Ideally we would wait for some kind of Event
                # instead, but this is mildly tricky to implement so for now we
                # take the lazy approach (it's possible that this will fail
                # randomly in CI, in which case it should be fixed properly).
                time.sleep(0.1)

            outputs.append(proc.getOutputs())

            # Return all the actual outputs
            return outputs[1:]

        # Start by configuring the processor to read data from EXtra-foam
        proc.client_type = ClientType.EXTRA_FOAM

        # Load context
        train_view = "view#train_id"
        xgm_view = "view#xgm"
        proc.setContext(self.generateContext())

        # Subscribe to the view and wait a bit to ensure that metropc records
        # the subscription by the time we start pushing data.
        proc._subscriber.subscribe(train_view.encode())
        proc._subscriber.subscribe(xgm_view.encode())
        proc.waitUntil(MetroEvent.INDEX)

        tids = list(range(10))
        outputs = send_trains(tids)

        # The others should all contain data
        output_tids = [output[train_view][0] for output in outputs]
        assert output_tids == tids, "Not all outputs have been received"

        # Because the client type is configured to be EXtra-foam, the xgm view
        # should also have been processed.
        xgm_outputs = [output[xgm_view][0] for output in outputs]
        assert len(xgm_outputs) == len(tids), "View that uses EXtra-foam data has not been processed"

        # Add a new view, this time we let it automatically be subscribed to
        proc.setContext(self.generateContext(with_raw=True))
        proc.waitUntil(MetroEvent.INDEX)

        digitizer_view = "view#digitizer"
        assert digitizer_view in proc._subscriptions.keys(), "New view has not been automatically subscribed to"

        # Set the processor to read data from a Karabo bridge
        proc.client_type = ClientType.KARABO_BRIDGE

        # Now we'll send some data again, but this time the data from EXtra-foam
        # should be ignored.
        outputs = send_trains(range(10, 20), with_digitizer=True)

        # The view using the bridge data ought to have been executed
        assert all(digitizer_view in output for output in outputs), "View using bridge data not executed completely in bridge mode"

        # And the view using the EXtra-foam data should not have been
        assert not any(xgm_view in output for output in outputs), "View using EXtra-foam data should not have been executed in bridge mode"

        # Now we go back to reading from EXtra-foam
        proc.client_type = ClientType.EXTRA_FOAM

        # Send another bunch of trains
        outputs = send_trains(range(20, 30), with_digitizer=True)

        # This time both the view using EXtra-foam data and the 'raw' bridge
        # data should be executed.
        assert all(digitizer_view in output for output in outputs), "View using bridge data not executed completely in EXtra-foam mode"
        assert all(xgm_view in output for output in outputs), "View using EXtra-foam data not executed completely in EXtra-foam mode"

        # Using the BOTH client type doesn't make sense
        proc.client_type = ClientType.BOTH
        with pytest.raises(RuntimeError):
            proc.process({ })

    def testMonitor(self, proc):
        assert len(proc._subscriptions) == 0, "Uninitialized pipeline has subscriptions"

        # Add some views
        proc.setContext(self.generateContext())
        proc.waitUntil(MetroEvent.INDEX)
        assert len(proc._subscriptions) == 2, "Initial views are not subscribed to"

        # Add another view
        proc.setContext(self.generateContext(with_raw=True))
        proc.waitUntil(MetroEvent.INDEX)
        assert len(proc._subscriptions) == 3, "View subscriptions lost when adding a view"

        # Clear context
        proc.setContext("")
        proc.waitUntil(MetroEvent.INDEX)
        assert len(proc._subscriptions) == 0, "View subscriptions are not removed"
