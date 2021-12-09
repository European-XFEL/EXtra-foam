import pytest

@pytest.fixture
def win(request):
    try:
        klass = request.module.window_type
    except AttributeError:
        raise AttributeError("Modules that use this fixture need to define a 'window_type' variable")

    window = klass("TEST")

    # We need to wait until the worker is waiting for notifications before
    # continuing, otherwise a race condition is possible that would cause
    # the worker to miss the notification to stop, which would cause the
    # tests to hang.
    window._worker_st._waiting_st.wait()

    yield window

    # Explicitly close the MainGUI to avoid error in GuiLogger
    window.close()
