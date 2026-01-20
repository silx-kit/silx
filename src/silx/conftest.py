import pytest
import logging
import os
from io import BytesIO

import h5py

logger = logging.getLogger(__name__)


def _set_qt_binding(binding):
    if binding is not None:
        binding = binding.lower()
        if binding == "pyqt5":
            logger.info("Force using PyQt5")
            import PyQt5.QtCore  # noqa
        elif binding == "pyside6":
            logger.info("Force using PySide6")
            import PySide6.QtCore  # noqa
        elif binding == "pyqt6":
            logger.info("Force using PyQt6")
            import PyQt6.QtCore  # noqa
        else:
            raise ValueError("Qt binding '%s' is unknown" % binding)


def pytest_addoption(parser):
    parser.addoption(
        "--qt-binding",
        type=str,
        default=None,
        dest="qt_binding",
        help="Force using a Qt binding: 'PyQt5', 'PySide6', 'PyQt6'",
    )
    parser.addoption(
        "--no-gui",
        dest="gui",
        default=True,
        action="store_false",
        help="Disable the test of the graphical use interface",
    )
    parser.addoption(
        "--no-opengl",
        dest="opengl",
        default=True,
        action="store_false",
        help="Disable tests using OpenGL",
    )
    parser.addoption(
        "--no-opencl",
        dest="opencl",
        default=True,
        action="store_false",
        help="Disable the test of the OpenCL part",
    )
    parser.addoption(
        "--high-mem",
        dest="high_mem",
        default=False,
        action="store_true",
        help="Enable tests with large memory consumption (>100Mbytes)",
    )


def pytest_configure(config):
    if not config.getoption("opencl", True):
        os.environ["SILX_OPENCL"] = "False"  # Disable OpenCL support in silx

    _set_qt_binding(config.option.qt_binding)


_FILTERWARNINGS = (
    r"ignore:tostring\(\) is deprecated\. Use tobytes\(\) instead\.:DeprecationWarning:OpenGL.GL.VERSION.GL_2_0",
    "ignore:Jupyter is migrating its paths to use standard platformdirs:DeprecationWarning",
    "ignore:Unable to import recommended hash 'siphash24.siphash13', falling back to 'hashlib.sha256'. Run 'python3 -m pip install siphash24' to install the recommended hash.:UserWarning:pytools.persistent_dict",
    "ignore:Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.:UserWarning",
    # Remove __array__ ignore once h5py v3.12 is released
    "ignore:__array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments.:DeprecationWarning",
    "ignore::pyopencl.RepeatedKernelRetrieval",
    # Deprecated pyparsing usage in matplotlib: https://github.com/matplotlib/matplotlib/issues/30617
    "ignore::DeprecationWarning:matplotlib._fontconfig_pattern",
    "ignore::DeprecationWarning:matplotlib._mathtext",
    "ignore::DeprecationWarning:pyparsing.util",
)


def pytest_collection_modifyitems(items):
    """Add warnings filters to all tests"""
    for item in items:
        item.add_marker(pytest.mark.filterwarnings("error"), append=False)
        for filter_string in _FILTERWARNINGS:
            item.add_marker(pytest.mark.filterwarnings(filter_string))


@pytest.fixture(scope="session")
def test_options(request):
    from .test import utils

    options = utils._TestOptions()
    options.configure(request.config.option)
    yield options


@pytest.fixture(scope="class")
def test_options_class_attr(request, test_options):
    """Provides test_options as class attribute

    Used as transition from TestCase to pytest
    """
    request.cls.test_options = test_options


@pytest.fixture(scope="session")
def use_opengl(test_options):
    """Fixture to flag test using a OpenGL.

    This can be skipped with `--no-opengl`.
    """
    if not test_options.WITH_GL_TEST:
        pytest.skip(test_options.WITH_GL_TEST_REASON, allow_module_level=True)


@pytest.fixture(scope="session")
def use_opencl(test_options):
    """Fixture to flag test using a OpenCL.

    This can be skipped with `--no-opencl`.
    """
    if not test_options.WITH_OPENCL_TEST:
        pytest.skip(test_options.WITH_OPENCL_TEST_REASON, allow_module_level=True)


@pytest.fixture(scope="session")
def use_large_memory(test_options):
    """Fixture to flag test using a large memory consumption.

    This can be enabled with `--high-mem`.
    """
    if not test_options.WITH_HIGH_MEM_TEST:
        pytest.skip(test_options.WITH_HIGH_MEM_TEST_REASON, allow_module_level=True)


@pytest.fixture(scope="session")
def use_gui(test_options):
    """Fixture to flag test using GUI.

    This can be skipped with `--no-gui`.
    """
    if not test_options.WITH_QT_TEST:
        pytest.skip(test_options.WITH_QT_TEST_REASON, allow_module_level=True)


@pytest.fixture(scope="session")
def qapp(use_gui, xvfb, request):
    _set_qt_binding(request.config.option.qt_binding)

    from silx.gui import qt

    app = qt.QApplication.instance()
    if app is None:
        app = qt.QApplication([])
    try:
        yield app
    finally:
        if app is not None:
            app.closeAllWindows()


@pytest.fixture
def qapp_utils(qapp):
    """Helper containing method to deal with QApplication and widget"""
    from silx.gui.utils.testutils import TestCaseQt

    utils = TestCaseQt()
    utils.setUpClass()
    utils.setUp()
    yield utils
    utils.tearDown()
    utils.tearDownClass()


@pytest.fixture
def qWidgetFactory(qapp, qapp_utils):
    """QWidget factory as fixture

    This fixture provides a function taking a QWidget subclass as argument
    which returns an instance of this QWidget making sure it is shown first
    and destroyed once the test is done.
    """
    from silx.gui import qt
    from silx.gui.qt.inspect import isValid

    widgets = []

    def createWidget(cls, *args, **kwargs):
        widget = cls(*args, **kwargs)
        widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        widget.show()
        qapp_utils.qWaitForWindowExposed(widget)
        widgets.append(widget)

        return widget

    yield createWidget

    qapp.processEvents()

    for widget in widgets:
        if isValid(widget):
            widget.close()
    qapp.processEvents()

    # Wait some time for all widgets to be deleted
    for _ in range(10):
        validWidgets = [widget for widget in widgets if isValid(widget)]
        if validWidgets:
            qapp_utils.qWait(10)

    validWidgets = [widget for widget in widgets if isValid(widget)]
    assert not validWidgets, f"Some widgets were not destroyed: {validWidgets}"

    # Make sure not to keep references on widgets
    widgets.clear()


@pytest.fixture
def tmp_h5py_file():
    with BytesIO() as buffer:
        with h5py.File(buffer, mode="w") as h5file:
            yield h5file
