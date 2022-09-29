import pytest
import logging
import os


logger = logging.getLogger(__name__)


def _set_qt_binding(binding):
    if binding is not None:
        binding = binding.lower()
        if binding == "pyqt5":
            logger.info("Force using PyQt5")
            import PyQt5.QtCore  # noqa
        elif binding == "pyside2":
            logger.info("Force using PySide2")
            import PySide2.QtCore  # noqa
        elif binding == "pyside6":
            logger.info("Force using PySide6")
            import PySide6.QtCore  # noqa
        elif binding == "pyqt6":
            logger.info("Force using PyQt6")
            import PyQt6.QtCore  # noqa
        else:
            raise ValueError("Qt binding '%s' is unknown" % binding)


def pytest_addoption(parser):
    parser.addoption("--qt-binding", type=str, default=None, dest="qt_binding",
                     help="Force using a Qt binding: 'PyQt5', 'PySide2', 'PySide6', 'PyQt6'")
    parser.addoption("--no-gui", dest="gui", default=True,
                     action="store_false",
                     help="Disable the test of the graphical use interface")
    parser.addoption("--no-opengl", dest="opengl", default=True,
                     action="store_false",
                     help="Disable tests using OpenGL")
    parser.addoption("--no-opencl", dest="opencl", default=True,
                     action="store_false",
                     help="Disable the test of the OpenCL part")
    parser.addoption("--low-mem", dest="low_mem", default=False,
                     action="store_true",
                     help="Disable test with large memory consumption (>100Mbyte")


def pytest_configure(config):
    if not config.getoption('opencl', True):
        os.environ['SILX_OPENCL'] = 'False'  # Disable OpenCL support in silx

    _set_qt_binding(config.option.qt_binding)


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

    This can be skipped with `--low-mem`.
    """
    if test_options.TEST_LOW_MEM:
        pytest.skip(test_options.TEST_LOW_MEM_REASON, allow_module_level=True)


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
