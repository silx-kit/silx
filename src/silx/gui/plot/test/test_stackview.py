import numpy
import pytest

from ...plot.StackView import StackView
from ...plot.StackView import StackViewMainWindow
from ...utils.testutils import SignalListener
from ....utils.array_like import ListOfImages


@pytest.fixture()
def stack_data() -> numpy.ndarray:
    return numpy.fromfunction(
        lambda i, j, k: (
            numpy.sin(i / 15.0) + numpy.cos(j / 4.0) + 2 * numpy.sin(k / 6.0)
        ),
        (10, 20, 30),
    )


@pytest.fixture()
def stack_view(qWidgetFactory, qapp_utils) -> StackView:
    widget = qWidgetFactory(StackView)
    qapp_utils.qWaitForWindowExposed(widget)
    return widget


def testScaleColormapRangeToStack(stack_view, stack_data):
    """Test scaleColormapRangeToStack"""
    stack_view.setStack(stack_data)
    stack_view.setColormap("viridis")
    colormap = stack_view.getColormap()

    # Colormap autoscale to image
    assert colormap.getVRange() == (None, None)
    stack_view.scaleColormapRangeToStack()

    # Colormap range set according to stack range
    assert colormap.getVRange() == (stack_data.min(), stack_data.max())


def testSetStack(stack_view, stack_data):
    stack_view.setStack(stack_data)
    stack_view.setColormap("viridis")
    transposed_stack = stack_view.getData()
    assert transposed_stack.shape == stack_data.shape
    assert numpy.array_equal(stack_data, transposed_stack)

    colormap = stack_view.getColormap()
    assert colormap["name"] == "viridis"


def testSetStackPerspective(stack_view, stack_data):
    stack_view.setStack(stack_data, perspective=1)
    transposed_stack = stack_view.getCurrentData()

    # get stack returns the transposed data, depending on the perspective
    assert transposed_stack.shape == (
        stack_data.shape[1],
        stack_data.shape[0],
        stack_data.shape[2],
    )
    numpy.array_equal(numpy.transpose(stack_data, axes=(1, 0, 2)), transposed_stack)


def testSetStackListOfImages(stack_view, stack_data):
    list_of_images = [stack_data[i] for i in range(stack_data.shape[0])]

    stack_view.setStack(list_of_images)
    original_stack_data = stack_view.getData(returnNumpyArray=True)
    transposed_stack_data = stack_view.getData(returnNumpyArray=True)
    assert transposed_stack_data.shape == stack_data.shape
    assert numpy.array_equal(stack_data, transposed_stack_data)
    assert numpy.array_equal(stack_data, original_stack_data)
    assert isinstance(transposed_stack_data, numpy.ndarray)

    stack_view.setStack(list_of_images, perspective=2)
    # getData(copy=False) must return the object set in setStack
    original_stack_data = stack_view.getData(copy=False)
    assert original_stack_data is list_of_images
    # getData(copy=False) returns a ListOfImages whose .images is the original data
    transposed_stack_data = stack_view.getCurrentData(copy=False)
    assert transposed_stack_data.shape == (
        stack_data.shape[2],
        stack_data.shape[0],
        stack_data.shape[1],
    )
    assert numpy.array_equal(
        numpy.array(transposed_stack_data),
        numpy.transpose(stack_data, axes=(2, 0, 1)),
    )
    assert isinstance(transposed_stack_data, ListOfImages)
    assert transposed_stack_data.images is list_of_images


def testPerspective(stack_view):
    stack_view.setStack(numpy.arange(24).reshape((2, 3, 4)))
    assert stack_view.getPerspective() == 0, "Default perspective is not 0 (dim1-dim2)."

    stack_view.setPerspective(1)
    assert stack_view.getPerspective() == 1, (
        "Plane selection combobox not updating perspective"
    )

    stack_view.setStack(numpy.arange(6).reshape((1, 2, 3)))
    assert stack_view.getPerspective() == 1, (
        "Perspective not preserved when calling setStack without specifying the perspective parameter."
    )

    stack_view.setStack(numpy.arange(24).reshape((2, 3, 4)), perspective=2)
    assert stack_view.getPerspective() == 2, (
        "Perspective not set in setStack(..., perspective=2)."
    )


def testPlotTitleContainsZInfo(stack_view):
    stack_view.setStack(
        numpy.arange(24).reshape((4, 3, 2)),
        calibrations=[(0, 1), (-10, 10), (3.14, 3.14)],
    )
    assert stack_view.getGraphTitle() == "Image z=0"
    stack_view.setFrameNumber(2)
    assert stack_view.getGraphTitle() == "Image z=2"

    stack_view.setPerspective(1)
    stack_view.setFrameNumber(0)
    assert stack_view.getGraphTitle() == "Image z=-10"
    stack_view.setFrameNumber(2)
    assert stack_view.getGraphTitle() == "Image z=10"

    stack_view.setPerspective(2)
    stack_view.setFrameNumber(0)
    assert stack_view.getGraphTitle() == "Image z=3.14"
    stack_view.setFrameNumber(1)
    assert stack_view.getGraphTitle() == "Image z=6.28"


def testSetTitleCallback(stack_view, stack_data):
    """Test setting the plot title with a user defined callback"""
    stack_view.setStack(
        numpy.arange(24).reshape((4, 3, 2)),
        calibrations=[(0, 1), (-10, 10), (3.14, 3.14)],
    )

    def title_callback(frame_idx):
        return "Cubed index title %d" % (frame_idx**3)

    stack_view.setTitleCallback(title_callback)
    assert stack_view.getGraphTitle() == "Cubed index title 0"
    stack_view.setFrameNumber(2)
    assert stack_view.getGraphTitle() == "Cubed index title 8"

    # perspective should not matter, only frame index
    stack_view.setPerspective(1)
    stack_view.setFrameNumber(0)
    assert stack_view.getGraphTitle() == "Cubed index title 0"
    stack_view.setFrameNumber(2)
    assert stack_view.getGraphTitle() == "Cubed index title 8"

    with pytest.raises(TypeError):
        # setTitleCallback should not accept non-callable objects like strings
        stack_view.setTitleCallback("This is a string")


def testStackFrameNumber(stack_view, stack_data):
    stack_view.setStack(stack_data)
    assert stack_view.getFrameNumber() == 0

    listener = SignalListener()
    stack_view.sigFrameChanged.connect(listener)

    stack_view.setFrameNumber(1)
    assert stack_view.getFrameNumber() == 1
    assert listener.arguments() == [(1,)]


def testStackViewMainWindow(qWidgetFactory, qapp_utils, stack_data):
    stackview = qWidgetFactory(StackViewMainWindow)
    qapp_utils.qWaitForWindowExposed(stackview)
    stackview.setStack(stack_data)
    stackview.setColormap("viridis")
    transposed_stack = stackview.getData()
    assert transposed_stack.shape == stack_data.shape
    assert numpy.array_equal(stack_data, transposed_stack)

    colormap = stackview.getColormap()
    assert colormap["name"] == "viridis"

    stackview.setStack(stack_data, perspective=1)
    transposed_stack = stackview.getCurrentData()
    # get stack returns the transposed data, depending on the perspective
    assert transposed_stack.shape == (
        stack_data.shape[1],
        stack_data.shape[0],
        stack_data.shape[2],
    )
    assert numpy.array_equal(
        numpy.transpose(stack_data, axes=(1, 0, 2)), transposed_stack
    )
