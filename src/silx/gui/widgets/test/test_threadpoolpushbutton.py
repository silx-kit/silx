import time
import pytest
from silx.gui import qt

from ..ThreadPoolPushButton import ThreadPoolPushButton
from ...utils.testutils import SignalListener
from ....utils.testutils import LoggingValidator


@pytest.fixture()
def button(qWidgetFactory):
    yield qWidgetFactory(ThreadPoolPushButton)


@pytest.fixture()
def listener():
    listener = SignalListener()
    yield listener
    listener.clear()


def testExecute(button, qapp_utils):
    result = []

    def appendToResult(name):
        result.append(name)

    button.setCallable(appendToResult, "a")
    button.executeCallable()
    qapp_utils.waitUntil(lambda: not button.hasPendingOperations())
    assert result == ["a"]


def testMultiExecution(button, qapp_utils):
    result = []

    def appendToResult(name):
        result.append(name)

    button.setCallable(appendToResult, "a")
    numberOfCalls = qt.silxGlobalThreadPool().maxThreadCount()
    for _ in range(numberOfCalls):
        button.executeCallable()
    qapp_utils.waitUntil(lambda: not button.hasPendingOperations())
    assert result == ["a"] * numberOfCalls


def testSaturateThreadPool(button, qapp_utils):
    result = []

    def appendToResult(name):
        result.append(name)
        time.sleep(0.1)

    button.setCallable(appendToResult, "a")
    numberOfCalls = qt.silxGlobalThreadPool().maxThreadCount() * 2
    for _ in range(numberOfCalls):
        button.executeCallable()
    qapp_utils.waitUntil(lambda: not button.hasPendingOperations())
    assert result == ["a"] * numberOfCalls


def testSuccess(listener, button, qapp):

    def compute():
        return "result"

    button.setCallable(compute)
    button.beforeExecuting.connect(listener.partial(test="be"))
    button.started.connect(listener.partial(test="s"))
    button.succeeded.connect(listener.partial(test="result"))
    button.failed.connect(listener.partial(test="Unexpected exception"))
    button.finished.connect(listener.partial(test="f"))
    button.executeCallable()
    qapp.processEvents()
    time.sleep(0.1)
    qapp.processEvents()
    result = listener.karguments(argumentName="test")
    assert result == ["be", "s", "result", "f"]


def testFail(button, listener, qapp):
    def computeFail(self):
        raise Exception("exception")

    button.setCallable(computeFail)
    button.beforeExecuting.connect(listener.partial(test="be"))
    button.started.connect(listener.partial(test="s"))
    button.succeeded.connect(listener.partial(test="Unexpected success"))
    button.failed.connect(listener.partial(test="exception"))
    button.finished.connect(listener.partial(test="f"))
    with LoggingValidator("silx.gui.widgets.ThreadPoolPushButton", error=1):
        button.executeCallable()
        qapp.processEvents()
        time.sleep(0.1)
        qapp.processEvents()
    result = listener.karguments(argumentName="test")
    assert result == ["be", "s", "exception", "f"]
