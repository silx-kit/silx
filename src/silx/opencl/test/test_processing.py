import gc
import os

import numpy
import pytest

from ..common import ocl

if ocl:
    import pyopencl.array

    from ..processing import OpenclProcessing


@pytest.mark.skipif(ocl is None, reason="PyOpenCl is missing")
def test_context_cache():

    op1 = OpenclProcessing()
    op2 = OpenclProcessing()
    assert op1.ctx is op2.ctx, "Context should be the same"

    os.environ["PYOPENCL_CTX"] = "0:0"
    op3 = OpenclProcessing()
    op4 = OpenclProcessing()

    assert op3.ctx is op4.ctx, "context should be the same"


@pytest.mark.skipif(ocl is None, reason="PyOpenCl is missing")
def test_shared_queue():
    op1 = OpenclProcessing()
    op2 = OpenclProcessing(queue=op1)
    op3 = OpenclProcessing(queue=op1.shared_queue)
    assert op2.ctx is op1.ctx
    assert op2.shared_queue is op1.shared_queue
    assert op2.queue is op1.queue
    assert op3.queue is op1.queue
    op4 = OpenclProcessing(ctx=op1.ctx)
    assert op4.queue is not op1.queue, "queue is not shared by default"


@pytest.mark.skipif(ocl is None, reason="PyOpenCl is missing")
def test_shared_queue_profiling():
    op1 = OpenclProcessing()
    op1.cl_mem["a"] = pyopencl.array.zeros(op1.queue, 10, numpy.float32)
    op2 = OpenclProcessing(queue=op1)
    op2.cl_mem["b"] = pyopencl.array.zeros(op2.queue, 10, numpy.float32)
    assert not op1.profile and not op2.profile

    op1.set_profiling(True)
    assert op1.profile and op2.profile
    assert op2.queue is op1.queue
    assert op1.queue.properties & pyopencl.command_queue_properties.PROFILING_ENABLE
    for op in (op1, op2):
        for array in op.cl_mem.values():
            assert array.queue is op1.queue

    op2.set_profiling(False)
    assert not op1.profile and not op2.profile
    assert op2.queue is op1.queue

    # Requesting profiling while sharing enables it for everyone
    op3 = OpenclProcessing(queue=op1, profile=True)
    assert op1.profile and op2.profile and op3.profile
    assert op3.queue is op1.queue


@pytest.mark.skipif(ocl is None, reason="PyOpenCl is missing")
def test_shared_queue_from_command_queue():
    op1 = OpenclProcessing()
    queue = pyopencl.CommandQueue(
        op1.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE
    )
    op2 = OpenclProcessing(queue=queue)
    assert op2.queue is queue
    assert op2.profile
    assert op2.ctx == op1.ctx
    op2.set_profiling(False)
    assert op2.queue is not queue
    assert not op2.profile

    with pytest.raises(TypeError):
        OpenclProcessing(queue="not a queue")


@pytest.mark.skipif(ocl is None, reason="PyOpenCl is missing")
def test_shared_queue_release():
    op1 = OpenclProcessing()
    op2 = OpenclProcessing(queue=op1)
    shared_queue = op1.shared_queue
    op2.queue = None
    assert op2.queue is None
    assert op2.shared_queue is None
    with pytest.raises(ValueError):
        OpenclProcessing(queue=op2)
    del op1
    gc.collect()
    shared_queue.set_profiling(True)  # No registered processing left
