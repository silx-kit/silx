import os
import pytest

from silx.opencl.common import ocl

if ocl:
    from silx.opencl.processing import OpenclProcessing


@pytest.mark.skipif(ocl is None, reason="PyOpenCl is missing")
def test_context_cache():

    op1 = OpenclProcessing()
    op2 = OpenclProcessing()
    assert op1.ctx is op2.ctx, "Context should be the same"

    os.environ["PYOPENCL_CTX"] = "0:0"
    op3 = OpenclProcessing()
    op4 = OpenclProcessing()

    assert op3.ctx is op4.ctx, "context should be the same"
