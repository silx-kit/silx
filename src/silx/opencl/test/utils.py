"""Helpers for the OpenCL tests"""

from __future__ import annotations

import os

from ..common import Device, Platform, get_pyopencl_ctx_tuple, ocl


def get_test_devices() -> list[tuple[Platform, Device]]:
    """Returns the OpenCL (platform, device) pairs the tests should run on."""
    if not ocl:
        return []

    pyopencl_ctx = os.environ.get("PYOPENCL_CTX")
    if pyopencl_ctx:
        platform_id, device_id = get_pyopencl_ctx_tuple(pyopencl_ctx)
        platform = ocl.platforms[platform_id]
        for device in platform.devices:
            if device.id == device_id:
                return [(platform, device)]
        return []

    return [
        (platform, device) for platform in ocl.platforms for device in platform.devices
    ]
