from silx.utils.deprecation import deprecated_warning

deprecated_warning(
    "Module",
    "silx.third_party.TiffIO",
    since_version="2.0.0",
    replacement="fabio.TiffIO",
)

from fabio.TiffIO import *
