from ._enum import *
from silx.utils.deprecation import deprecated_warning

deprecated_warning(
    "Class",
    "silx.utils.enum.Enum",
    since_version="2.1.1",
    replacement="Python built-in Enum class",
)
