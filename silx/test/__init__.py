"""silx test suite.

silx.gui tests depends on Qt.
To disable them, set WITH_QT_TEST environement variable to 'False'.
"""

import logging
import os
import unittest


logging.basicConfig()
logger = logging.getLogger(__name__)


from .test_version import suite as test_version_suite
from ..io.test import suite as test_io_suite


if os.environ.get('WITH_QT_TEST', 'True') == 'True':
    from ..gui.test import suite as test_gui_suite
else:
    logger.warning('silx.gui tests are disabled (WITH_QT_TEST=False)')
    test_gui_suite = None


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_version_suite())
    if test_gui_suite is not None:
        test_suite.addTest(test_gui_suite())
    test_suite.addTest(test_io_suite())
    return test_suite
