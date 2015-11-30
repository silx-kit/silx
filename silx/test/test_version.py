"""Basic test of top-level package import and existence of version info."""
import unittest

import silx


class TestVersion(unittest.TestCase):
    def test_version(self):
        self.assertTrue(isinstance(silx.version, str))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestVersion))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
