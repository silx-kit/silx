# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
import copy
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy
import pytest
import random

from silx.math.fit import bgtheories
from silx.math.fit.functions import sum_gauss


def _strip(x, y):
    return bgtheories.strip_bg(x, y, 4, 20)


def _snip(x, y):
    return bgtheories.snip_bg(x, y, 16)


def _noise(length, seed=0):
    return numpy.random.default_rng(seed).random(length) * 10


@pytest.mark.parametrize("bg", [_strip, _snip])
def test_bg_cache_changed_data_lengths(bg):
    for length in (200, 100, 200):
        x = numpy.arange(float(length))
        y = _noise(length)
        # Same input again: the cached background is returned
        actual = bg(x, y)
        assert actual.shape == y.shape
        assert bg(x, y.copy()) is actual


@pytest.mark.parametrize("bg", [_strip, _snip])
def test_bg_cache_input_modified_in_place(bg):
    x = numpy.arange(200.0)
    y = _noise(200, seed=2)
    before = bg(x, y).copy()
    # Without a copy of y in the cache, the modified y looks unchanged
    y += 100.0
    assert not numpy.array_equal(bg(x, y), before)


def test_strip_cache_threshold_factor():
    x = numpy.arange(200.0)
    y = _noise(200)
    bgtheories.CONFIG["StripThresholdFactor"] = 1.0
    before = _strip(x, y).copy()
    # Without the factor in the cache key, the result of factor 1.0 is returned
    bgtheories.CONFIG["StripThresholdFactor"] = 5.0
    assert not numpy.array_equal(_strip(x, y), before)


def test_bg_cache_not_shared_between_strip_and_snip():
    x = numpy.arange(200.0)
    y = _noise(200)
    bgtheories.CONFIG["SmoothingFlag"] = False
    unsmoothed = _snip(x, y)
    bgtheories.CONFIG["SmoothingFlag"] = True
    # strip_bg must not make the unsmoothed snip result look valid
    _strip(x, y)
    smoothed = _snip(x, y)
    assert not numpy.array_equal(smoothed, unsmoothed)


@pytest.mark.parametrize("bg", [_strip, _snip])
def test_bg_cache_unchanged_when_computation_fails(bg):
    x = numpy.arange(200.0)
    y1 = _noise(200)
    y2 = _noise(200, seed=1)
    cached = bg(x, y1)
    bgtheories.CONFIG["AnchorsFlag"] = True
    bgtheories.CONFIG["AnchorsList"] = ["not a number"]
    for _ in range(2):
        # Without computing before publishing the cache, the second call
        # returns the cached background of y1 instead of raising
        with pytest.raises(TypeError):
            bg(x, y2)
    bgtheories.CONFIG.update(AnchorsFlag=False, AnchorsList=[])
    assert bg(x, y1) is cached


class _BlockingAnchor:
    """Anchor that blocks its first comparison until released, then is ignored"""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __le__(self, other: float) -> bool:
        if not self.entered.is_set():
            self.entered.set()
            self.release.wait(timeout=10)
        return True

    def __deepcopy__(self, memo: dict) -> "_BlockingAnchor":
        return self


@pytest.mark.parametrize("bg", [_strip, _snip])
def test_bg_cache_while_computed_in_other_thread(bg):
    x = numpy.arange(200.0)
    y_old = _noise(200, seed=3)
    y_new = _noise(200, seed=4)
    bg(x, y_old)
    anchor = _BlockingAnchor()
    bgtheories.CONFIG.update(AnchorsFlag=True, AnchorsList=[anchor])
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            # The other thread computes the background of y_new and blocks
            # in the anchor comparison
            future = executor.submit(bg, x, y_new)
            assert anchor.entered.wait(timeout=10)
            # Same call in this thread during that computation
            during = bg(x, y_new)
        finally:
            anchor.release.set()
        after = future.result(timeout=10)
    # The anchor is ignored, so the result is the background without anchors
    bgtheories.CONFIG.update(AnchorsFlag=False, AnchorsList=[])
    expected = bg(x, y_new)
    # Without publishing the cache after the computation, `during` is the
    # cached background of y_old
    numpy.testing.assert_array_equal(during, expected)
    numpy.testing.assert_array_equal(after, expected)


class TestBgTheories(unittest.TestCase):
    """ """

    def setUp(self):
        self.x = numpy.arange(100)
        self.y = 10 + 0.05 * self.x + sum_gauss(self.x, 10.0, 45.0, 15.0)
        # add a very narrow high amplitude peak to test strip and snip
        self.y += sum_gauss(self.x, 100.0, 75.0, 2.0)
        self.narrow_peak_index = list(self.x).index(75)
        random.seed()

    def tearDown(self):
        pass

    def testTheoriesAttrs(self):
        for theory_name in bgtheories.THEORY:
            self.assertIsInstance(theory_name, str)
            self.assertTrue(hasattr(bgtheories.THEORY[theory_name], "function"))
            self.assertTrue(
                hasattr(bgtheories.THEORY[theory_name].function, "__call__")
            )
        # Ensure legacy functions are not renamed accidentally
        self.assertTrue(
            {"No Background", "Constant", "Linear", "Strip", "Snip"}.issubset(
                set(bgtheories.THEORY)
            )
        )

    def testNoBg(self):
        nobgfun = bgtheories.THEORY["No Background"].function
        self.assertTrue(
            numpy.array_equal(nobgfun(self.x, self.y), numpy.zeros_like(self.x))
        )
        # default estimate
        self.assertEqual(
            bgtheories.THEORY["No Background"].estimate(self.x, self.y), ([], [])
        )

    def testConstant(self):
        consfun = bgtheories.THEORY["Constant"].function
        c = random.random() * 100
        self.assertTrue(
            numpy.array_equal(consfun(self.x, self.y, c), c * numpy.ones_like(self.x))
        )
        # default estimate
        esti_par, cons = bgtheories.THEORY["Constant"].estimate(self.x, self.y)
        self.assertEqual(cons, [[0, 0, 0]])
        self.assertAlmostEqual(esti_par, min(self.y))

    def testLinear(self):
        linfun = bgtheories.THEORY["Linear"].function
        a = random.random() * 100
        b = random.random() * 100
        self.assertTrue(numpy.array_equal(linfun(self.x, self.y, a, b), a + b * self.x))
        # default estimate
        esti_par, cons = bgtheories.THEORY["Linear"].estimate(self.x, self.y)

        self.assertEqual(cons, [[0, 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(esti_par[0], 10, places=3)
        self.assertAlmostEqual(esti_par[1], 0.05, places=3)

    def testStrip(self):
        stripfun = bgtheories.THEORY["Strip"].function
        anchors = sorted(random.sample(list(self.x), 4))
        anchors_indices = [list(self.x).index(a) for a in anchors]

        # we really want to strip away the narrow peak
        anchors_indices_copy = copy.deepcopy(anchors_indices)
        for idx in anchors_indices_copy:
            if abs(idx - self.narrow_peak_index) < 5:
                anchors_indices.remove(idx)
                anchors.remove(self.x[idx])

        width = 2
        niter = 1000
        bgtheories.THEORY["Strip"].configure(AnchorsList=anchors, AnchorsFlag=True)

        bg = stripfun(self.x, self.y, width, niter)

        # assert peak amplitude has been decreased
        self.assertLess(bg[self.narrow_peak_index], self.y[self.narrow_peak_index])

        # default estimate
        for i in anchors_indices:
            self.assertEqual(bg[i], self.y[i])

        # estimated parameters are equal to the default ones in the config dict
        bgtheories.THEORY["Strip"].configure(StripWidth=7, StripIterations=8)
        esti_par, cons = bgtheories.THEORY["Strip"].estimate(self.x, self.y)
        self.assertTrue(numpy.array_equal(cons, [[3, 0, 0], [3, 0, 0]]))
        self.assertEqual(esti_par, [7, 8])

    def testSnip(self):
        snipfun = bgtheories.THEORY["Snip"].function
        anchors = sorted(random.sample(list(self.x), 4))
        anchors_indices = [list(self.x).index(a) for a in anchors]

        # we want to strip away the narrow peak, so remove nearby anchors
        anchors_indices_copy = copy.deepcopy(anchors_indices)
        for idx in anchors_indices_copy:
            if abs(idx - self.narrow_peak_index) < 5:
                anchors_indices.remove(idx)
                anchors.remove(self.x[idx])

        width = 16
        bgtheories.THEORY["Snip"].configure(AnchorsList=anchors, AnchorsFlag=True)
        bg = snipfun(self.x, self.y, width)

        # assert peak amplitude has been decreased
        self.assertLess(
            bg[self.narrow_peak_index],
            self.y[self.narrow_peak_index],
            "Snip didn't decrease the peak amplitude.",
        )

        # anchored data must remain fixed
        for i in anchors_indices:
            self.assertEqual(bg[i], self.y[i])

        # estimated parameters are equal to the default ones in the config dict
        bgtheories.THEORY["Snip"].configure(SnipWidth=7)
        esti_par, cons = bgtheories.THEORY["Snip"].estimate(self.x, self.y)
        self.assertTrue(numpy.array_equal(cons, [[3, 0, 0]]))
        self.assertEqual(esti_par, [7])
