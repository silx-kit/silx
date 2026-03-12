# /*##########################################################################
#
# Copyright (c) 2024 European Synchrotron Radiation Facility
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
# ###########################################################################*/

"""
This script benchmarks image streaming performance with the pygfx backend.

Pre-generates image frames, then streams them at maximum rate to measure
pure rendering throughput. VSync is disabled to measure actual GPU
throughput beyond the monitor refresh rate.

Usage::

    python imageStreamingBenchmark.py
    python imageStreamingBenchmark.py --size 2048 --duration 10
"""

from __future__ import annotations

__license__ = "MIT"

import time
import argparse

import numpy

from silx.gui import qt
from silx.gui.plot.PlotWindow import PlotWindow
from silx.gui.colors import Colormap

_NUM_PREGEN_FRAMES = 20


def _pregenerate_frames(size, n=_NUM_PREGEN_FRAMES):
    """Pre-generate a pool of test frames with moving Gaussian peaks."""
    frames = []
    for i in range(n):
        t = i * 0.3
        cx = (numpy.sin(t) * 0.5 + 0.5) * size
        cy = (numpy.cos(t) * 0.5 + 0.5) * size
        sigma = size / 8
        y, x = numpy.ogrid[:size, :size]
        img = numpy.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        img += 0.05 * numpy.random.random((size, size))
        frames.append(img.astype(numpy.float32))
    return frames


class StreamingBenchmark(qt.QWidget):
    """Interactive image streaming benchmark widget."""

    def __init__(self, image_size=1024, duration=5.0):
        super().__init__()
        self.setWindowTitle("Image Streaming Benchmark (pygfx)")
        self._duration = duration
        self._image_size = image_size

        layout = qt.QVBoxLayout(self)

        # Controls
        ctrl = qt.QHBoxLayout()
        ctrl.addWidget(qt.QLabel("Size:"))
        self._size_combo = qt.QComboBox()
        self._size_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self._size_combo.setCurrentText(str(image_size))
        ctrl.addWidget(self._size_combo)

        ctrl.addWidget(qt.QLabel("Norm:"))
        self._norm_combo = qt.QComboBox()
        self._norm_combo.addItems(["linear", "log", "sqrt", "gamma", "arcsinh"])
        ctrl.addWidget(self._norm_combo)

        ctrl.addStretch()
        self._status = qt.QLabel("Ready")
        self._status.setMinimumWidth(400)
        font = self._status.font()
        font.setPointSize(13)
        font.setBold(True)
        self._status.setFont(font)
        ctrl.addWidget(self._status)

        self._start_btn = qt.QPushButton("Start")
        self._stop_btn = qt.QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        ctrl.addWidget(self._start_btn)
        ctrl.addWidget(self._stop_btn)
        layout.addLayout(ctrl)

        # Disable VSync to measure actual GPU throughput
        from silx.gui.plot.backends.BackendPygfx import BackendPygfx

        BackendPygfx.VSYNC = False

        # Plot with toolbar (includes colormap dialog button)
        self._plot = PlotWindow(
            backend="pygfx", colormap=True, mask=False, roi=False, fit=False
        )
        self._plot.setGraphTitle("pygfx streaming (vsync off)")
        self._plot.setKeepDataAspectRatio(True)
        layout.addWidget(self._plot)

        # Results table
        self._results_text = qt.QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(200)
        self._results_text.setFontFamily("monospace")
        self._results_text.setText(
            "Results will appear here after each run.\n"
            "Try different sizes and normalizations to compare.\n\n"
            "VSync is disabled to measure actual GPU throughput.\n"
            "plot_ms = updateImageData + _draw() (GPU pipeline)\n"
            "other   = Qt processEvents overhead\n"
            "total   = plot + other (~1000/FPS)"
        )
        layout.addWidget(self._results_text)

        # State
        self._timer = qt.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._frame_count = 0
        self._t_start = 0.0
        self._frame_plot = []
        self._frame_other = []
        self._frames = []
        self._results = []

        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)

    def _start(self):
        size = int(self._size_combo.currentText())
        norm = self._norm_combo.currentText()

        # Set colormap with fixed range (no autoscale overhead)
        if norm == "log":
            cm = Colormap("viridis", normalization="log", vmin=0.01, vmax=1.5)
        elif norm == "gamma":
            cm = Colormap("viridis", normalization="gamma", vmin=0.0, vmax=1.5)
            cm.setGammaNormalizationParameter(2.2)
        elif norm == "arcsinh":
            cm = Colormap("viridis", normalization="arcsinh", vmin=-0.5, vmax=1.5)
        else:
            cm = Colormap("viridis", normalization=norm, vmin=0.0, vmax=1.5)

        self._plot.setDefaultColormap(cm)
        self._image_size = size
        self._frame_count = 0
        self._frame_plot = []
        self._frame_other = []

        # Pre-generate frames
        self._status.setText(
            f"Generating {_NUM_PREGEN_FRAMES} frames ({size}x{size})..."
        )
        qt.QApplication.processEvents()
        self._frames = _pregenerate_frames(size)

        # Warm-up frame
        self._plot.addImage(self._frames[0], legend="bench", resetzoom=True)
        self._plot._backend._draw()
        qt.QApplication.processEvents()

        self._t_start = time.perf_counter()
        self._last_fps_time = self._t_start

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._size_combo.setEnabled(False)
        self._norm_combo.setEnabled(False)
        self._status.setText(f"Running: {size}x{size} {norm}...")
        self._timer.start(0)

    def _stop(self):
        self._timer.stop()
        elapsed = time.perf_counter() - self._t_start
        n = max(len(self._frame_plot), 1)
        avg_fps = n / elapsed if elapsed > 0 else 0

        plot = numpy.array(self._frame_plot) * 1000
        other = numpy.array(self._frame_other) * 1000

        avg_plot = float(numpy.mean(plot)) if len(plot) else 0
        avg_other = float(numpy.mean(other)) if len(other) else 0

        size = self._image_size
        norm = self._norm_combo.currentText()

        self._results.append((size, norm, avg_fps, avg_plot, avg_other, n, elapsed))

        # Update results table
        lines = [
            f"{'Size':>6} {'Norm':>8} {'FPS':>7} "
            f"{'plot_ms':>8} {'other':>7} {'total':>7} "
            f"{'Frames':>7} {'Time':>5}"
        ]
        lines.append("-" * 62)
        for s, no, fps, pm, om, fr, t in self._results:
            lines.append(
                f"{s:>6} {no:>8} {fps:>7.1f} "
                f"{pm:>8.2f} {om:>7.2f} {pm + om:>7.2f} "
                f"{fr:>7} {t:>4.1f}s"
            )
        self._results_text.setText("\n".join(lines))

        # Also print to console
        print("\n".join(lines[-1:]))

        self._status.setText(
            f"Done: {avg_fps:.1f} FPS | "
            f"plot {avg_plot:.1f} + other {avg_other:.1f}ms"
        )
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._size_combo.setEnabled(True)
        self._norm_combo.setEnabled(True)

    def _tick(self):
        img = self._frames[self._frame_count % len(self._frames)]

        # Plot update + GPU render
        t0 = time.perf_counter()
        self._plot.updateImageData(img, legend="bench")
        self._plot._backend._draw()
        t1 = time.perf_counter()

        # Qt event processing
        qt.QApplication.processEvents()
        t2 = time.perf_counter()

        self._frame_plot.append(t1 - t0)
        self._frame_other.append(t2 - t1)
        self._frame_count += 1

        # Update status every 0.5s
        if t2 - self._last_fps_time >= 0.5:
            n = self._frame_count
            elapsed = t2 - self._t_start
            fps = n / elapsed if elapsed > 0 else 0
            avg_plot = numpy.mean(self._frame_plot) * 1000
            avg_other = numpy.mean(self._frame_other) * 1000
            self._status.setText(
                f"{self._image_size}x{self._image_size} | FPS: {fps:.1f} | "
                f"plot {avg_plot:.1f} + other {avg_other:.1f}ms"
            )
            self._last_fps_time = t2

        # Auto-stop after duration
        if t2 - self._t_start >= self._duration:
            self._stop()


def main():
    parser = argparse.ArgumentParser(
        description="Image streaming benchmark (pygfx backend)"
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=1024,
        help="Initial image size (default: 1024)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=5.0,
        help="Seconds per run (default: 5)",
    )
    args = parser.parse_args()

    app = qt.QApplication([])
    w = StreamingBenchmark(image_size=args.size, duration=args.duration)
    w.resize(900, 700)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
