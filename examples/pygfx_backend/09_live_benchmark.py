"""Live update FPS benchmark: matplotlib vs opengl vs pygfx.

Measures actual draw FPS for each backend with identical workloads.
"""

import time
import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D


class FPSCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._count = 0
        self._start = time.perf_counter()

    def tick(self):
        self._count += 1

    @property
    def fps(self):
        elapsed = time.perf_counter() - self._start
        return self._count / elapsed if elapsed > 0 else 0

    @property
    def count(self):
        return self._count


class BenchmarkWidget(qt.QWidget):
    def __init__(self, n_points=1000, duration=5.0):
        super().__init__()
        self.setWindowTitle("Live Update FPS Benchmark")

        self._n_points = n_points
        self._duration = duration
        self._x = numpy.linspace(0, 4 * numpy.pi, n_points)
        self._phase = 0.0

        layout = qt.QVBoxLayout(self)

        # Info label
        self._label = qt.QLabel(
            f"Points: {n_points} | Duration: {duration}s per backend | Starting..."
        )
        self._label.setAlignment(qt.Qt.AlignCenter)
        font = self._label.font()
        font.setPointSize(14)
        self._label.setFont(font)
        layout.addWidget(self._label)

        # Plot area
        plot_layout = qt.QHBoxLayout()
        layout.addLayout(plot_layout)

        self._backends = ["mpl", "opengl", "pygfx"]
        self._plots = {}
        self._fps_labels = {}

        for backend in self._backends:
            container = qt.QVBoxLayout()
            try:
                plot = Plot1D(backend=backend)
                plot.setGraphTitle(backend)
                plot.setGraphYLimits(-1.5, 1.5)
                plot.setActiveCurveHandling(False)
                self._plots[backend] = plot
                container.addWidget(plot)
            except Exception as e:
                err = qt.QLabel(f"{backend}: {e}")
                container.addWidget(err)

            fps_label = qt.QLabel("waiting...")
            fps_label.setAlignment(qt.Qt.AlignCenter)
            fps_font = fps_label.font()
            fps_font.setPointSize(12)
            fps_font.setBold(True)
            fps_label.setFont(fps_font)
            self._fps_labels[backend] = fps_label
            container.addWidget(fps_label)

            plot_layout.addLayout(container)

        # Results label
        self._result_label = qt.QLabel("")
        self._result_label.setAlignment(qt.Qt.AlignCenter)
        font2 = self._result_label.font()
        font2.setPointSize(13)
        self._result_label.setFont(font2)
        layout.addWidget(self._result_label)

        # State
        self._current_backend_idx = 0
        self._counter = FPSCounter()
        self._results = {}

        # Timer for updates
        self._timer = qt.QTimer(self)
        self._timer.timeout.connect(self._update)

        # Start after a short delay
        qt.QTimer.singleShot(500, self._startNextBackend)

    def _startNextBackend(self):
        if self._current_backend_idx >= len(self._backends):
            self._showResults()
            return

        backend = self._backends[self._current_backend_idx]
        if backend not in self._plots:
            self._current_backend_idx += 1
            self._startNextBackend()
            return

        self._label.setText(
            f"Benchmarking: {backend} | {self._n_points} points | " f"{self._duration}s"
        )
        self._fps_labels[backend].setText("running...")
        self._fps_labels[backend].setStyleSheet("color: blue;")
        self._phase = 0.0
        self._counter.reset()
        self._timer.start(1)  # as fast as possible

    def _update(self):
        backend = self._backends[self._current_backend_idx]
        plot = self._plots.get(backend)
        if plot is None:
            return

        self._phase += 0.1
        y = numpy.sin(self._x + self._phase)
        plot.addCurve(
            self._x, y, legend="bench", color="blue", linewidth=2, resetzoom=False
        )
        self._counter.tick()

        fps = self._counter.fps
        self._fps_labels[backend].setText(f"{fps:.1f} FPS")

        if time.perf_counter() - self._counter._start >= self._duration:
            self._timer.stop()
            final_fps = self._counter.fps
            self._results[backend] = final_fps
            self._fps_labels[backend].setText(f"{final_fps:.1f} FPS")
            self._fps_labels[backend].setStyleSheet("color: green;")
            self._current_backend_idx += 1
            qt.QTimer.singleShot(300, self._startNextBackend)

    def _showResults(self):
        lines = ["Results:"]
        for backend, fps in self._results.items():
            lines.append(f"  {backend}: {fps:.1f} FPS")
        self._label.setText(
            " | ".join(f"{b}: {f:.1f} FPS" for b, f in self._results.items())
        )
        self._result_label.setText(
            f"Points: {self._n_points} | Duration: {self._duration}s each"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Live update FPS benchmark")
    parser.add_argument(
        "-n",
        "--points",
        type=int,
        default=1000,
        help="Number of curve points (default: 1000)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=5.0,
        help="Seconds per backend (default: 5)",
    )
    args = parser.parse_args()

    app = qt.QApplication([])
    w = BenchmarkWidget(n_points=args.points, duration=args.duration)
    w.resize(1500, 500)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
