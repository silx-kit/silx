import sys

from silx.gui import qt
from silx.gui.widgets.CollapsibleWidget import CollapsibleWidget

app = qt.QApplication(sys.argv)

mainWidget = qt.QWidget()
layout = qt.QVBoxLayout(mainWidget)

sliderWidget = CollapsibleWidget("Sliders")
sliderLayout = qt.QHBoxLayout()
sliderLayout.addWidget(qt.QSlider())
sliderLayout.addWidget(qt.QSlider())
sliderLayout.addWidget(qt.QSlider())
sliderWidget.setContentsLayout(sliderLayout)
layout.addWidget(sliderWidget)

paramWidget = CollapsibleWidget("Parameters")
paramLayout = qt.QFormLayout()
paramLayout.addRow("File", qt.QLineEdit())
paramLayout.addRow("Scan", qt.QSpinBox())
paramLayout.addRow("Overwrite", qt.QCheckBox())
paramWidget.setContentsLayout(paramLayout)
layout.addWidget(paramWidget)
layout.addStretch(1)

mainWidget.show()
app.exec()
