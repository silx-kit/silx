from silx.gui import qt
from silx.gui.plot import Plot2D
from scipy.misc import ascent
app = qt.QApplication([])
plot = Plot2D()
data = ascent()
plot.addImage(data)

app.exec_()
