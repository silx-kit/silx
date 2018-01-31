

import sys
import numpy
from silx.gui import qt
from silx.gui.plot.GridImageWidget import GridImageWidget

app = qt.QApplication(sys.argv)

giw = GridImageWidget()
giw.setNCols(3)
giw.setNRows(2)

giw.setFrames(numpy.random.rand(50, 150, 200))


giw.show()
app.exec_()
