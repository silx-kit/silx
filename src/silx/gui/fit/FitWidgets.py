# coding: utf-8
# /*##########################################################################
# Copyright (C) 2004-2016 European Synchrotron Radiation Facility
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
# ######################################################################### */
"""Collection of widgets used to build
:class:`silx.gui.fit.FitWidget.FitWidget`"""

from collections import OrderedDict

from silx.gui import qt
from silx.gui.fit.Parameters import Parameters

QTVERSION = qt.qVersion()

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "13/10/2016"


class FitActionsButtons(qt.QWidget):
    """Widget with 3 ``QPushButton``:

    The buttons can be accessed as public attributes::

        - ``EstimateButton``
        - ``StartFitButton``
        - ``DismissButton``

    You will typically need to access these attributes to connect the buttons
    to actions. For instance, if you have 3 functions ``estimate``,
    ``runfit`` and  ``dismiss``, you can connect them like this::

        >>> fit_actions_buttons = FitActionsButtons()
        >>> fit_actions_buttons.EstimateButton.clicked.connect(estimate)
        >>> fit_actions_buttons.StartFitButton.clicked.connect(runfit)
        >>> fit_actions_buttons.DismissButton.clicked.connect(dismiss)

    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.resize(234, 53)

        grid_layout = qt.QGridLayout(self)
        grid_layout.setContentsMargins(11, 11, 11, 11)
        grid_layout.setSpacing(6)
        layout = qt.QHBoxLayout(None)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.EstimateButton = qt.QPushButton(self)
        self.EstimateButton.setText("Estimate")
        layout.addWidget(self.EstimateButton)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        layout.addItem(spacer)

        self.StartFitButton = qt.QPushButton(self)
        self.StartFitButton.setText("Start Fit")
        layout.addWidget(self.StartFitButton)
        spacer_2 = qt.QSpacerItem(20, 20,
                                  qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        layout.addItem(spacer_2)

        self.DismissButton = qt.QPushButton(self)
        self.DismissButton.setText("Dismiss")
        layout.addWidget(self.DismissButton)

        grid_layout.addLayout(layout, 0, 0)


class FitStatusLines(qt.QWidget):
    """Widget with 2 greyed out write-only ``QLineEdit``.

    These text widgets can be accessed as public attributes::

        - ``StatusLine``
        - ``ChisqLine``

    You will typically need to access these widgets to update the displayed
    text::

        >>> fit_status_lines = FitStatusLines()
        >>> fit_status_lines.StatusLine.setText("Ready")
        >>> fit_status_lines.ChisqLine.setText("%6.2f" % 0.01)

    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.resize(535, 47)

        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.StatusLabel = qt.QLabel(self)
        self.StatusLabel.setText("Status:")
        layout.addWidget(self.StatusLabel)

        self.StatusLine = qt.QLineEdit(self)
        self.StatusLine.setText("Ready")
        self.StatusLine.setReadOnly(1)
        layout.addWidget(self.StatusLine)

        self.ChisqLabel = qt.QLabel(self)
        self.ChisqLabel.setText("Reduced chisq:")
        layout.addWidget(self.ChisqLabel)

        self.ChisqLine = qt.QLineEdit(self)
        self.ChisqLine.setMaximumSize(qt.QSize(16000, 32767))
        self.ChisqLine.setText("")
        self.ChisqLine.setReadOnly(1)
        layout.addWidget(self.ChisqLine)


class FitConfigWidget(qt.QWidget):
    """Widget whose purpose is to select a fit theory and a background
    theory, load a new fit theory definition file and provide
    a "Configure" button to open an advanced configuration dialog.

    This is used in :class:`silx.gui.fit.FitWidget.FitWidget`, to offer
    an interface to quickly modify the main parameters prior to running a fit:

      - select a fitting function through :attr:`FunComBox`
      - select a background function through :attr:`BkgComBox`
      - open a dialog for modifying advanced parameters through
        :attr:`FunConfigureButton`
    """
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.setWindowTitle("FitConfigGUI")

        layout = qt.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.FunLabel = qt.QLabel(self)
        self.FunLabel.setText("Function")
        layout.addWidget(self.FunLabel, 0, 0)

        self.FunComBox = qt.QComboBox(self)
        self.FunComBox.addItem("Add Function(s)")
        self.FunComBox.setItemData(self.FunComBox.findText("Add Function(s)"),
                                   "Load fit theories from a file",
                                   qt.Qt.ToolTipRole)
        layout.addWidget(self.FunComBox, 0, 1)

        self.BkgLabel = qt.QLabel(self)
        self.BkgLabel.setText("Background")
        layout.addWidget(self.BkgLabel, 1, 0)

        self.BkgComBox = qt.QComboBox(self)
        self.BkgComBox.addItem("Add Background(s)")
        self.BkgComBox.setItemData(self.BkgComBox.findText("Add Background(s)"),
                                   "Load background theories from a file",
                                   qt.Qt.ToolTipRole)
        layout.addWidget(self.BkgComBox, 1, 1)

        self.FunConfigureButton = qt.QPushButton(self)
        self.FunConfigureButton.setText("Configure")
        self.FunConfigureButton.setToolTip(
                "Open a configuration dialog for the selected function")
        layout.addWidget(self.FunConfigureButton, 0, 2)

        self.BgConfigureButton = qt.QPushButton(self)
        self.BgConfigureButton.setText("Configure")
        self.BgConfigureButton.setToolTip(
                "Open a configuration dialog for the selected background")
        layout.addWidget(self.BgConfigureButton, 1, 2)

        self.WeightCheckBox = qt.QCheckBox(self)
        self.WeightCheckBox.setText("Weighted fit")
        self.WeightCheckBox.setToolTip(
                "Enable usage of weights in the least-square problem.\n Use" +
                " the uncertainties (sigma) if provided, else use sqrt(y).")

        layout.addWidget(self.WeightCheckBox, 0, 3, 2, 1)

        layout.setColumnStretch(4, 1)


class ParametersTab(qt.QTabWidget):
    """This widget provides tabs to display and modify fit parameters. Each
    tab contains a table with fit data such as parameter names, estimated
    values, fit constraints, and final fit results.

    The usual way to initialize the table is to fill it with the fit
    parameters from a :class:`silx.math.fit.fitmanager.FitManager` object, after
    the estimation process or after the final fit.

    In the following example we use a :class:`ParametersTab` to display the
    results of two separate fits::

        from silx.math.fit import fittheories
        from silx.math.fit import fitmanager
        from silx.math.fit import functions
        from silx.gui import qt
        import numpy

        a = qt.QApplication([])

        # Create synthetic data
        x = numpy.arange(1000)
        y1 = functions.sum_gauss(x, 100, 400, 100)

        fit = fitmanager.FitManager(x=x, y=y1)

        fitfuns = fittheories.FitTheories()
        fit.addtheory(theory="Gaussian",
                      function=functions.sum_gauss,
                      parameters=("height", "peak center", "fwhm"),
                      estimate=fitfuns.estimate_height_position_fwhm)
        fit.settheory('Gaussian')
        fit.configure(PositiveFwhmFlag=True,
                      PositiveHeightAreaFlag=True,
                      AutoFwhm=True,)

        # Fit
        fit.estimate()
        fit.runfit()

        # Show first fit result in a tab in our widget
        w = ParametersTab()
        w.show()
        w.fillFromFit(fit.fit_results, view='Gaussians')

        # new synthetic data
        y2 = functions.sum_splitgauss(x,
                                  100, 400, 100, 40,
                                  10, 600, 50, 500,
                                  80, 850, 10, 50)
        fit.setData(x=x, y=y2)

        # Define new theory
        fit.addtheory(theory="Asymetric gaussian",
                      function=functions.sum_splitgauss,
                      parameters=("height", "peak center", "left fwhm", "right fwhm"),
                      estimate=fitfuns.estimate_splitgauss)
        fit.settheory('Asymetric gaussian')

        # Fit
        fit.estimate()
        fit.runfit()

        # Show first fit result in another tab in our widget
        w.fillFromFit(fit.fit_results, view='Asymetric gaussians')
        a.exec_()

    """

    def __init__(self, parent=None, name="FitParameters"):
        """

        :param parent: Parent widget
        :param name: Widget title
        """
        qt.QTabWidget.__init__(self, parent)
        self.setWindowTitle(name)
        self.setContentsMargins(0, 0, 0, 0)

        self.views = OrderedDict()
        """Dictionary of views. Keys are view names,
         items are :class:`Parameters` widgets"""

        self.latest_view = None
        """Name of latest view"""

        # the widgets/tables themselves
        self.tables = {}
        """Dictionary of :class:`silx.gui.fit.parameters.Parameters` objects.
        These objects store fit results
        """

        self.setContentsMargins(10, 10, 10, 10)

    def setView(self, view=None, fitresults=None):
        """Add or update a table. Fill it with data from a fit

        :param view: Tab name to be added or updated. If ``None``, use the
            latest view.
        :param fitresults: Fit data to be added to the table
        :raise: KeyError if no view name specified and no latest view
            available.
        """
        if view is None:
            if self.latest_view is not None:
                view = self.latest_view
            else:
                raise KeyError(
                    "No view available. You must specify a view" +
                    " name the first time you call this method."
                )

        if view in self.tables.keys():
            table = self.tables[view]
        else:
            # create the parameters instance
            self.tables[view] = Parameters(self)
            table = self.tables[view]
            self.views[view] = table
            self.addTab(table, str(view))

        if fitresults is not None:
            table.fillFromFit(fitresults)

        self.setCurrentWidget(self.views[view])
        self.latest_view = view

    def renameView(self, oldname=None, newname=None):
        """Rename a view (tab)

        :param oldname: Name of the view to be renamed
        :param newname: New name of the view"""
        error = 1
        if newname is not None:
            if newname not in self.views.keys():
                if oldname in self.views.keys():
                    parameterlist = self.tables[oldname].getFitResults()
                    self.setView(view=newname, fitresults=parameterlist)
                    self.removeView(oldname)
                    error = 0
        return error

    def fillFromFit(self, fitparameterslist, view=None):
        """Update a view with data from a fit (alias for :meth:`setView`)

        :param view: Tab name to be added or updated (default: latest view)
        :param fitparameterslist: Fit data to be added to the table
        """
        self.setView(view=view, fitresults=fitparameterslist)

    def getFitResults(self, name=None):
        """Call :meth:`getFitResults` for the
        :class:`silx.gui.fit.parameters.Parameters` corresponding to the
        latest table or to the named table (if ``name`` is not
        ``None``). This return a list of dictionaries in the format used by
        :class:`silx.math.fit.fitmanager.FitManager` to store fit parameter
        results.

        :param name: View name.
        """
        if name is None:
            name = self.latest_view
        return self.tables[name].getFitResults()

    def removeView(self, name):
        """Remove a view by name.

        :param name: View name.
        """
        if name in self.views:
            index = self.indexOf(self.tables[name])
            self.removeTab(index)
            index = self.indexOf(self.views[name])
            self.removeTab(index)
            del self.tables[name]
            del self.views[name]

    def removeAllViews(self, keep=None):
        """Remove all views, except the one specified (argument
        ``keep``)

        :param keep: Name of the view to be kept."""
        for view in self.tables:
            if view != keep:
                self.removeView(view)

    def getHtmlText(self, name=None):
        """Return the table data as HTML

        :param name: View name."""
        if name is None:
            name = self.latest_view
        table = self.tables[name]
        lemon = ("#%x%x%x" % (255, 250, 205)).upper()
        hcolor = ("#%x%x%x" % (230, 240, 249)).upper()
        text = ""
        text += "<nobr>"
        text += "<table>"
        text += "<tr>"
        ncols = table.columnCount()
        for l in range(ncols):
            text += ('<td align="left" bgcolor="%s"><b>' % hcolor)
            if QTVERSION < '4.0.0':
                text += (str(table.horizontalHeader().label(l)))
            else:
                text += (str(table.horizontalHeaderItem(l).text()))
            text += "</b></td>"
        text += "</tr>"
        nrows = table.rowCount()
        for r in range(nrows):
            text += "<tr>"
            item = table.item(r, 0)
            newtext = ""
            if item is not None:
                newtext = str(item.text())
            if len(newtext):
                color = "white"
                b = "<b>"
            else:
                b = ""
                color = lemon
            try:
                # MyQTable item has color defined
                cc = table.item(r, 0).color
                cc = ("#%x%x%x" % (cc.red(), cc.green(), cc.blue())).upper()
                color = cc
            except:
                pass
            for c in range(ncols):
                item = table.item(r, c)
                newtext = ""
                if item is not None:
                    newtext = str(item.text())
                if len(newtext):
                    finalcolor = color
                else:
                    finalcolor = "white"
                if c < 2:
                    text += ('<td align="left" bgcolor="%s">%s' %
                             (finalcolor, b))
                else:
                    text += ('<td align="right" bgcolor="%s">%s' %
                             (finalcolor, b))
                text += newtext
                if len(b):
                    text += "</td>"
                else:
                    text += "</b></td>"
            item = table.item(r, 0)
            newtext = ""
            if item is not None:
                newtext = str(item.text())
            if len(newtext):
                text += "</b>"
            text += "</tr>"
            text += "\n"
        text += "</table>"
        text += "</nobr>"
        return text

    def getText(self, name=None):
        """Return the table data as CSV formatted text, using tabulation
        characters as separators.

        :param name: View name."""
        if name is None:
            name = self.latest_view
        table = self.tables[name]
        text = ""
        ncols = table.columnCount()
        for l in range(ncols):
            text += (str(table.horizontalHeaderItem(l).text())) + "\t"
        text += "\n"
        nrows = table.rowCount()
        for r in range(nrows):
            for c in range(ncols):
                newtext = ""
                if c != 4:
                    item = table.item(r, c)
                    if item is not None:
                        newtext = str(item.text())
                else:
                    item = table.cellWidget(r, c)
                    if item is not None:
                        newtext = str(item.currentText())
                text += newtext + "\t"
            text += "\n"
        text += "\n"
        return text


def test():
    from silx.math.fit import fittheories
    from silx.math.fit import fitmanager
    from silx.math.fit import functions
    from silx.gui.plot.PlotWindow import PlotWindow
    import numpy

    a = qt.QApplication([])

    x = numpy.arange(1000)
    y1 = functions.sum_gauss(x, 100, 400, 100)

    fit = fitmanager.FitManager(x=x, y=y1)

    fitfuns = fittheories.FitTheories()
    fit.addtheory(name="Gaussian",
                  function=functions.sum_gauss,
                  parameters=("height", "peak center", "fwhm"),
                  estimate=fitfuns.estimate_height_position_fwhm)
    fit.settheory('Gaussian')
    fit.configure(PositiveFwhmFlag=True,
                  PositiveHeightAreaFlag=True,
                  AutoFwhm=True,)

    # Fit
    fit.estimate()
    fit.runfit()

    w = ParametersTab()
    w.show()
    w.fillFromFit(fit.fit_results, view='Gaussians')

    y2 = functions.sum_splitgauss(x,
                                  100, 400, 100, 40,
                                  10, 600, 50, 500,
                                  80, 850, 10, 50)
    fit.setdata(x=x, y=y2)

    # Define new theory
    fit.addtheory(name="Asymetric gaussian",
                  function=functions.sum_splitgauss,
                  parameters=("height", "peak center", "left fwhm", "right fwhm"),
                  estimate=fitfuns.estimate_splitgauss)
    fit.settheory('Asymetric gaussian')

    # Fit
    fit.estimate()
    fit.runfit()

    w.fillFromFit(fit.fit_results, view='Asymetric gaussians')

    # Plot
    pw = PlotWindow(control=True)
    pw.addCurve(x, y1, "Gaussians")
    pw.addCurve(x, y2, "Asymetric gaussians")
    pw.show()

    a.exec_()


if __name__ == "__main__":
    test()
