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
"""This module defines a views used by :class:`silx.gui.data.DataViewer`."""

import logging
import os

from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.dialog.ColormapDialog import ColormapDialog
from silx.utils.deprecation import deprecated

from ._utils import normalizeData

__authors__ = ["V. Valls", "P. Knobel"]
__license__ = "MIT"

_logger = logging.getLogger(__name__)


class DataViewHooks:
    """A set of hooks defined to custom the behaviour of the data views."""

    def getColormap(self, view):
        """Returns a colormap for this view."""
        return None

    def getColormapDialog(self, view):
        """Returns a color dialog for this view."""
        return None

    def viewWidgetCreated(self, view, plot):
        """Called when the widget of the view was created"""
        return


class DataView:
    """Holder for the data view."""

    UNSUPPORTED = -1
    """Priority returned when the requested data can't be displayed by the
    view."""

    TITLE_PATTERN = "{datapath}{slicing} {permuted}"
    """Pattern used to format the title of the plot.

    Supported fields: `{directory}`, `{filename}`, `{datapath}`, `{slicing}`, `{permuted}`.
    """

    def __init__(self, parent, modeId=None, icon=None, label=None):
        """Constructor

        :param qt.QWidget parent: Parent of the hold widget
        """
        self.__parent = parent
        self.__widget = None
        self.__modeId = modeId
        if label is None:
            label = self.__class__.__name__
        self.__label = label
        if icon is None:
            icon = qt.QIcon()
        self.__icon = icon
        self.__hooks = None

    def getHooks(self):
        """Returns the data viewer hooks used by this view.

        :rtype: DataViewHooks
        """
        return self.__hooks

    def setHooks(self, hooks):
        """Set the data view hooks to use with this view.

        :param DataViewHooks hooks: The data view hooks to use
        """
        self.__hooks = hooks

    def defaultColormap(self):
        """Returns a default colormap.

        :rtype: Colormap
        """
        colormap = None
        if self.__hooks is not None:
            colormap = self.__hooks.getColormap(self)
        if colormap is None:
            colormap = Colormap(name="viridis")
        return colormap

    def defaultColorDialog(self):
        """Returns a default color dialog.

        :rtype: ColormapDialog
        """
        dialog = None
        if self.__hooks is not None:
            dialog = self.__hooks.getColormapDialog(self)
        if dialog is None:
            dialog = ColormapDialog()
            dialog.setModal(False)
        return dialog

    def icon(self):
        """Returns the default icon"""
        return self.__icon

    def label(self):
        """Returns the default label"""
        return self.__label

    def modeId(self):
        """Returns the mode id"""
        return self.__modeId

    def normalizeData(self, data):
        """Returns a normalized data if the embed a numpy or a dataset.
        Else returns the data."""
        return normalizeData(data)

    @deprecated(reason="Not used", since_version="3.0.0")
    def customAxisNames(self):
        return []

    @deprecated(reason="Not used", since_version="3.0.0")
    def setCustomAxisValue(self, name, value):
        pass

    def isWidgetInitialized(self):
        """Returns true if the widget is already initialized."""
        return self.__widget is not None

    def select(self):
        """Called when the view is selected to display the data."""
        return

    def getWidget(self):
        """Returns the widget hold in the view and displaying the data.

        :returns: qt.QWidget
        """
        if self.__widget is None:
            self.__widget = self.createWidget(self.__parent)
            hooks = self.getHooks()
            if hooks is not None:
                hooks.viewWidgetCreated(self, self.__widget)
        return self.__widget

    def createWidget(self, parent):
        """Create the the widget displaying the data

        :param qt.QWidget parent: Parent of the widget
        :returns: qt.QWidget
        """
        raise NotImplementedError()

    def clear(self):
        """Clear the data from the view"""
        return None

    def setData(self, data):
        """Set the data displayed by the view

        :param data: Data to display
        :type data: numpy.ndarray or h5py.Dataset
        """
        return None

    def __formatSlices(self, indices):
        """Format an iterable of slice objects

        :param indices: The slices to format
        :type indices: Union[None,List[Union[slice,int]]]
        :rtype: str
        """
        if indices is None:
            return ""

        def formatSlice(slice_):
            start, stop, step = slice_.start, slice_.stop, slice_.step
            string = ("" if start is None else str(start)) + ":"
            if stop is not None:
                string += str(stop)
            if step not in (None, 1):
                string += ":" + step
            return string

        return (
            "["
            + ", ".join(
                formatSlice(index) if isinstance(index, slice) else str(index)
                for index in indices
            )
            + "]"
        )

    def titleForSelection(self, selection):
        """Build title from given selection information.

        :param NamedTuple selection: Data selected
        :rtype: str
        """
        if selection is None or selection.filename is None:
            return None
        else:
            directory, filename = os.path.split(selection.filename)
            try:
                slicing = self.__formatSlices(selection.slice)
            except Exception:
                _logger.debug("Error while formatting slices", exc_info=True)
                slicing = "[sliced]"

            permuted = "(permuted)" if selection.permutation is not None else ""

            try:
                title = self.TITLE_PATTERN.format(
                    directory=directory,
                    filename=filename,
                    datapath=selection.datapath,
                    slicing=slicing,
                    permuted=permuted,
                )
            except Exception:
                _logger.debug("Error while formatting title", exc_info=True)
                title = selection.datapath + slicing

            return title

    def setDataSelection(self, selection):
        """Set the data selection displayed by the view

        If called, it have to be called directly after `setData`.

        :param selection: Data selected
        :type selection: NamedTuple
        """
        pass

    def axesNames(self, data, info):
        """Returns names of the expected axes of the view, according to the
        input data. A none value will disable the default axes selectior.

        :param data: Data to display
        :type data: numpy.ndarray or h5py.Dataset
        :param DataInfo info: Pre-computed information on the data
        :rtype: list[str] or None
        """
        return []

    def getReachableViews(self):
        """Returns the views that can be returned by `getMatchingViews`.

        :param object data: Any object to be displayed
        :param DataInfo info: Information cached about this data
        :rtype: List[DataView]
        """
        return [self]

    def getMatchingViews(self, data, info):
        """Returns the views according to data and info from the data.

        :param object data: Any object to be displayed
        :param DataInfo info: Information cached about this data
        :rtype: List[DataView]
        """
        priority = self.getCachedDataPriority(data, info)
        if priority == DataView.UNSUPPORTED:
            return []
        return [self]

    def getCachedDataPriority(self, data, info):
        try:
            priority = info.getPriority(self)
        except KeyError:
            priority = self.getDataPriority(data, info)
            info.cachePriority(self, priority)
        return priority

    def getDataPriority(self, data, info):
        """
        Returns the priority of using this view according to a data.

        - `UNSUPPORTED` means this view can't display this data
        - `1` means this view can display the data
        - `100` means this view should be used for this data
        - `1000` max value used by the views provided by silx
        - ...

        :param object data: The data to check
        :param DataInfo info: Pre-computed information on the data
        :rtype: int
        """
        return DataView.UNSUPPORTED

    def __lt__(self, other):
        return str(self) < str(other)
