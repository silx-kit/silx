from silx.gui import icons, qt

from .modes import RAW_MODE
from ._DataView import DataView
from .views import _ArrayView, _HexaView, _RecordView, _ScalarView


class _CompositeDataView(DataView):
    """Contains sub views"""

    def getViews(self):
        """Returns the direct sub views registered in this view.

        :rtype: List[DataView]
        """
        raise NotImplementedError()

    def getReachableViews(self):
        """Returns all views that can be reachable at on point.

        This method return any sub view provided (recursivly).

        :rtype: List[DataView]
        """
        raise NotImplementedError()

    def getMatchingViews(self, data, info):
        """Returns sub views matching this data and info.

        This method return any sub view provided (recursivly).

        :param object data: Any object to be displayed
        :param DataInfo info: Information cached about this data
        :rtype: List[DataView]
        """
        raise NotImplementedError()

    def isSupportedData(self, data, info):
        """If true, the composite view allow sub views to access to this data.
        Else this this data is considered as not supported by any of sub views
        (incliding this composite view).

        :param object data: Any object to be displayed
        :param DataInfo info: Information cached about this data
        :rtype: bool
        """
        return True


class SelectOneDataView(_CompositeDataView):
    """Data view which can display a data using different view according to
    the kind of the data."""

    def __init__(self, parent, modeId=None, icon=None, label=None):
        """Constructor

        :param qt.QWidget parent: Parent of the hold widget
        """
        super().__init__(parent, modeId, icon, label)
        self.__views = {}
        self.__currentView = None

    def getCurrentView(self):
        return self.__currentView

    def setHooks(self, hooks):
        """Set the data context to use with this view.

        :param DataViewHooks hooks: The data view hooks to use
        """
        super().setHooks(hooks)
        if hooks is not None:
            for v in self.__views:
                v.setHooks(hooks)

    def addView(self, dataView):
        """Add a new dataview to the available list."""
        hooks = self.getHooks()
        if hooks is not None:
            dataView.setHooks(hooks)
        self.__views[dataView] = None

    def getReachableViews(self):
        views = []
        addSelf = False
        for v in self.__views:
            if isinstance(v, SelectManyDataView):
                views.extend(v.getReachableViews())
            else:
                addSelf = True
        if addSelf:
            # Single views are hidden by this view
            views.insert(0, self)
        return views

    def getMatchingViews(self, data, info):
        if not self.isSupportedData(data, info):
            return []
        view = self.__getBestView(data, info)
        if isinstance(view, SelectManyDataView):
            return view.getMatchingViews(data, info)
        else:
            return [self]

    def getViews(self):
        """Returns the list of registered views

        :rtype: List[DataView]
        """
        return list(self.__views.keys())

    def __getBestView(self, data, info):
        """Returns the best view according to priorities."""
        if not self.isSupportedData(data, info):
            return None
        views = [(v.getCachedDataPriority(data, info), v) for v in self.__views.keys()]
        views = filter(lambda t: t[0] > DataView.UNSUPPORTED, views)
        views = sorted(views, key=lambda t: t[0], reverse=True)

        if len(views) == 0:
            return None
        elif views[0][0] == DataView.UNSUPPORTED:
            return None
        else:
            return views[0][1]

    def __updateDisplayedView(self):
        widget = self.getWidget()
        if self.__currentView is None:
            return

        # load the widget if it is not yet done
        index = self.__views[self.__currentView]
        if index is None:
            w = self.__currentView.getWidget()
            index = widget.addWidget(w)
            self.__views[self.__currentView] = index
        if widget.currentIndex() != index:
            widget.setCurrentIndex(index)
            self.__currentView.select()

    def select(self):
        self.__updateDisplayedView()
        if self.__currentView is not None:
            self.__currentView.select()

    def createWidget(self, parent):
        return qt.QStackedWidget()

    def clear(self):
        for v in self.__views.keys():
            v.clear()

    def setData(self, data):
        if self.__currentView is None:
            return
        self.__updateDisplayedView()
        self.__currentView.setData(data)

    def setDataSelection(self, selection):
        if self.__currentView is None:
            return
        self.__currentView.setDataSelection(selection)

    def axesNames(self, data, info):
        view = self.__getBestView(data, info)
        self.__currentView = view
        return view.axesNames(data, info)

    def getDataPriority(self, data, info):
        view = self.__getBestView(data, info)
        self.__currentView = view
        if view is None:
            return DataView.UNSUPPORTED
        else:
            return view.getCachedDataPriority(data, info)

    def replaceView(self, modeId, newView):
        """Replace a data view with a custom view.
        Return True in case of success, False in case of failure.

        .. note::

            This method must be called just after instantiation, before
            the viewer is used.

        :param int modeId: Unique mode ID identifying the DataView to
            be replaced.
        :param DataViews.DataView newView: New data view
        :return: True if replacement was successful, else False
        """
        oldView = None
        for view in self.__views:
            if view.modeId() == modeId:
                oldView = view
                break
            elif isinstance(view, _CompositeDataView):
                # recurse
                hooks = self.getHooks()
                if hooks is not None:
                    newView.setHooks(hooks)
                if view.replaceView(modeId, newView):
                    return True
        if oldView is None:
            return False

        # replace oldView with new view in dict
        self.__views = dict(
            (newView, None) if view is oldView else (view, idx)
            for view, idx in self.__views.items()
        )
        return True


# NOTE: SelectOneDataView was introduced with silx 0.10
CompositeDataView = SelectOneDataView


class SelectManyDataView(_CompositeDataView):
    """Data view which can select a set of sub views according to
    the kind of the data.

    This view itself is abstract and is not exposed.
    """

    def __init__(self, parent, views=None):
        """Constructor

        :param qt.QWidget parent: Parent of the hold widget
        """
        super().__init__(parent, modeId=None, icon=None, label=None)
        if views is None:
            views = []
        self.__views = views

    def setHooks(self, hooks):
        """Set the data context to use with this view.

        :param DataViewHooks hooks: The data view hooks to use
        """
        super().setHooks(hooks)
        if hooks is not None:
            for v in self.__views:
                v.setHooks(hooks)

    def addView(self, dataView):
        """Add a new dataview to the available list."""
        hooks = self.getHooks()
        if hooks is not None:
            dataView.setHooks(hooks)
        self.__views.append(dataView)

    def getViews(self):
        """Returns the list of registered views

        :rtype: List[DataView]
        """
        return list(self.__views)

    def getReachableViews(self):
        views = []
        for v in self.__views:
            views.extend(v.getReachableViews())
        return views

    def getMatchingViews(self, data, info):
        """Returns the views according to data and info from the data.

        :param object data: Any object to be displayed
        :param DataInfo info: Information cached about this data
        """
        if not self.isSupportedData(data, info):
            return []
        views = [
            v
            for v in self.__views
            if v.getCachedDataPriority(data, info) != DataView.UNSUPPORTED
        ]
        return views

    def select(self):
        raise RuntimeError("Abstract view")

    def createWidget(self, parent):
        raise RuntimeError("Abstract view")

    def clear(self):
        for v in self.__views:
            v.clear()

    def setData(self, data):
        raise RuntimeError("Abstract view")

    def axesNames(self, data, info):
        raise RuntimeError("Abstract view")

    def getDataPriority(self, data, info):
        if not self.isSupportedData(data, info):
            return DataView.UNSUPPORTED
        priorities = [v.getCachedDataPriority(data, info) for v in self.__views]
        priorities = [v for v in priorities if v != DataView.UNSUPPORTED]
        priorities = sorted(priorities)
        if len(priorities) == 0:
            return DataView.UNSUPPORTED
        return priorities[-1]

    def replaceView(self, modeId, newView):
        """Replace a data view with a custom view.
        Return True in case of success, False in case of failure.

        .. note::

            This method must be called just after instantiation, before
            the viewer is used.

        :param int modeId: Unique mode ID identifying the DataView to
            be replaced.
        :param DataViews.DataView newView: New data view
        :return: True if replacement was successful, else False
        """
        oldView = None
        for iview, view in enumerate(self.__views):
            if view.modeId() == modeId:
                oldView = view
                break
            elif isinstance(view, CompositeDataView):
                # recurse
                hooks = self.getHooks()
                if hooks is not None:
                    newView.setHooks(hooks)
                if view.replaceView(modeId, newView):
                    return True

        if oldView is None:
            return False

        # replace oldView with new view in dict
        self.__views[iview] = newView
        return True


class _RawView(CompositeDataView):
    """View displaying data as raw data.

    This implementation use a 2d-array view, or a record array view, or a
    raw text output.
    """

    def __init__(self, parent):
        super().__init__(
            parent=parent, modeId=RAW_MODE, label="Raw", icon=icons.getQIcon("view-raw")
        )
        self.addView(_HexaView(parent))
        self.addView(_ScalarView(parent))
        self.addView(_ArrayView(parent))
        self.addView(_RecordView(parent))
