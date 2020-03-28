import sys
import numpy


def createWindow(parent, settings):
    # Local import to avoid early import (like h5py)
    # SOme libraries have to be configured first properly
    from silx.gui.plot.actions import PlotAction
    from silx.app.view.Viewer import Viewer
    from silx.app.view.ApplicationContext import ApplicationContext

    class RandomColorAction(PlotAction):
        def __init__(self, plot, parent=None):
            super(RandomColorAction, self).__init__(
                plot, icon="colormap", text='Color',
                tooltip='Random plot background color',
                triggered=self.__randomColor,
                checkable=False, parent=parent)

        def __randomColor(self):
            color = "#%06X" % numpy.random.randint(0xFFFFFF)
            self.plot.setBackgroundColor(color)

    class MyApplicationContext(ApplicationContext):
        """This class is shared to all the silx view application."""
    
        def findPrintToolBar(self, plot):
            # FIXME: It would be better to use the Qt API
            return plot._outputToolBar

        def viewWidgetCreated(self, view, widget):
            """Called when the widget of the view was created.

            So we can custom it.
            """
            from silx.gui.plot import Plot1D
            if isinstance(widget, Plot1D):
                toolBar = self.findPrintToolBar(widget)
                action = RandomColorAction(widget, widget)
                toolBar.addAction(action)

    class MyViewer(Viewer):
        def createApplicationContext(self, settings):
            return MyApplicationContext(self, settings)

    window = MyViewer(parent=parent, settings=settings)
    window.setWindowTitle(window.windowTitle() + " [custom]")
    return window


def main(args):
    from silx.app.view import main as silx_view_main
    # Monkey patch the main window creation
    silx_view_main.createWindow = createWindow
    # Use the default launcher
    silx_view_main.main(args)


if __name__ == '__main__':
    main(sys.argv)
