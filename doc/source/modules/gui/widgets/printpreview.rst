
.. currentmodule:: silx.gui.widgets

:mod:`PrintPreview`: Print preview dialog
-----------------------------------------

.. automodule:: silx.gui.widgets.PrintPreview

Widgets
+++++++

.. autoclass:: silx.gui.widgets.PrintPreview.PrintPreviewDialog
    :members:
    :exclude-members: printDialog, showEvent
    :show-inheritance:


.. autoclass:: silx.gui.widgets.PrintPreview.SingletonPrintPreviewDialog
    :show-inheritance:

Example
+++++++

.. code-block:: python

    import sys
    from silx.gui import qt
    from silx.gui.widgets import PrintPreviewDialog

    a = qt.QApplication(sys.argv)

    if len(sys.argv) < 2:
        print("give an image file as parameter please.")
        sys.exit(1)

    if len(sys.argv) > 2:
        print("only one parameter please.")
        sys.exit(1)

    filename = sys.argv[1]
    w = PrintPreviewDialog()
    w.resize(400, 500)

    comment = ""
    for i in range(20):
        comment += "Line number %d: En un lugar de La Mancha de cuyo nombre ...\n"

    if filename[-3:] == "svg":
        item = qt.QSvgRenderer(filename, w.page)
        w.addSvgItem(item, title=filename,
                     comment=comment, commentPosition="CENTER")
    else:
        w.addPixmap(qt.QPixmap.fromImage(qt.QImage(filename)),
                    title=filename,
                    comment=comment,
                    commentPosition="CENTER")
        w.addImage(qt.QImage(filename), comment=comment, commentPosition="LEFT")

    w.exec_()
    a.exec_()
