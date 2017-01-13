

from silx.gui import qt


# define custom fit config dialog
class PolyDegreeConfig(qt.QDialog):
    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setModal(True)

        label = qt.QLabel("Enter order of polynomial")

        self.polyDegEdit = qt.QLineEdit(self)
        self.polyDegEdit.setToolTip(
            "Order of polynomial function"
        )
        self.polyDegEdit.setValidator(qt.QIntValidator())

        self.ok = qt.QPushButton("ok", self)
        self.ok.clicked.connect(self.accept)
        cancel = qt.QPushButton("cancel", self)
        cancel.clicked.connect(self.reject)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.polyDegEdit)
        layout.addWidget(self.ok)
        layout.addWidget(cancel)

        self.output = {}

    def setDefault(self, default_dict):
        if default_dict is not None:
            self.output.update(default_dict)

    def accept(self):
        self.output["PolyDeg"] = int(self.polyDegEdit.text())
        qt.QDialog.accept(self)

    def reject(self):
        self.output = {}
        qt.QDialog.reject(self)


def getPolyDialog(parent=None, default=None):
    """Return an instance of :class:`PolyDegreeConfig`"""
    dialog = PolyDegreeConfig(parent)
    dialog.setDefault(default)
    return dialog
