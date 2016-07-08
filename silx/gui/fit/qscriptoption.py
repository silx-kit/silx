#/*##########################################################################
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
# #########################################################################*/
"""
This module is a refactored version of
*PyMca5.PyMcaGui.math.fitting.QScriptOption*

It defines a widget with customizable tabs, storing all user input in an
internal dictionary.
"""
import sys
from collections import OrderedDict
from silx.gui import qt

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "08/07/2016"


QTVERSION = qt.qVersion()

_tuple_type = type(())

# def uic_load_pixmap_RadioField(name):
#     pix = qt.QPixmap()
#     m = qt.QMimeSourceFactory.defaultFactory().data(name)
#
#     if m:
#         qt.QImageDrag.decode(m, pix)
#
#     return pix


class TabSheets(qt.QDialog):
    """QDialog widget with a variable number of tabs and
    a few predefined optional buttons (*OK, Cancel, Help, Defaults)* at the
    bottom.

    QPushButton attributes:

        - :attr:`buttonHelp`
        - :attr:`buttonDefaults`
        - :attr:`buttonOk` (connected to :meth:`QDialog.accept`)
        - :attr:`buttonCancel` (connected to :meth:`QDialog.reject`)

    QTabWidget:
        - :attr:`tabWidget`

    """
    def __init__(self, parent=None, modal=False, nohelp=True, nodefaults=True):
        """

        :param parent: parent widget
        :param modal: If ``True``, make dialog modal (block input to other
            visible windows). Default is ``False``.
        :param nohelp: If ``True`` (default), don't add *help* button
        :param nodefaults: If ``True`` (default), don't add *Defaults* button
        """
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle(str("TabSheets"))
        self.setModal(modal)

        tabsheetslayout = qt.QVBoxLayout(self)
        tabsheetslayout.setContentsMargins(11, 11, 11, 11)
        tabsheetslayout.setSpacing(6)

        self.tabWidget = qt.QTabWidget(self)

        self.Widget8 = qt.QWidget(self.tabWidget)
        self.Widget9 = qt.QWidget(self.tabWidget)
        self.tabWidget.addTab(self.Widget8, str("Tab"))
        self.tabWidget.addTab(self.Widget9, str("Tab"))

        tabsheetslayout.addWidget(self.tabWidget)

        layout2 = qt.QHBoxLayout(None)
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.setSpacing(6)

        if not nohelp:
            self.buttonHelp = qt.QPushButton(self)
            self.buttonHelp.setText(str("Help"))
            layout2.addWidget(self.buttonHelp)

        if not nodefaults:
            self.buttonDefaults = qt.QPushButton(self)
            self.buttonDefaults.setText(str("Defaults"))
            layout2.addWidget(self.buttonDefaults)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        layout2.addItem(spacer)

        self.buttonOk = qt.QPushButton(self)
        self.buttonOk.setText(str("OK"))
        layout2.addWidget(self.buttonOk)

        self.buttonCancel = qt.QPushButton(self)
        self.buttonCancel.setText(str("Cancel"))
        layout2.addWidget(self.buttonCancel)
        tabsheetslayout.addLayout(layout2)

        self.buttonOk.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)


class QScriptOption(TabSheets):
    """Subclass of :class:`TabSheets` with added feature of defining
    content (entry fields, text, radio buttons) to add to the tabs.

    """

    def __init__(self, parent=None, name=None, modal=True,
                 sheets=(), default=None, nohelp=True, nodefaults=True):
        """

        :param parent: Parent widget
        :param name: Window title. If ``None``, use *"TabSheets"*
        :param modal: If ``True``, make dialog modal (block input to other
            visible windows).
        :param sheets: Tuple of dictionaries containing parameters for
            sheets/tabs.
            An example of valid dictionary illustrating the
            format is::

                {'notetitle': "First Sheet",
                 'fields': (["Label", 'Simple Entry'],
                            ["EntryField", 'entry', 'MyLabel'],
                            ["CheckField", 'label', 'Check Label'])}

            The string in the ``notetitle`` item is used as sheet/tab name.
            The ``fields`` item is used as a parameter for :class:`FieldSheet`.

        :param default: Default dictionary
        :param nohelp: If ``True``, don't add *help* button
        :param nodefaults: If ``True``, don't add *Defaults* button
        :param name: Window title. If ``None``, use *"TabSheets"*
        """
        TabSheets.__init__(self, parent, modal,
                           nohelp, nodefaults)
        if default is None or not hasattr(default, "keys"):
            default = {}

        if name is not None:
            self.setWindowTitle(str(name))

        self.sheets = OrderedDict()
        """Ordered dictionary indexed by tab/sheet names , and containing
        :class:`FieldSheet` objects
        """

        self.default = default
        """Default dictionary used to reinitialize :attr:`output` at init
        and whenever :meth:`defaults` is called
        (when *Defaults* or *Cancel* button is clicked).
        """

        self.output = {}
        """Output dictionary storing user input from  all fields contained
        in the sheets.
        """
        self.output.update(self.default)

        # remove any tabs initially present (2 placeholder tabs added in
        # TabSheets)
        ntabs = self.tabWidget.count()
        for i in range(ntabs):
            self.tabWidget.setCurrentIndex(0)
            self.tabWidget.removeTab(self.tabWidget.currentIndex())

        # Add sheets specified in parameters
        for sheet in sheets:
            name = sheet['notetitle']
            a = FieldSheet(fields=sheet['fields'])
            self.sheets[name] = a
            a.setdefaults(self.default)
            self.tabWidget.addTab(self.sheets[name], str(name))
            if QTVERSION < '4.2.0':
                i = self.tabWidget.indexOf(self.sheets[name])
                self.tabWidget.setCurrentIndex(i)
            else:
                self.tabWidget.setCurrentWidget(self.sheets[name])

        # perform the binding to the buttons
        self.buttonOk.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)
        if not nodefaults:
            self.buttonDefaults.clicked.connect(self.defaults)
        if not nohelp:
            self.buttonHelp.clicked.connect(self.myhelp)

    def accept(self):
        """When *OK* is clicked, update :attr:`output` with data from
        :attr:`sheets` (user input)"""
        self.output.update(self.default)
        for name, sheet in self.sheets.items():
            self.output.update(sheet.get())

        # avoid pathological None cases
        for key in list(self.output.keys()):
            if self.output[key] is None:
                if key in self.default:
                    self.output[key] = self.default[key]
        super(QScriptOption, self).accept()

    def reject(self):
        """When *Cancel is clicked, reinitialize :attr:`output` and quit
        """
        self.default()
        super(QScriptOption, self).reject()

    def defaults(self):
        """Reinitialize :attr:`output` with :attr:`default`
        """
        self.output = {}
        self.output.update(self.default)

    def myhelp(self):
        """Print help to standard output when the *Help* button is clicked.
        """
        print("Default - Sets back to the initial parameters")
        print("Cancel  - Sets back to the initial parameters and quits")
        print("OK      - Updates the parameters and quits")


class FieldSheet(qt.QWidget):
    """Widget displaying a variable number of fields in a vertical layout.
    """
    def __init__(self, parent=None, fields=()):
        """

        :param parent: Parent widget
        :param fields: Tuple of lists defining fields::

               [field_type, key, parameters]

            ``field_type`` can be *"Label", "CheckField", or "EntryField"*

            ``key`` is a
        """
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # self.fields = ([,,,])
        self.fields = []
        self.nbfield = 1
        for field in fields:
            fieldtype = field[0]
            key = field[1] if len(field) == 3 else None
            parameters = field[-1]

            myfield = None
            if fieldtype == "Label":
                myfield = MyLabel(self, keys=key, params=parameters)
            elif fieldtype == "CheckField":
                myfield = MyCheckField(self, keys=key, params=parameters)
            elif fieldtype == "EntryField":
                myfield = MyEntryField(self, keys=key, params=parameters)
            # elif fieldtype == "RadioField":
            #     myfield = RadioField(self, keys=key, params=parameters)

            if myfield is not None:
                self.fields.append(myfield)
                layout.addWidget(myfield)

    def get(self):
        """Return a dictionary with all values stored in the various fields
        """
        result = {}
        for field in self.fields:
            result.update(field.getvalue())
        return result

    def setdefaults(self, dict):
        """Set all fields with values from a dictionary.

        :param dict: Dictionary of values to be updated in fields with
            matching keys.
        """
        for field in self.fields:
            field.setdefaults(dict)


class Label(qt.QWidget):
    """Simple text label inside a QWidget

    :attr:`TextLabel` is the QLabel widget"""
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.resize(373, 44)

        textfieldlayout = qt.QHBoxLayout(self)
        layout2 = qt.QHBoxLayout(None)
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.setSpacing(6)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding, qt.QSizePolicy.Minimum)
        layout2.addItem(spacer)

        self.TextLabel = qt.QLabel(self)

        self.TextLabel.setText(str("TextLabel"))
        layout2.addWidget(self.TextLabel)
        spacer_2 = qt.QSpacerItem(20, 20,
                                  qt.QSizePolicy.Expanding, qt.QSizePolicy.Minimum)
        layout2.addItem(spacer_2)
        textfieldlayout.addLayout(layout2)


class MyLabel(Label):
    """Simple label with dummy methods to conform to the interface required
    by :class:`FieldSheet`"""
    def __init__(self, parent=None,
                 keys=(), params=()):
        Label.__init__(self, parent)
        self.TextLabel.setText(str(params))

    def getvalue(self):
        """return empty dict"""
        return {}

    def setvalue(self):
        """pass"""
        pass

    def setdefaults(self, dict):
        """pass"""
        pass


class EntryField(qt.QWidget):
    """Entry field with a QLineEdit (:attr:`Entry`) and a QLabel
    (:attr:`TextLabel`)"""
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        layout1 = qt.QHBoxLayout(self)

        self.TextLabel = qt.QLabel(self)
        self.TextLabel.setText("TextLabel")

        self.Entry = qt.QLineEdit(self)
        layout1.addWidget(self.TextLabel)
        layout1.addWidget(self.Entry)

# FIXME: why do we need multiple keys in each field when they are all updated
# with the same value?
class MyEntryField(EntryField):
    """Entry field with a QLineEdit (:attr:`Entry`), a QLabel
    (:attr:`TextLabel`), and 3 methods to interact with
    :class:`FieldSheet`: :meth:`getvalue`, :meth:`setvalue` and
    :meth:`setdefaults`

    These methods can be used to get or set the internal dictionary
    storing user input from the entry field."""

    def __init__(self, parent=None,
                 keys=(), params=()):
        """

        :param parent: Parent widget
        :param keys: Keys of :attr:`dict`
        :param params: Text to be displayed in the label.
        """
        EntryField.__init__(self, parent)
        self.dict = {}
        """Dictionary storing user input"""
        if type(keys) == _tuple_type:
            for key in keys:
                self.dict[key] = None
        else:
            self.dict[keys] = None
        self.TextLabel.setText(str(params))
        self.Entry.textChanged[str].connect(self.setvalue)

    def getvalue(self):
        """Return :attr:`dict`"""
        return self.dict

    def setvalue(self, value):
        """Update all values in :attr:`dict` with ``value``"""
        for key in self.dict.keys():
            self.dict[key] = str(value)

    def setdefaults(self, ddict):
        """Update values in :attr:`dict` with values in
        ``ddict`` if keys match, then update the entry
        value with each value."""
        for key in list(self.dict.keys()):
            if key in ddict:
                self.dict[key] = ddict[key]
                # This will probably trigger setvalue which updates all
                # values to the same value, so at the end I expect all
                # values to be equal to the las one. Do we really want this?
                self.Entry.setText(str(ddict[key]))


class CheckField(qt.QWidget):
    """Check field with a QCheckBox (:attr:`CheckBox`) """
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.resize(321, 45)

        CheckFieldLayout = qt.QHBoxLayout(self)
        CheckFieldLayout.setContentsMargins(11, 11, 11, 11)
        CheckFieldLayout.setSpacing(6)

        self.CheckBox = qt.QCheckBox(self)
        self.CheckBox.setText("CheckBox")
        CheckFieldLayout.addWidget(self.CheckBox)


class MyCheckField(CheckField):
    """Check field with a QCheckBox (:attr:`CheckBox`) and 3 methods to
    interact with :class:`FieldSheet`: :meth:`getvalue`, :meth:`setvalue` and
    :meth:`setdefaults`

    These methods can be used to get or set the internal dictionary
    storing user input from the entry field."""
    def __init__(self, parent=None,
                 keys=(), params=()):
        """

        :param parent: Parent widget
        :param keys: Keys of :attr:`dict`
        :param params: Text to be displayed in the label.
        """
        CheckField.__init__(self, parent)
        self.dict = {}
        """Dictionary storing user input"""
        if type(keys) == _tuple_type:
            for key in keys:
                self.dict[key] = None
        else:
            self.dict[keys] = None
        self.CheckBox.setText(str(params))
        self.CheckBox.stateChanged[int].connect(self.setvalue)

    def getvalue(self):
        """Return :attr:`dict`"""
        return self.dict

    def setvalue(self, value):
        """Update all values in :attr:`dict` with 0 if the checkbox
        has been un-ticked or 1 if it has been ticked"""
        if value:
            val = 1
        else:
            val = 0
        for key in self.dict.keys():
            self.dict[key] = val

    def setdefaults(self, ddict):
        """Update values in :attr:`dict` with values in
        ``ddict`` if keys match, then update the checkbox
        with each value.

        :param ddict: Dictionary whose values must be integers
            or convertible to integers. All values which don't
            convert to zero will result in the corresponding key
            being set to 1 in :attr:`dict`"""
        for key in self.dict.keys():
            if key in ddict:
                if int(ddict[key]):
                    self.CheckBox.setChecked(1)
                    self.dict[key] = 1
                else:
                    self.CheckBox.setChecked(0)
                    self.dict[key] = 0
#
# # FIXME: deactivated, does not work (pyqt3?)
# class RadioField(qt.QWidget):
#     def __init__(self,parent = None,
#                  keys=(), params = ()):
#             qt.QWidget.__init__(self,parent)
#             RadioFieldLayout = qt.QHBoxLayout(self)
#             RadioFieldLayout.setContentsMargins(11, 11, 11, 11)
#             RadioFieldLayout.setSpacing(6)
#
#             self.RadioFieldBox = qt.QButtonGroup(self)
#             self.RadioFieldBox.setColumnLayout(0,qt.Qt.Vertical)
#             self.RadioFieldBox.layout().setSpacing(6)
#             self.RadioFieldBox.layout().setContentsMargins(11, 11, 11, 11)
#             RadioFieldBoxLayout = qt.QVBoxLayout(self.RadioFieldBox.layout())
#             RadioFieldBoxLayout.setAlignment(qt.Qt.AlignTop)
#             Layout1 = qt.QVBoxLayout(None, 0, 6, "Layout1")
#
#             self.dict={}
#             if type(keys) == _tuple_type:
#                 for key in keys:
#                     self.dict[key]=1
#             else:
#                 self.dict[keys]=1
#             self.RadioButton=[]
#             i=0
#             for text in params:
#                 self.RadioButton.append(qt.QRadioButton(self.RadioFieldBox,
#                                                         "RadioButton"+"%d" % i))
#                 self.RadioButton[-1].setSizePolicy(qt.QSizePolicy(1,1,0,0,
#                                 self.RadioButton[-1].sizePolicy().hasHeightForWidth()))
#                 self.RadioButton[-1].setText(str(text))
#                 Layout1.addWidget(self.RadioButton[-1])
#                 i=i+1
#
#             RadioFieldBoxLayout.addLayout(Layout1)
#             RadioFieldLayout.addWidget(self.RadioFieldBox)
#             self.RadioButton[0].setChecked(1)
#             self.RadioFieldBox.clicked[int].connect(self.setvalue)
#
#     def getvalue(self):
#         return self.dict
#
#     def setvalue(self,value):
#         if value:
#             val=1
#         else:
#             val=0
#         for key in self.dict.keys():
#             self.dict[key]=val
#         return
#
#     def setdefaults(self, ddict):
#         for key in list(self.dict.keys()):
#             if key in ddict:
#                 self.dict[key]=ddict[key]
#                 i=int(ddict[key])
#                 self.RadioButton[i].setChecked(1)
#         return


def test():
    a = qt.QApplication(sys.argv)
    w = FieldSheet(fields=(["Label",'Simple Entry'],
                          ["EntryField", 'entry', 'MyLabel'],
                          ["CheckField", 'label', 'Check Label'],))
                          # ["RadioField",'radio',('Button1','hmmm','3')]))
    sheet1 = {'notetitle': "First Sheet",
              'fields': (["Label", 'Simple Entry'],
                         ["EntryField", 'entry', 'MyLabel'],
                         ["CheckField", 'label', 'Check Label'])}
    sheet2 = {'notetitle': "Second Sheet",
              'fields': (["Label", 'Simple Radio Buttons'],
                         ["EntryField", 'entry2', 'MyLabel2'],
                         ["CheckField", 'label2', 'Check Label2'],
                         ["EntryField", 'entry3', 'MyLabel3'],
                         ["CheckField", 'label3', 'Check Label3'])}
    w = QScriptOption(name='QScriptOptions', sheets=(sheet1, sheet2),
                      default={'entry': 'type here', 'label': 1, "label3": 0,
                               'entry3': "chanson d'automne"})

    w.show()
    a.exec_()
    print(w.output)

if __name__ == "__main__":
    test()
