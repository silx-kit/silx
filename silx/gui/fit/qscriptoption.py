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
"""
__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "30/06/2016"
import sys
#from silx.gui import qt
from PyMca5.PyMcaGui import PyMcaQt as qt


QTVERSION = qt.qVersion()

_tuple_type = type(())


def uic_load_pixmap_RadioField(name):
    pix = qt.QPixmap()
    m = qt.QMimeSourceFactory.defaultFactory().data(name)

    if m:
        qt.QImageDrag.decode(m, pix)

    return pix


class TabSheets(qt.QDialog):

    def __init__(self, parent=None, name=None, modal=0, nohelp=1, nodefaults=1):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle(str("TabSheets"))
        self.setModal(modal)

        TabSheetsLayout = qt.QVBoxLayout(self)
        TabSheetsLayout.setContentsMargins(11, 11, 11, 11)
        TabSheetsLayout.setSpacing(6)

        self.tabWidget = qt.QTabWidget(self)

        self.Widget8 = qt.QWidget(self.tabWidget)
        self.Widget9 = qt.QWidget(self.tabWidget)
        self.tabWidget.addTab(self.Widget8, str("Tab"))
        self.tabWidget.addTab(self.Widget9, str("Tab"))

        TabSheetsLayout.addWidget(self.tabWidget)

        Layout2 = qt.QHBoxLayout(None)
        Layout2.setContentsMargins(0, 0, 0, 0)
        Layout2.setSpacing(6)

        if not nohelp:
            self.buttonHelp = qt.QPushButton(self)
            self.buttonHelp.setText(str("Help"))
            Layout2.addWidget(self.buttonHelp)

        if not nodefaults:
            self.buttonDefaults = qt.QPushButton(self)
            self.buttonDefaults.setText(str("Defaults"))
            Layout2.addWidget(self.buttonDefaults)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        Layout2.addItem(spacer)

        self.buttonOk = qt.QPushButton(self)
        self.buttonOk.setText(str("OK"))
        Layout2.addWidget(self.buttonOk)

        self.buttonCancel = qt.QPushButton(self)
        self.buttonCancel.setText(str("Cancel"))
        Layout2.addWidget(self.buttonCancel)
        TabSheetsLayout.addLayout(Layout2)

        self.buttonOk.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)


class QScriptOption(TabSheets):

    def __init__(self, parent=None, name=None, modal=1,
                 sheets=(), default=None, nohelp=1, nodefaults=1):
        TabSheets.__init__(self, parent, name, modal,
                           nohelp, nodefaults)

        if name is not None:
            self.setWindowTitle(str(name))

        self.sheets = {}
        self.sheetslist = []
        self.default = default
        self.output = {}
        self.output.update(self.default)
        ntabs = self.tabWidget.count()

        # remove anything not having to do with my sheets
        for i in range(ntabs):
            self.tabWidget.setCurrentIndex(0)
            self.tabWidget.removeTab(self.tabWidget.currentIndex())

        for sheet in sheets:
            name = sheet['notetitle']
            a = FieldSheet(fields=sheet['fields'])
            self.sheets[name] = a
            a.setdefaults(self.default)
            self.sheetslist.append(name)
            self.tabWidget.addTab(self.sheets[name], str(name))
            if QTVERSION < '4.2.0':
                i = self.tabWidget.indexOf(self.sheets[name])
                self.tabWidget.setCurrentIndex(i)
            else:
                self.tabWidget.setCurrentWidget(self.sheets[name])
        # perform the binding to the buttons
        self.buttonOk.clicked.connect(self.myaccept)
        self.buttonCancel.clicked.connect(self.myreject)
        if not nodefaults:
            self.buttonDefaults.clicked.connect(self.defaults)
        if not nohelp:
            self.buttonHelp.clicked.connect(self.myhelp)

    def myaccept(self):
        self.output.update(self.default)
        for name, sheet in self.sheets.items():
            self.output.update(sheet.get())
        # avoid pathologicval None cases
        for key in list(self.output.keys()):
            if self.output[key] is None:
                if key in self.default:
                    self.output[key] = self.default[key]

        self.accept()
        return

    def myreject(self):
        self.output = {}
        self.output.update(self.default)
        self.reject()
        return

    def defaults(self):
        self.output = {}
        self.output.update(self.default)

    def myhelp(self):
        print("Default - Sets back to the initial parameters")
        print("Cancel  - Sets back to the initial parameters and quits")
        print("OK      - Updates the parameters and quits")


class FieldSheet(qt.QWidget):

    def __init__(self, parent=None, fields=()):
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # self.fields = ([,,,])
        self.fields = []
        self.nbfield = 1
        for field in fields:
            fieldtype = field[0]
            if len(field) == 3:
                key = field[1]
            else:
                key = None
            parameters = field[-1]
            if fieldtype == "TextField":
                myTextField = MyTextField(self, keys=key, params=parameters)
                self.fields.append(myTextField)
                layout.addWidget(myTextField)
            if fieldtype == "CheckField":
                myCheckField = MyCheckField(self, keys=key, params=parameters)
                self.fields.append(myCheckField)
                layout.addWidget(myCheckField)
            if fieldtype == "EntryField":
                myEntryField = MyEntryField(self, keys=key, params=parameters)
                self.fields.append(myEntryField)
                layout.addWidget(myEntryField)
            # if fieldtype == "RadioField":
            #     radioField = RadioField(self,keys=key,params=parameters)
            #     self.fields.append(radioField)
            #     layout.addWidget(radioField)

    def get(self):
        result = {}
        for field in self.fields:
            result.update(field.getvalue())
        return result

    def setdefaults(self, dict):
        for field in self.fields:
            field.setdefaults(dict)
        return


class TextField(qt.QWidget):

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.resize(373, 44)
        # try:
        #     self.setSizePolicy(qt.QSizePolicy(1,1,0,0,self.sizePolicy().hasHeightForWidth()))
        # except:
        #     print("TextField Bad Size policy")

        TextFieldLayout = qt.QHBoxLayout(self)
        Layout2 = qt.QHBoxLayout(None)
        Layout2.setContentsMargins(0, 0, 0, 0)
        Layout2.setSpacing(6)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding, qt.QSizePolicy.Minimum)
        Layout2.addItem(spacer)

        self.TextLabel = qt.QLabel(self)
        # try:
        #     self.TextLabel.setSizePolicy(qt.QSizePolicy(7,1,0,0,self.TextLabel.sizePolicy().hasHeightForWidth()))
        # except:
        #     print("TextField Bad Size policy")

        self.TextLabel.setText(str("TextLabel"))
        Layout2.addWidget(self.TextLabel)
        spacer_2 = qt.QSpacerItem(20, 20,
                                  qt.QSizePolicy.Expanding, qt.QSizePolicy.Minimum)
        Layout2.addItem(spacer_2)
        TextFieldLayout.addLayout(Layout2)


class MyTextField(TextField):

    def __init__(self, parent=None,
                 keys=(), params=()):
        TextField.__init__(self, parent)
        self.TextLabel.setText(str(params))

    def getvalue(self):
        pass
        return

    def setvalue(self):
        pass
        return

    def setdefaults(self, dict):
        pass
        return


class EntryField(qt.QWidget):

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        Layout1 = qt.QHBoxLayout(self)

        self.TextLabel = qt.QLabel(self)
        self.TextLabel.setText("TextLabel")

        self.Entry = qt.QLineEdit(self)
        Layout1.addWidget(self.TextLabel)
        Layout1.addWidget(self.Entry)


class MyEntryField(EntryField):

    def __init__(self, parent=None,
                 keys=(), params=()):
        EntryField.__init__(self, parent)
        self.dict = {}
        if type(keys) == _tuple_type:
            for key in keys:
                self.dict[key] = None
        else:
            self.dict[keys] = None
        self.TextLabel.setText(str(params))
        self.Entry.textChanged[str].connect(self.setvalue)

    def getvalue(self):
        return self.dict

    def setvalue(self, value):
        for key in self.dict.keys():
            self.dict[key] = str(value)
        return

    def setdefaults(self, ddict):
        for key in list(self.dict.keys()):
            if key in ddict:
                self.dict[key] = ddict[key]
                self.Entry.setText(str(ddict[key]))
        return


class CheckField(qt.QWidget):

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

    def __init__(self, parent=None,
                 keys=(), params=()):
        CheckField.__init__(self, parent)
        self.dict = {}
        if type(keys) == _tuple_type:
            for key in keys:
                self.dict[key] = None
        else:
            self.dict[keys] = None
        self.CheckBox.setText(str(params))
        self.CheckBox.stateChanged[int].connect(self.setvalue)

    def getvalue(self):
        return self.dict

    def setvalue(self, value):
        if value:
            val = 1
        else:
            val = 0
        for key in self.dict.keys():
            self.dict[key] = val
        return

    def setdefaults(self, ddict):
        for key in self.dict.keys():
            if key in ddict:
                if int(ddict[key]):
                    self.CheckBox.setChecked(1)
                    self.dict[key] = 1
                else:
                    self.CheckBox.setChecked(0)
                    self.dict[key] = 0
        return

# FIXME: deactivated, does not work (pyqt3?)
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
#             Layout1 = qt.QVBoxLayout(None,0,6,"Layout1")
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
    app.lastWindowClosed.connect(app.quit)
    # w = FieldSheet(fields=(["TextField",'Simple Entry'],
    #                       ["EntryField",'entry','MyLabel'],
    #                       ["CheckField",'label','Check Label'],
    #                       ["RadioField",'radio',('Button1','hmmm','3')]))
    sheet1 = {'notetitle': "First Sheet",
              'fields': (["TextField", 'Simple Entry'],
                         ["EntryField", 'entry', 'MyLabel'],
                         ["CheckField", 'label', 'Check Label'])}
    sheet2 = {'notetitle': "Second Sheet",
              'fields': (["TextField", 'Simple Radio Buttons'],
                         ["RadioField", 'radio', ('Button1', 'hmmm', '3')])}
    w = QScriptOption(name='QScriptOptions', sheets=(sheet1, sheet2),
                      default={'radio': 1, 'entry': 'type here', 'label': 1})

    w.show()
    a.exec_()
    print(w.output)

if __name__ == "__main__":
    test()
