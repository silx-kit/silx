#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import os

from PyMca5.PyMcaGui import PyMcaQt as qt
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = qt.safe_str

QTVERSION = qt.qVersion()

from PyMca5.PyMcaCore import PyMcaDirs
from PyMca5.PyMcaIO import ConfigDict

DEBUG = 0
class McaROIWidget(qt.QWidget):
    sigMcaROIWidgetSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, name=None):
        super(McaROIWidget, self).__init__(parent)
        if name is not None:
            self.setWindowTitle(name)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        ##############
        self.headerLabel = qt.QLabel(self)
        self.headerLabel.setAlignment(qt.Qt.AlignHCenter)
        self.setHeader('<b>Channel ROIs of XXXXXXXXXX<\b>')
        layout.addWidget(self.headerLabel)
        ##############
        self.mcaROITable     = McaROITable(self)
        rheight = self.mcaROITable.horizontalHeader().sizeHint().height()
        self.mcaROITable.setMinimumHeight(4*rheight)
        #self.mcaROITable.setMaximumHeight(4*rheight)
        self.fillFromROIDict = self.mcaROITable.fillFromROIDict
        self.addROI          = self.mcaROITable.addROI
        self.getROIListAndDict=self.mcaROITable.getROIListAndDict
        layout.addWidget(self.mcaROITable)
        self.roiDir = None
        #################

        hbox = qt.QWidget(self)
        hboxlayout = qt.QHBoxLayout(hbox)
        hboxlayout.setContentsMargins(0, 0, 0, 0)
        hboxlayout.setSpacing(0)

        hboxlayout.addWidget(qt.HorizontalSpacer(hbox))

        self.addButton = qt.QPushButton(hbox)
        self.addButton.setText("Add ROI")
        self.delButton = qt.QPushButton(hbox)
        self.delButton.setText("Delete ROI")
        self.resetButton = qt.QPushButton(hbox)
        self.resetButton.setText("Reset")

        hboxlayout.addWidget(self.addButton)
        hboxlayout.addWidget(self.delButton)
        hboxlayout.addWidget(self.resetButton)
        hboxlayout.addWidget(qt.HorizontalSpacer(hbox))

        self.loadButton = qt.QPushButton(hbox)
        self.loadButton.setText("Load")
        self.saveButton = qt.QPushButton(hbox)
        self.saveButton.setText("Save")
        hboxlayout.addWidget(self.loadButton)
        hboxlayout.addWidget(self.saveButton)
        layout.setStretchFactor(self.headerLabel, 0)
        layout.setStretchFactor(self.mcaROITable, 1)
        layout.setStretchFactor(hbox, 0)

        layout.addWidget(hbox)

        self.addButton.clicked.connect(self._add)
        self.delButton.clicked.connect(self._del)
        self.resetButton.clicked.connect(self._reset)

        self.loadButton.clicked.connect(self._load)
        self.saveButton.clicked.connect(self._save)
        self.mcaROITable.sigMcaROITableSignal.connect(self._forward)

    def _add(self):
        if DEBUG:
            print("McaROIWidget._add")
        ddict={}
        ddict['event']   = "AddROI"
        roilist, roidict  = self.mcaROITable.getROIListAndDict()
        ddict['roilist'] = roilist
        ddict['roidict'] = roidict
        self.emitSignal(ddict)

    def _del(self):
        row = self.mcaROITable.currentRow()
        if row >= 0:
            index = self.mcaROITable.labels.index('Type')
            text = str(self.mcaROITable.item(row, index).text())
            if text.upper() != 'DEFAULT':
                index = self.mcaROITable.labels.index('ROI')
                key = str(self.mcaROITable.item(row, index).text())
            else:
                # This is to prevent deleting ICR ROI, that is
                # usually initialized as "Default" type.
                return
            roilist,roidict    = self.mcaROITable.getROIListAndDict()
            row = roilist.index(key)
            del roilist[row]
            del roidict[key]
            if len(roilist) > 0:
                currentroi = roilist[0]
            else:
                currentroi = None

            self.mcaROITable.fillFromROIDict(roilist=roilist,
                                             roidict=roidict,
                                             currentroi=currentroi)
            ddict={}
            ddict['event']      = "DelROI"
            ddict['roilist']    = roilist
            ddict['roidict']    = roidict
            self.emitSignal(ddict)

    def _forward(self,ddict):
        self.emitSignal(ddict)

    def _reset(self):
        ddict={}
        ddict['event']   = "ResetROI"
        roilist0, roidict0  = self.mcaROITable.getROIListAndDict()
        index = 0
        for key in roilist0:
            if roidict0[key]['type'].upper() == 'DEFAULT':
                index = roilist0.index(key)
                break
        roilist=[]
        roidict = {}
        if len(roilist0):
            roilist.append(roilist0[index])
            roidict[roilist[0]] = {}
            roidict[roilist[0]].update(roidict0[roilist[0]])
            self.mcaROITable.fillFromROIDict(roilist=roilist,
                                             roidict=roidict)
        ddict['roilist'] = roilist
        ddict['roidict'] = roidict
        self.emitSignal(ddict)

    def _load(self):
        if self.roiDir is None:
            self.roiDir = PyMcaDirs.inputDir
        elif not os.path.isdir(self.roiDir):
            self.roiDir = PyMcaDirs.inputDir
        outfile = qt.QFileDialog(self)
        if hasattr(outfile, "setFilters"):
            outfile.setFilter('PyMca  *.ini')
        else:
            outfile.setNameFilters(['PyMca  *.ini', 'All *'])
        outfile.setFileMode(outfile.ExistingFile)
        outfile.setDirectory(self.roiDir)
        ret = outfile.exec_()
        if not ret:
            outfile.close()
            del outfile
            return
        # pyflakes bug http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=666494
        outputFile = qt.safe_str(outfile.selectedFiles()[0])
        outfile.close()
        del outfile
        self.roiDir = os.path.dirname(outputFile)
        self.load(outputFile)

    def load(self, filename):
        d = ConfigDict.ConfigDict()
        d.read(filename)
        current = ""
        if self.mcaROITable.rowCount():
            row = self.mcaROITable.currentRow()
            item = self.mcaROITable.item(row, 0)
            if item is not None:
                current = str(item.text())
        self.fillFromROIDict(roilist=d['ROI']['roilist'],
                             roidict=d['ROI']['roidict'])
        if current in d['ROI']['roidict'].keys():
            if current in d['ROI']['roilist']:
                row = d['ROI']['roilist'].index(current, 0)
                self.mcaROITable.setCurrentCell(row, 0)
                self.mcaROITable._cellChangedSlot(row, 2)
                return
        self.mcaROITable.setCurrentCell(0, 0)
        self.mcaROITable._cellChangedSlot(0, 2)

    def _save(self):
        if self.roiDir is None:
            self.roiDir = PyMcaDirs.outputDir
        elif not os.path.isdir(self.roiDir):
            self.roiDir = PyMcaDirs.outputDir
        outfile = qt.QFileDialog(self)
        if hasattr(outfile, "setFilters"):
            outfile.setFilter('PyMca  *.ini')
        else:
            outfile.setNameFilters(['PyMca  *.ini', 'All *'])
        outfile.setFileMode(outfile.AnyFile)
        outfile.setAcceptMode(qt.QFileDialog.AcceptSave)
        outfile.setDirectory(self.roiDir)
        ret = outfile.exec_()
        if not ret:
            outfile.close()
            del outfile
            return
        # pyflakes bug http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=666494
        outputFile = qt.safe_str(outfile.selectedFiles()[0])
        extension = ".ini"
        outfile.close()
        del outfile
        if len(outputFile) < len(extension[:]):
            outputFile += extension[:]
        elif outputFile[-4:] != extension[:]:
            outputFile += extension[:]
        if os.path.exists(outputFile):
            try:
                os.remove(outputFile)
            except IOError:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
                msg.exec_()
                return
        self.roiDir = os.path.dirname(outputFile)
        self.save(outputFile)

    def save(self, filename):
        d= ConfigDict.ConfigDict()
        d['ROI'] = {}
        d['ROI'] = {'roilist': self.mcaROITable.roilist * 1,
                    'roidict':{}}
        d['ROI']['roidict'].update(self.mcaROITable.roidict)
        d.write(filename)

    def setData(self,*var,**kw):
        self.info ={}
        if 'legend' in kw:
            self.info['legend'] = kw['legend']
            del kw['legend']
        else:
            self.info['legend'] = 'Unknown Type'
        if 'xlabel' in kw:
            self.info['xlabel'] = kw['xlabel']
            del kw['xlabel']
        else:
            self.info['xlabel'] = 'X'
        if 'rois' in kw:
            rois = kw['rois']
            self.mcaROITable.fillfromrois(rois)
        self.setHeader(text="%s ROIs of %s" % (self.info['xlabel'],
                                               self.info['legend']))

    def setHeader(self,*var,**kw):
        if len(var):
            text = var[0]
        elif 'text' in kw:
            text = kw['text']
        elif 'header' in kw:
            text = kw['header']
        else:
            text = ""
        self.headerLabel.setText("<b>%s<\b>" % text)

    def emitSignal(self, ddict):
        self.sigMcaROIWidgetSignal.emit(ddict)

class McaROITable(qt.QTableWidget):
    sigMcaROITableSignal = qt.pyqtSignal(object)

    def __init__(self, *args,**kw):
        super(McaROITable, self).__init__(*args)
        self.setRowCount(1)
        self.labels=['ROI','Type','From','To','Raw Counts','Net Counts']
        self.setColumnCount(len(self.labels))
        i=0
        if QTVERSION > '4.2.0':
            self.setSortingEnabled(False)
        if 'labels' in kw:
            for label in kw['labels']:
                item = self.horizontalHeaderItem(i)
                if item is None:
                    item = qt.QTableWidgetItem(label,
                                           qt.QTableWidgetItem.Type)
                item.setText(label)
                self.setHorizontalHeaderItem(i,item)
                i = i + 1
        else:
            for label in self.labels:
                item = self.horizontalHeaderItem(i)
                if item is None:
                    item = qt.QTableWidgetItem(label,
                                           qt.QTableWidgetItem.Type)
                item.setText(label)
                self.setHorizontalHeaderItem(i,item)
                i=i+1

        self.roidict={}
        self.roilist=[]
        if 'roilist' in kw:
            self.roilist = kw['roilist']
        if 'roidict' in kw:
            self.roidict.update(kw['roilist'])
        self.building = False
        self.build()
        #self.connect(self,qt.SIGNAL("cellClicked(int, int)"),self._mySlot)
        #self.connect(self,qt.SIGNAL("cellChanged(int, int)"),self._cellChangedSlot)
        self.cellClicked[(int, int)].connect(self._mySlot)
        self.cellChanged[(int, int)].connect(self._cellChangedSlot)
        verticalHeader = self.verticalHeader()
        verticalHeader.sectionClicked[int].connect(self._rowChangedSlot)

    def build(self):
        self.fillFromROIDict(roilist=self.roilist,roidict=self.roidict)

    def fillFromROIDict(self,roilist=[],roidict={},currentroi=None):
        self.building = True
        line0  = 0
        self.roilist = []
        self.roidict = {}
        for key in roilist:
            if key in roidict.keys():
                roi = roidict[key]
                self.roilist.append(key)
                self.roidict[key] = {}
                self.roidict[key].update(roi)
                line0 = line0 + 1
                nlines=self.rowCount()
                if (line0 > nlines):
                    self.setRowCount(line0)
                line = line0 -1
                self.roidict[key]['line'] = line
                ROI = key
                roitype = QString("%s" % roi['type'])
                fromdata= QString("%6g" % (roi['from']))
                todata  = QString("%6g" % (roi['to']))
                if 'rawcounts' in roi:
                    rawcounts= QString("%6g" % (roi['rawcounts']))
                else:
                    rawcounts = " ?????? "
                if 'netcounts' in roi:
                    netcounts= QString("%6g" % (roi['netcounts']))
                else:
                    netcounts = " ?????? "
                fields  = [ROI,roitype,fromdata,todata,rawcounts,netcounts]
                col = 0
                for field in fields:
                    key2 = self.item(line, col)
                    if key2 is None:
                        key2 = qt.QTableWidgetItem(field,
                                                   qt.QTableWidgetItem.Type)
                        self.setItem(line,col,key2)
                    else:
                        key2.setText(field)
                    if (ROI.upper() == 'ICR') or (ROI.upper() == 'DEFAULT'):
                            key2.setFlags(qt.Qt.ItemIsSelectable|
                                          qt.Qt.ItemIsEnabled)
                    else:
                        if col in [0, 2, 3]:
                            key2.setFlags(qt.Qt.ItemIsSelectable|
                                          qt.Qt.ItemIsEnabled|
                                          qt.Qt.ItemIsEditable)
                        else:
                            key2.setFlags(qt.Qt.ItemIsSelectable|
                                          qt.Qt.ItemIsEnabled)
                    col=col+1
        self.setRowCount(line0)
        i = 0
        for label in self.labels:
            self.resizeColumnToContents(i)
            i=i+1
        self.sortByColumn(2, qt.Qt.AscendingOrder)
        for i in range(len(self.roilist)):
            key = str(self.item(i, 0).text())
            self.roilist[i] = key
            self.roidict[key]['line'] = i
        if len(self.roilist) == 1:
            self.selectRow(0)
        else:
            if currentroi in self.roidict.keys():
                self.selectRow(self.roidict[currentroi]['line'])
                if DEBUG:
                    print("Qt4 ensureCellVisible to be implemented")
        self.building = False

    def addROI(self, roi, key=None):
        nlines = self.numRows()
        self.setNumRows(nlines+1)
        line = nlines
        if key is None:
            key = "%d " % line
        self.roidict[key] = {}
        self.roidict[key]['line'] = line
        self.roidict[key]['type'] = roi['type']
        self.roidict[key]['from'] = roi['from']
        self.roidict[key]['to']   = roi['to']
        ROI = key
        roitype = QString("%s" % roi['type'])
        fromdata= QString("%6g" % (roi['from']))
        todata  = QString("%6g" % (roi['to']))
        if 'rawcounts' in roi:
            rawcounts= QString("%6g" % (roi['rawcounts']))
        else:
            rawcounts = " ?????? "
        self.roidict[key]['rawcounts']   = rawcounts
        if 'netcounts' in roi:
            netcounts= QString("%6g" % (roi['netcounts']))
        else:
            netcounts = " ?????? "
        self.roidict[key]['netcounts']   = netcounts
        fields  = [ROI,roitype,fromdata,todata,rawcounts,netcounts]
        col = 0
        for field in fields:
            if (ROI == 'ICR') or (ROI.upper() == 'DEFAULT'):
                key=qttable.QTableItem(self,qttable.QTableItem.Never,field)
            else:
                if col == 0:
                    key=qttable.QTableItem(self,qttable.QTableItem.OnTyping,field)
                else:
                    key=qttable.QTableItem(self,qttable.QTableItem.Never,field)
            self.setItem(line,col,key)
            col=col+1
        self.sortByColumn(2, qt.Qt.AscendingOrder)
        for i in range(len(self.roilist)):
            nkey = str(self.text(i,0))
            self.roilist[i] = nkey
            self.roidict[nkey]['line'] = i
        self.selectRow(self.roidict[key]['line'])
        self.ensureCellVisible(self.roidict[key]['line'],0)

    def getROIListAndDict(self):
        return self.roilist,self.roidict

    def _mySlot(self, *var, **kw):
        #selection changed event
        #get the current selection
        row = self.currentRow()
        col = self.currentColumn()
        if row >= 0:
            ddict = {}
            ddict['event'] = "selectionChanged"
            ddict['row'  ] = row
            ddict['col'  ] = col
            if row >= len(self.roilist):
                if DEBUG:
                    print("deleting???")
                return
                row = 0
            item = self.item(row, 0)
            if item is None:
                text=""
            else:
                text = str(item.text())
            self.roilist[row] = text
            ddict['roi'  ] = self.roidict[self.roilist[row]]
            ddict['key']   = self.roilist[row]
            ddict['colheader'] = self.labels[col]
            ddict['rowheader'] = "%d" % row
            self.emitSignal(ddict)

    def _rowChangedSlot(self, row):
        self._emitSelectionChangedSignal(row, 0)

    def _cellChangedSlot(self, row, col):
        if DEBUG:
            print("_cellChangedSlot(%d, %d)" % (row, col))
        if self.building:
            return
        if col == 0:
            self.nameSlot(row, col)
        else:
            self._valueChanged(row, col)

    def _valueChanged(self, row, col):
        if col not in [2, 3]:
            return
        item = self.item(row, col)
        if item is None:
            return
        text = str(item.text())
        try:
            value = float(text)
        except:
            return
        if row >= len(self.roilist):
            if DEBUG:
                print("deleting???")
            return
        if QTVERSION < '4.0.0':
            text = str(self.text(row, 0))
        else:
            item = self.item(row, 0)
            if item is None:
                text=""
            else:
                text = str(item.text())
        if not len(text):
            return
        if col == 2:
            self.roidict[text]['from'] = value
        elif col ==3:
            self.roidict[text]['to'] = value
        self._emitSelectionChangedSignal(row, col)

    def nameSlot(self, row, col):
        if col != 0: return
        if row >= len(self.roilist):
            if DEBUG:
                print("deleting???")
            return
        item = self.item(row, col)
        if item is None:
            text=""
        else:
            text = str(item.text())
        if len(text) and (text not in self.roilist):
            old = self.roilist[row]
            self.roilist[row] = text
            self.roidict[text] = {}
            self.roidict[text].update(self.roidict[old])
            del self.roidict[old]
            self._emitSelectionChangedSignal(row, col)

    def _emitSelectionChangedSignal(self, row, col):
        ddict = {}
        ddict['event'] = "selectionChanged"
        ddict['row'  ] = row
        ddict['col'  ] = col
        ddict['roi'  ] = self.roidict[self.roilist[row]]
        ddict['key']   = self.roilist[row]
        ddict['colheader'] = self.labels[col]
        ddict['rowheader'] = "%d" % row
        self.emitSignal(ddict)

    def mySlot(self,*var,**kw):
        if len(var) == 0:
            self._mySlot()
            return
        if len(var) == 2:
            ddict={}
            row = var[0]
            col = var[1]
            if col == 0:
                if row >= len(self.roilist):
                    if DEBUG:
                        print("deleting???")
                    return
                    row = 0
                item = self.item(row, col)
                if item is None:
                    text=""
                else:
                    text = str(item.text())
                if len(text) and (text not in self.roilist):
                    old = self.roilist[row]
                    self.roilist[row] = text
                    self.roidict[text] = {}
                    self.roidict[text].update(self.roidict[old])
                    del self.roidict[old]
                    ddict = {}
                    ddict['event'] = "selectionChanged"
                    ddict['row'  ] = row
                    ddict['col'  ] = col
                    ddict['roi'  ] = self.roidict[self.roilist[row]]
                    ddict['key']   = self.roilist[row]
                    ddict['colheader'] = self.labels[col]
                    ddict['rowheader'] = "%d" % row
                    self.emitSignal(ddict)
                else:
                    if item is None:
                        item = qt.QTableWidgetItem(text,
                                   qt.QTableWidgetItem.Type)
                    else:
                        item.setText(text)
                    self._mySlot()

    def emitSignal(self, ddict):
        self.sigMcaROITableSignal.emit(ddict)

class SimpleComboBox(qt.QComboBox):
        def __init__(self,parent = None,name = None,fl = 0,options=['1','2','3']):
            qt.QComboBox.__init__(self,parent)
            self.setOptions(options)

        def setOptions(self,options=['1','2','3']):
            self.clear()
            self.insertStrList(options)

        def getCurrent(self):
            return   self.currentItem(),str(self.currentText())

if __name__ == '__main__':
    app = qt.QApplication([])
    demo = McaROIWidget()
    demo.show()
    app.exec_()

