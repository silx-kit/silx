# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/03/2016"

cimport cython

cdef extern from "SpecFileCython.h":
    struct _SpecFile:
        pass
# Renaming struct because we have too many SpecFile items (files, classes…)
ctypedef _SpecFile SpecFileHandle

cdef extern from "SpecFileCython.h":
    # sfinit
    SpecFileHandle* SfOpen(char*, int*)
    int SfClose(SpecFileHandle*)
    char* SfError(int)
    
    # sfindex
    long* SfList(SpecFileHandle*, int*)
    long SfScanNo(SpecFileHandle*)
    long SfIndex(SpecFileHandle*, long, long)
    long SfNumber(SpecFileHandle*, long)
    long SfOrder(SpecFileHandle*, long)
    
    # sfdata
    int SfData(SpecFileHandle*, long, double***, long**, int*)
    long SfDataLine(SpecFileHandle*, long, long, double**, int*)
    long SfDataColByName(SpecFileHandle*, long, char*, double**, int*)
    
    # sfheader
    #char* SfTitle(SpecFileHandle*, long, int*)
    long SfHeader(SpecFileHandle*, long, char*, char***, int*)
    long SfFileHeader(SpecFileHandle*, long, char*, char***, int*)
    char* SfCommand(SpecFileHandle*, long, int*)
    long SfNoColumns(SpecFileHandle*, long, int*)
    char* SfDate(SpecFileHandle*, long, int*)
    
    # sflabel
    long SfAllLabels(SpecFileHandle*, long, char***, int*)
    char* SfLabel(SpecFileHandle*, long, long, int *)
    long SfAllMotors(SpecFileHandle*, long, char***, int*)
    long SfAllMotorPos(SpecFileHandle*, long, double**, int*)
    double SfMotorPosByName(SpecFileHandle*, long, char*, int*)
    
    # sftools
    void freeArrNZ(void***, long)

    # sfmca
    long SfNoMca(SpecFileHandle*, long, int*)
    int  SfGetMca(SpecFileHandle*, long, long , double**, int*)
    long SfMcaCalib(SpecFileHandle*, long, double**, int*)

