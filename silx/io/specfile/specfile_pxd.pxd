# coding: utf-8
#pxd
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
    
    # sftools
    void freeArrNZ(void***, long)

