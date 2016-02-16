#/*##########################################################################
# Copyright (C) 2004-2016 European Synchrotron Radiation Facility, Grenoble, France
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
#from PyMca5.PyMcaIO.specfilewrapper import Specfile

# General comment: our scan indices go from 0 to N-1 while the C SpecFile 
# library expects indices between 1 and N. 

__author__ = "P. Knobel - ESRF Data Analysis"
__contact__ = "pierre.knobel@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This module is a cython binding to wrap the C-library SpecFile.

Classes
=======

- :class:`SpecFile`
- :class:`Scan`
"""

# TODO: 
# - MCA


import os.path
import re
import numpy

cimport numpy
cimport cython
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

numpy.import_array()

from specfile_pxd cimport *

debugging = True

def debug_msg(msg):
    if debugging:
        print("Debug message: " + str(msg))
        

SF_ERR_NO_ERRORS = 0
SF_ERR_MEMORY_ALLOC = 1
SF_ERR_FILE_OPEN = 2
SF_ERR_FILE_CLOSE = 3
SF_ERR_FILE_READ = 4
SF_ERR_FILE_WRITE = 5
SF_ERR_LINE_NOT_FOUND = 6
SF_ERR_SCAN_NOT_FOUND = 7
SF_ERR_HEADER_NOT_FOUND = 8
SF_ERR_LABEL_NOT_FOUND = 9
SF_ERR_MOTOR_NOT_FOUND = 10
SF_ERR_POSITION_NOT_FOUND = 11
SF_ERR_LINE_EMPTY = 12
SF_ERR_USER_NOT_FOUND = 13
SF_ERR_COL_NOT_FOUND = 14
SF_ERR_MCA_NOT_FOUND = 15

class Scan(object):
    '''
    SpecFile scan

    :param specfile: Parent SpecFile from which this scan is extracted.
    :type specfile: :class:SpecFile
    :param scan_index: Unique index defining the scan in the SpecFile
    :type scan_index: int
    '''    
    def __init__(self, specfile, scan_index):
        self._specfile = specfile
        # unique index 0 - len(specfile)-1
        self.index = scan_index
        # number: first value on #S line
        self.number = specfile.number(scan_index)
        # order can be > 1 if a same number is used mor than once in specfile
        self.order = specfile.order(scan_index)
        
        self.data = self._specfile.data(self.index)
        self.nlines, self.ncolumns = self.data.shape
        
        self.header_lines = self._specfile.scan_header(self.index)
        '''List of raw header lines, including the leading "#L"'''
        
        if self.record_exists_in_hdr('L'):
            self.labels = self._specfile.labels(self.index)
        
        self.header_dict = {}
        '''Dictionary of header strings, keys without leading "#"'''
        for line in self.header_lines:
            match = re.search(r"#(\w+) *(.*)", line)
            if match:
                # header type 
                hkey = match.group(1).lstrip("#").strip()
                hvalue = match.group(2).strip()
                self.header_dict[hkey] = hvalue
            else:
                # this shouldn't happen
                print("Warning: unable to parse file header line " + line)
        
        # Alternative: call dedicated function for each header
        # (results will be different from previous solution, value types may
        # be integers or arrays instead of just strings)
#         if self.record_exists_in_hdr('N'):
#             self.header_dict['N'] = self._specfile.columns(self.index)
#         if self.record_exists_in_hdr('S'):
#             self.header_dict['S'] = self._specfile.command(self.index)
#         if self.record_exists_in_hdr('L'):
#             self.header_dict['L'] = self._specfile.labels(self.index)
            
        self.file_header_lines = self._specfile.file_header(self.index)
        '''List of raw file header lines relevant to this scan, 
        including the leading "#L"'''
        
        self.file_header_dict = {}
        '''Dictionary of file header strings, keys without leading "#"'''
        for line in self.file_header_lines:
            match = re.search(r"#(\w+) *(.*)", line)
            if match:
                # header type 
                hkey = match.group(1).lstrip("#").strip()
                hvalue = match.group(2).strip()
                self.file_header_dict[hkey] = hvalue
            else:
                # this shouldn't happen
                print("Warning: unable to parse file header line " + line)
                
        self.motor_names = self._specfile.all_motor_names(self.index)
        self.motor_positions = self._specfile.all_motor_positions(self.index)
            
            
    def record_exists_in_hdr(self, record):
        '''Check whether a scan header line  exists.
        
        This should be used before attempting to retrieve header information 
        using a C function that may crash with a "segmentation fault" if the 
        header isn't defined in the SpecFile.
        
        :param record_type: single upper case letter corresponding to the
                            header you want to test (e.g. 'L' for labels)
        :type record_type: str
        :returns: True or False
        :rtype: boolean
        
        '''
        for line in self.header_lines:
            if line.startswith("#" + record):
                return True
        return False
    
    def record_exists_in_file_hdr(self, record):
        '''Check whether a file header line  exists.
        
        This should be used before attempting to retrieve header information 
        using a C function that may crash with a "segmentation fault" if the 
        header isn't defined in the SpecFile.
        
        :param record_type: single upper case letter corresponding to the
                            header you want to test (e.g. 'L' for labels)
        :type record_type: str
        :returns: True or False
        :rtype: boolean
        
        '''
        for line in self.file_header_lines:
            if line.startswith("#" + record):
                return True
        return False
    
    def data_line(self, line_index):
        '''Returns data for a given line of this scan.
        
        :param line_index: Index of data line to retrieve (starting with 0)
        :type line_index: int
        :return: Line data as a 1D array of doubles
        :rtype: numpy.ndarray 
        '''   
        #return self._specfile.data_line(self.index, line_index)
        return self.data[line_index, :]


cdef class SpecFile(object):
    '''
    Class wrapping the C SpecFile library.

    :param filename: Path of the SpecFile to read
    :type label_text: string
    '''
    
    cdef:
        SpecFileHandle *handle   #SpecFile struct in SpecFile.h
        str filename
        int __open_failed
        int iter_counter
    
   
    def __cinit__(self, filename):
        cdef int error = SF_ERR_NO_ERRORS
        self.__open_failed = 0
        if os.path.isfile(filename):
            self.handle =  SfOpen(filename, &error)
        else:
            self.__open_failed = 1
            self._handle_error(SF_ERR_FILE_OPEN)
        if error:
            self.__open_failed = 1
            self._handle_error(error)
       
    def __init__(self, filename):
        self.filename = filename
        # iterator counter
        self.iter_counter = 0
        
    def __dealloc__(self):
        '''Destructor: Calls SfClose(self.handle)'''
        #SfClose creates a segmentation fault if file failed to open
        if not self.__open_failed:            
            if SfClose(self.handle):
                print("Error while closing")
                                        
    def __len__(self):
        '''Returns the number of scans in the SpecFile'''
        return SfScanNo(self.handle)

    def __iter__(self):
        return self

    def __next__(self):
        'Returns the next value till current is lower than high'
        if self.iter_counter >= len(self):
            raise StopIteration
        else:
            self.iter_counter += 1
            return Scan(self, self.iter_counter - 1)
        
    def _get_error_string(self, error_code):
        '''Returns the error message corresponding to the error code.
        
        :param code: Error code
        :type code: int
        '''   
        return (<bytes> SfError(error_code)).encode('utf-8)') 
    
    def _handle_error(self, error_code):
        '''Inspect error code, raise adequate error type if necessary.
        
        :param code: Error code
        :type code: int
        '''
        # TODO:
        #  sort error types            
        error_message = self._get_error_string(error_code)
        if error_code in (SF_ERR_LINE_NOT_FOUND,
                           SF_ERR_SCAN_NOT_FOUND,
                           SF_ERR_HEADER_NOT_FOUND,
                           SF_ERR_LABEL_NOT_FOUND,
                           SF_ERR_MOTOR_NOT_FOUND,
                           SF_ERR_POSITION_NOT_FOUND,
                           SF_ERR_USER_NOT_FOUND,
                           SF_ERR_COL_NOT_FOUND,
                           SF_ERR_MCA_NOT_FOUND):
            raise IndexError(error_message)
        elif error_code in (SF_ERR_FILE_OPEN,
                             SF_ERR_FILE_CLOSE,
                             SF_ERR_FILE_READ,
                             SF_ERR_FILE_WRITE):
            raise IOError(error_message)  
        elif error_code in (SF_ERR_LINE_EMPTY,):     # 
            raise ValueError(error_message)   
        elif error_code in (SF_ERR_MEMORY_ALLOC,):
            raise MemoryError(error_message) 
        
    
    def index(self, scan_number, scan_order=1):
        '''Returns scan index from scan number and order.
        
        :param scan_number: Scan number (possibly non-unique). 
        :type scan_number: int
        :param scan_order: Scan order. 
        :type scan_order: int default 1
        :returns: Unique scan index
        :rtype: int
        
        
        Scan indices are increasing from 0 to len(self)-1 in the order in 
        which they appear in the file.
        Scan numbers are defined by users and are not necessarily unique.
        The scan order for a given scan number increments each time the scan 
        number appers in a given file.'''
        idx = SfIndex(self.handle, scan_number, scan_order)
        if idx == -1:
            self._handle_error(SF_ERR_SCAN_NOT_FOUND)
        return idx - 1
    
    def number(self, scan_index):
        '''Returns scan number from scan index.
        
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int
        :returns: User defined scan number.
        :rtype: int
        '''
        idx = SfNumber(self.handle, scan_index + 1)
        if idx == -1:
            self._handle_error(SF_ERR_SCAN_NOT_FOUND)
        return idx
    
    def order(self, scan_index):
        '''Returns scan order from scan index.
        
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int
        :returns: Scan order (sequential number incrementing each time a 
                 non-unique occurrence of a scan number is encountered).
        :rtype: int
        '''
        ordr = SfOrder(self.handle, scan_index + 1)
        if ordr == -1:
            self._handle_error(SF_ERR_SCAN_NOT_FOUND)
        return ordr

    def list(self):  
        '''Returns list (1D numpy array) of scan numbers in SpecFile.
         
        :return: list of scan numbers (#S n ...) in the same order as in the
                 original SpecFile.
        :rtype: numpy array 
        '''    
        cdef:
            long *scan_numbers
            int error = SF_ERR_NO_ERRORS
            
        scan_numbers = SfList(self.handle, &error)
        self._handle_error(error)

        ret_list = []
        for i in range(len(self)):
            ret_list.append(scan_numbers[i])

        free(scan_numbers)
        return ret_list
       
    def __getitem__(self, key):
        '''Return a Scan object
        
        The Scan instance returned here keeps a reference to its parent SpecFile 
        instance in order to use its method to retrieve data and headers.
        
        Example:
        --------
        
        .. code-block:: python
            
            from specfile import SpecFile
            sf = SpecFile("t.dat")
            scan2 = sf[2]
            nlines, ncolumns = myscan.data.shape
            labels_list = scan2.header_dict['L']
        '''
        msg = "The scan identification key can be an integer representing "
        msg += "the unique scan index or a string 'N.M' with N being the scan"
        msg += "number and M the order (eg '2.3')"
        
        if isinstance(key, int):
            scan_index = key 
        elif isinstance(key, str):
            try:
                (number, order) = map(int, key.split("."))
                scan_index = self.index(number, order)
            except (ValueError, IndexError):
                # self.index can raise an index error
                # int() can raise a value error
                raise KeyError(msg)
        else:
            raise TypeError(msg) 
                
        if not 0 <= scan_index < len(self): 
            msg = "Scan index must be in range 0-%d" % (len(self) - 1)
            raise IndexError(msg)
        
        return Scan(self, scan_index)
    
    def data(self, scan_index): 
        '''Returns data for the specified scan index.
        
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int
        :return: Complete scan data as a 2D array of doubles
        :rtype: numpy.ndarray
        '''        
        cdef: 
            double** mydata
            long* data_info
            int i, j
            int error = SF_ERR_NO_ERRORS
            long nlines, ncolumns, regular

        sfdata_error = SfData(self.handle, 
                              scan_index + 1, 
                              &mydata, 
                              &data_info, 
                              &error)
                  
        self._handle_error(error)
        
        nlines = data_info[0] 
        ncolumns = data_info[1]
        regular = data_info[2]
        
        cdef numpy.ndarray ret_array = numpy.empty((nlines, ncolumns), 
                                                   dtype=numpy.double)
        for i in range(nlines):
            for j in range(ncolumns):
                ret_array[i, j] = mydata[i][j]    
        
        freeArrNZ(<void ***>&mydata, nlines)
        free(data_info)
        
        # nlines and ncolumns can be accessed as ret_array.shape
        return ret_array
    
    def scan_header(self, scan_index):
        '''Return list of scan header lines.
        
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int
        :return: List of raw scan header lines, including the leading "#L"
        :rtype: list of str
        '''
        cdef: 
            char** lines
            int error = SF_ERR_NO_ERRORS

        nlines = SfHeader(self.handle, 
                          scan_index + 1, 
                          "",           # no pattern matching 
                          &lines, 
                          &error)
        
        self._handle_error(error)
        
        lines_list = []
        for i in range(nlines):
            line =  str(<bytes>lines[i].encode('utf-8'))
            lines_list.append(line)
                
        freeArrNZ(<void***>&lines, nlines)
        return lines_list
    
    def file_header(self, scan_index):
        '''Return list of file header lines.
        
        A file header contains all lines between a "#F" header line and
        a #S header line (start of scan). We need to specify a scan number
        because there can be more than one file header in a given file. 
        A file header applies to all subsequent scans, until a new file
        header is defined.
        
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int
        :return: List of raw file header lines, including the leading "#L"
        :rtype: list of str
        '''
        cdef: 
            char** lines
            int error = SF_ERR_NO_ERRORS

        nlines = SfFileHeader(self.handle, 
                             scan_index + 1, 
                             "",          # no pattern matching
                             &lines, 
                             &error)
        self._handle_error(error)

        lines_list = []
        for i in range(nlines):
            line =  str(<bytes>lines[i].encode('utf-8'))
            lines_list.append(line)
                
        freeArrNZ(<void***>&lines, nlines)
        return lines_list     
    
    def columns(self, scan_index): 
        '''Return number of columns in a scan from the #N header line
        (without #N and scan number)
        
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int
        :return: Number of columns in scan from #N record
        :rtype: int
        '''
        cdef: 
            int error = SF_ERR_NO_ERRORS
            
        ncolumns = SfNoColumns(self.handle, 
                               scan_index + 1, 
                               &error)
        self._handle_error(error)
        
        return ncolumns
        
    def command(self, scan_index): 
        '''Return #S line (without #S and scan number)
        
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int
        :return: S line
        :rtype: str
        '''
        cdef: 
            int error = SF_ERR_NO_ERRORS
            
        s_record = <bytes> SfCommand(self.handle, 
                                     scan_index + 1, 
                                     &error)
        self._handle_error(error)

        return str(s_record.encode('utf-8)'))
    
    def date(self, scan_index):  
        '''Return date from #D line

        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int       
        :return: Date from #D line
        :rtype: str
        '''
        cdef: 
            int error = SF_ERR_NO_ERRORS
            
        d_record = <bytes> SfDate(self.handle, 
                                  scan_index + 1,
                                  &error)
        self._handle_error(error)
        
        return str(d_record.encode('utf-8'))
    
    def labels(self, scan_index):
        '''Return all labels from #L line
          
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int     
        :return: All labels from #L line
        :rtype: list of strings
        ''' 
        cdef: 
            char** all_labels
            int error = SF_ERR_NO_ERRORS

        nlabels = SfAllLabels(self.handle, 
                              scan_index + 1, 
                              &all_labels,
                              &error)
        self._handle_error(error)
        
        #labels_list = [None] * num_labels
        labels_list = []
        for i in range(nlabels):
            #labels_list[i] = str(<bytes>all_labels[i].encode('utf-8'))
            labels_list.append(str(<bytes>all_labels[i].encode('utf-8')))
            
        freeArrNZ(<void***>&all_labels, nlabels)
        return labels_list
     
    def all_motor_names(self, scan_index):
        '''Return all motor names from #O lines
          
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int     
        :return: All motor names
        :rtype: list of strings
        ''' 
        cdef: 
            char** all_motors
            int error = SF_ERR_NO_ERRORS
         
        nmotors = SfAllMotors(self.handle, 
                              scan_index + 1, 
                              &all_motors,
                              &error)
        self._handle_error(error)
        
        motors_list = []
        for i in range(nmotors):
            motors_list.append(str(<bytes>all_motors[i].encode('utf-8')))
        
        freeArrNZ(<void***>&all_motors, nmotors)
        return motors_list
         
    def all_motor_positions(self, scan_index):
        '''Return all motor positions
          
        :param scan_index: Unique scan index between 0 and len(self)-1. 
        :type scan_index: int     
        :return: All motor names
        :rtype: list of double
        ''' 
        cdef: 
            double* motor_positions
            int error = SF_ERR_NO_ERRORS
         
        nmotors = SfAllMotorPos(self.handle, 
                                scan_index + 1, 
                                &motor_positions,
                                &error)
        self._handle_error(error)
        
        motor_positions_list = []
        for i in range(nmotors):
            motor_positions_list.append(motor_positions[i])
        
        free(motor_positions)
        return motor_positions_list
    

