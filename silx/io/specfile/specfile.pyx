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
"""
This module is a cython binding to wrap the C library SpecFile.

Examples
========

Start by importing :class:`SpecFile` and instantiate it:

.. code-block:: python

    from silx.io.specfile import SpecFile

    sf = SpecFile("test.dat")

A :class:`SpecFile` instance can be accessed like a dictionary to obtain a
:class:`Scan` instance.

If the key is a string representing two values
separated by a dot (e.g. ``"1.2"``), they will be treated as the scan number
(``#S`` header line) and the scan order:

.. code-block:: python

    # get second occurrence of scan "#S 1"
    myscan = sf["1.2"]

    # access scan data as a numpy array
    nlines, ncolumns = myscan.data.shape

If the key is an integer, it will be treated as a 0-based index:

.. code-block:: python

    first_scan = sf[0]
    second_scan = sf[1]

It is also possible to browse through all scans using :class:`SpecFile` as
an iterator:

.. code-block:: python

    for scan in sf:
        print(scan.scan_header['S'])

MCA data can be selectively loaded using an instance of :class:`MCA` provided
by :class:`Scan`:

.. code-block:: python

    # Only one MCA line is loaded in memory
    second_mca = first_scan.mca[1]

    # Iterating trough all MCA in a scan:
    for mca_data in first_scan.mca:
        print(sum(mca_data))

Classes
=======

- :class:`SpecFile`
- :class:`Scan`
- :class:`MCA`
"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "19/02/2016"

import os.path
import logging
import numpy
import re
import sys

cimport numpy
cimport cython
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

numpy.import_array()

from specfile_pxd cimport *

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

class SfNoMcaError(Exception):
    """Custom exception raised when ``SfNoMca()`` returns ``-1`` for an
    unknown reason.
    """
    # TODO: understand reason why SfNoMca() would return -1
    #       (if (sfSetCurrent(sf,index,error) == -1 ))
    pass

class MCA():
    """
    MCA data in a Scan.

    This class is defined to allow lazy loading of a single MCA from within
    a :class:`Scan` instance.

    :param scan: Parent Scan instance
    :type scan: :class:`Scan`
    """
    def __init__(self, scan):
        self._scan = scan

    def __len__(self):
        """
        :return: Number of mca in Scan
        :rtype: int
        """
        return self._scan._specfile.number_of_mca(self._scan.index)

    def __getitem__(self, key):
        """
        :param key: Index of MCA within Scan
        :type key: int

        :return: Single MCA
        :rtype: 1D numpy array
        """
        if isinstance(key, int):
            mca_index = key
            # allow negative index, like lists
            if mca_index < 0:
                mca_index = len(self) + mca_index
        else:
            raise TypeError("MCA index should be an integer (%s provided)" %
                            (type(key)))

        if not 0 <= mca_index < len(self):
            msg = "MCA index must be in range 0-%d" % (len(self) - 1)
            raise IndexError(msg)

        return self._scan._specfile.get_mca(self._scan.index,
                                            mca_index)

    def __iter__(self):
        for mca_index in range(len(self)):
            yield self._scan._specfile.get_mca(self._scan.index, mca_index)


class Scan():
    """
    SpecFile scan

    :param specfile: Parent SpecFile from which this scan is extracted.
    :type specfile: :class:`SpecFile`
    :param scan_index: Unique index defining the scan in the SpecFile
    :type scan_index: int

    List of public class attributes:

    :ivar index: Unique scan index 0 - len(specfile)-1
    :vartype index: int
    :ivar scan_header_lines: List of raw header lines, including the leading
        ``#``
    :vartype scan_header_lines: list of strings
    :ivar scan_header: Dictionary of header strings, keys without leading
        ``#``.  Note: this does not include MCA header lines starting with
        ``#@``.
    :vartype scan_header: dict
    :ivar file_header_lines: List of raw file header lines relevant to this
        scan, including the leading ``#``
    :vartype file_header_lines: list of strings
    :ivar file_header: Dictionary of file header strings, keys without
        the leading ``#``
    :vartype file_header: dict
    :ivar mca_header: Dictionary of mca header strings, keys without
        the leading ``#@``
    :vartype mca_header: dict
    """
    def __init__(self, specfile, scan_index):
        self._specfile = specfile
        #: unique index 0 - len(specfile)-1
        self.index = scan_index
        # number: first value on #S line
        self.number = specfile.number(scan_index)
        # order can be > 1 if a same number is used mor than once in specfile
        self.order = specfile.order(scan_index)

        self.scan_header_lines = self._specfile.scan_header(self.index)
        
        if self.record_exists_in_hdr('L'):
            self.labels = self._specfile.labels(self.index)
        
        self.scan_header = {}
        self.mca_header = {}
        for line in self.scan_header_lines:
            match = re.search(r"#(\w+) *(.*)", line)
            match_mca = re.search(r"#@(\w+) *(.*)", line)
            if match:
                # header type 
                hkey = match.group(1).lstrip("#").strip()
                hvalue = match.group(2).strip()
                self.scan_header[hkey] = hvalue
            elif match_mca:
                hkey = match_mca.group(1).lstrip("#").strip()
                hvalue = match_mca.group(2).strip()
                self.mca_header[hkey] = hvalue
            else:
                # this shouldn't happen
                logging.warn("Unable to parse scan header line " + line)
        
        # Alternative: call dedicated function for each header
        # (results will be different from previous solution, value types may
        # be integers or arrays instead of just strings)
#         if self.record_exists_in_hdr('N'):
#             self.scan_header['N'] = self._specfile.columns(self.index)
#         if self.record_exists_in_hdr('S'):
#             self.scan_header['S'] = self._specfile.command(self.index)
#         if self.record_exists_in_hdr('L'):
#             self.scan_header['L'] = self._specfile.labels(self.index)
        # etc...
            
        self.file_header_lines = self._specfile.file_header(self.index)
        
        self.file_header = {}
        for line in self.file_header_lines:
            match = re.search(r"#(\w+) *(.*)", line)
            if match:
                # header type 
                hkey = match.group(1).lstrip("#").strip()
                hvalue = match.group(2).strip()
                self.file_header[hkey] = hvalue
            else:
                logging.warn("Unable to parse file header line " + line)
                
        self.motor_names = self._specfile.all_motor_names(self.index)
        self.motor_positions = self._specfile.all_motor_positions(self.index)

        self._data_loaded = False
        self._mca_instantiated = False

    @property
    def data(self):
        """Scan data as a numpy.ndarray with the usual attributes
        (e.g. data.shape).
        """
        if not self._data_loaded:
            self._data = self._specfile.data(self.index)
#            self.nlines, self.ncolumns = self._data.shape
            self._data_loaded = True
        return self._data

    @data.setter
    def data(self, data):
        raise NotImplementedError("Scan.data is a read-only attribute")

    @property
    def mca(self):
        """MCA data in this scan.

        Each multichannel analysis is a 1D numpy array. Metadata about
        MCA data - ``#@...`` scan header lines - is to be found in
        :py:attr:`mca_header`.

        :rtype: :class:`MCA`
        """
        if not self._mca_instantiated:
            self._mca = MCA(self)
            self._mca_instantiated = True
        return self._mca

    @mca.setter
    def mca(self, mca):
        raise NotImplementedError("Scan.mca is a read-only attribute")
            
    def record_exists_in_hdr(self, record):
        """Check whether a scan header line  exists.
        
        This should be used before attempting to retrieve header information 
        using a C function that may crash with a *segmentation fault* if the
        header isn't defined in the SpecFile.
        
        :param record: single upper case letter corresponding to the
                       header you want to test (e.g. ``L`` for labels)
        :type record: str

        :return: True or False
        :rtype: boolean
        
        """
        for line in self.scan_header_lines:
            if line.startswith("#" + record):
                return True
        return False
    
    def record_exists_in_file_hdr(self, record):
        """Check whether a file header line  exists.
        
        This should be used before attempting to retrieve header information 
        using a C function that may crash with a *segmentation fault* if the
        header isn't defined in the SpecFile.
        
        :param record_type: single upper case letter corresponding to the
                            header you want to test
        :type record_type: str

        :return: True or False
        :rtype: boolean
        
        """
        for line in self.file_header_lines:
            if line.startswith("#" + record):
                return True
        return False
    
    def data_line(self, line_index):
        """Returns data for a given line of this scan.
        
        :param line_index: Index of data line to retrieve (starting with 0)
        :type line_index: int

        :return: Line data as a 1D array of doubles
        :rtype: numpy.ndarray 
        """
        #return self._specfile.data_line(self.index, line_index)
        return self.data[line_index, :]

    def data_column_by_name(self, label):
        """Returns data for a given column.

        :param label: Label of data column to retrieve, as defined on the
            ``#N`` line of the scan header.
        :type label: str

        :return: Line data as a 1D array of doubles
        :rtype: numpy.ndarray
        """
        return self._specfile.data_column_by_name(self.index, label)

    def motor_position_by_name(self, name):
        """Returns position for a given motor.

        :param name: Name of motor, as defined on the ``#O`` line of the
           file header.
        :type name: str

        :return: Motor position
        :rtype: float
        """
        return self._specfile.motor_position_by_name(self.index, name)

def string_to_char_star(string_):
    """Convert a string to ASCII encoded bytes if using python3"""
    if sys.version.startswith("3"):
        return bytes(string_, "ascii")
    return string_


cdef class SpecFile(object):
    """
    Class wrapping the C SpecFile library.

    :param filename: Path of the SpecFile to read
    :type filename: string
    """
    
    cdef:
        SpecFileHandle *handle
        str filename
        int __open_failed
    
   
    def __cinit__(self, filename):
        cdef int error = SF_ERR_NO_ERRORS
        self.__open_failed = 0

        if os.path.isfile(filename):
            filename = string_to_char_star(filename)
            self.handle =  SfOpen(filename, &error)
        else:
            self.__open_failed = 1
            self._handle_error(SF_ERR_FILE_OPEN)
        if error:
            self.__open_failed = 1
            self._handle_error(error)
       
    def __init__(self, filename):
        self.filename = filename
        
    def __dealloc__(self):
        """Destructor: Calls SfClose(self.handle)"""
        #SfClose makes a segmentation fault if file failed to open
        if not self.__open_failed:            
            if SfClose(self.handle):
                logging.warn("Error while closing SpecFile")
                                        
    def __len__(self):
        """Returns the number of scans in the SpecFile"""
        return SfScanNo(self.handle)

    def __iter__(self):
        for scan_index in range(len(self)):
            yield Scan(self, scan_index)

    def __getitem__(self, key):
        """Return a Scan object

        The Scan instance returned here keeps a reference to its parent SpecFile
        instance in order to use its method to retrieve data and headers.
        """
        msg = "The scan identification key can be an integer representing "
        msg += "the unique scan index or a string 'N.M' with N being the scan"
        msg += "number and M the order (eg '2.3')"

        if isinstance(key, int):
            scan_index = key
            # allow negative index, like lists
            if scan_index < 0:
                scan_index = len(self) + scan_index
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
        
    def _get_error_string(self, error_code):
        """Returns the error message corresponding to the error code.
        
        :param code: Error code
        :type code: int
        """
        return (<bytes> SfError(error_code)).decode()
    
    def _handle_error(self, error_code):
        """Inspect error code, raise adequate error type if necessary.
        
        :param code: Error code
        :type code: int
        """
        # TODO:
        #  sort error types            
        error_message = self._get_error_string(error_code)
        if error_code in (SF_ERR_LINE_NOT_FOUND,
                          SF_ERR_SCAN_NOT_FOUND,
                          SF_ERR_HEADER_NOT_FOUND,
                          SF_ERR_LABEL_NOT_FOUND,
                          SF_ERR_MOTOR_NOT_FOUND,
                          SF_ERR_USER_NOT_FOUND,
                          SF_ERR_MCA_NOT_FOUND):
            raise IndexError(error_message)
        elif error_code in (SF_ERR_POSITION_NOT_FOUND,  #SfMotorPosByName
                            SF_ERR_COL_NOT_FOUND):      #SfDataColByName
            raise KeyError(error_message)
        elif error_code in (SF_ERR_FILE_OPEN,
                            SF_ERR_FILE_CLOSE,
                            SF_ERR_FILE_READ,
                            SF_ERR_FILE_WRITE):
            raise IOError(error_message)  
        elif error_code in (SF_ERR_LINE_EMPTY,):
            raise ValueError(error_message)   
        elif error_code in (SF_ERR_MEMORY_ALLOC,):
            raise MemoryError(error_message) 
        
    
    def index(self, scan_number, scan_order=1):
        """Returns scan index from scan number and order.
        
        :param scan_number: Scan number (possibly non-unique). 
        :type scan_number: int
        :param scan_order: Scan order. 
        :type scan_order: int default 1

        :return: Unique scan index
        :rtype: int
        
        
        Scan indices are increasing from ``0`` to ``len(self)-1`` in the
        order in which they appear in the file.
        Scan numbers are defined by users and are not necessarily unique.
        The scan order for a given scan number increments each time the scan 
        number appers in a given file.
        """
        idx = SfIndex(self.handle, scan_number, scan_order)
        if idx == -1:
            self._handle_error(SF_ERR_SCAN_NOT_FOUND)
        return idx - 1
    
    def number(self, scan_index):
        """Returns scan number from scan index.
        
        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: User defined scan number.
        :rtype: int
        """
        idx = SfNumber(self.handle, scan_index + 1)
        if idx == -1:
            self._handle_error(SF_ERR_SCAN_NOT_FOUND)
        return idx
    
    def order(self, scan_index):
        """Returns scan order from scan index.
        
        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: Scan order (sequential number incrementing each time a
                 non-unique occurrence of a scan number is encountered).
        :rtype: int
        """
        ordr = SfOrder(self.handle, scan_index + 1)
        if ordr == -1:
            self._handle_error(SF_ERR_SCAN_NOT_FOUND)
        return ordr

    def list(self):  
        """Returns list (1D numpy array) of scan numbers in SpecFile.
         
        :return: list of scan numbers (from `` #S``  lines) in the same order
            as in the original SpecFile.
        :rtype: numpy array 
        """
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
    
    def data(self, scan_index):
        """Returns data for the specified scan index.

        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: Complete scan data as a 2D array of doubles
        :rtype: numpy.ndarray
        """
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
        return ret_array

    def data_column_by_name(self, scan_index, label):
        """Returns data column for the specified scan index and column label.

        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int
        :param label: Label of data column, as defined in the ``#L`` line
            of the scan header.
        :type label: str

        :return: Data column as a 1D array of doubles
        :rtype: numpy.ndarray
        """
        cdef:
            double* data_column
            long i, nlines
            int error = SF_ERR_NO_ERRORS

        label = string_to_char_star(label)

        nlines = SfDataColByName(self.handle,
                                 scan_index + 1,
                                 label,
                                 &data_column,
                                 &error)
        self._handle_error(error)

        cdef numpy.ndarray ret_array = numpy.empty((nlines,),
                                                   dtype=numpy.double)
        for i in range(nlines):
            ret_array[i] = data_column[i]

        free(data_column)
        return ret_array

    def scan_header(self, scan_index):
        """Return list of scan header lines.
        
        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: List of raw scan header lines
        :rtype: list of str
        """
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
            line =  <bytes>lines[i].decode()
            lines_list.append(line)
                
        freeArrNZ(<void***>&lines, nlines)
        return lines_list
    
    def file_header(self, scan_index):
        """Return list of file header lines.
        
        A file header contains all lines between a ``#F`` header line and
        a ``#S`` header line (start of scan). We need to specify a scan
        number because there can be more than one file header in a given file.
        A file header applies to all subsequent scans, until a new file
        header is defined.
        
        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: List of raw file header lines
        :rtype: list of str
        """
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
            line =  <bytes>lines[i].decode()
            lines_list.append(line)
                
        freeArrNZ(<void***>&lines, nlines)
        return lines_list     
    
    def columns(self, scan_index): 
        """Return number of columns in a scan from the ``#N`` header line
        (without ``#N`` and scan number)
        
        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: Number of columns in scan from ``#N`` line
        :rtype: int
        """
        cdef: 
            int error = SF_ERR_NO_ERRORS
            
        ncolumns = SfNoColumns(self.handle, 
                               scan_index + 1, 
                               &error)
        self._handle_error(error)
        
        return ncolumns
        
    def command(self, scan_index): 
        """Return ``#S`` line (without ``#S`` and scan number)
        
        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: S line
        :rtype: str
        """
        cdef: 
            int error = SF_ERR_NO_ERRORS
            
        s_record = <bytes> SfCommand(self.handle, 
                                     scan_index + 1, 
                                     &error)
        self._handle_error(error)

        return s_record.decode()
    
    def date(self, scan_index):  
        """Return date from ``#D`` line

        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: Date from ``#D`` line
        :rtype: str
        """
        cdef: 
            int error = SF_ERR_NO_ERRORS
            
        d_line = <bytes> SfDate(self.handle,
                                scan_index + 1,
                                &error)
        self._handle_error(error)
        
        return d_line.decode()
    
    def labels(self, scan_index):
        """Return all labels from ``#L`` line
          
        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: All labels from ``#L`` line
        :rtype: list of strings
        """
        cdef: 
            char** all_labels
            int error = SF_ERR_NO_ERRORS

        nlabels = SfAllLabels(self.handle, 
                              scan_index + 1, 
                              &all_labels,
                              &error)
        self._handle_error(error)

        labels_list = []
        for i in range(nlabels):
            labels_list.append(<bytes>all_labels[i].decode())
            
        freeArrNZ(<void***>&all_labels, nlabels)
        return labels_list
     
    def all_motor_names(self, scan_index):
        """Return all motor names from ``#O`` lines
          
        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: All motor names
        :rtype: list of strings
        """
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
            motors_list.append(<bytes>all_motors[i].decode())
        
        freeArrNZ(<void***>&all_motors, nmotors)
        return motors_list
         
    def all_motor_positions(self, scan_index):
        """Return all motor positions
          
        :param scan_index: Unique scan index between ``0``
            and ``len(self)-1``.
        :type scan_index: int

        :return: All motor positions
        :rtype: list of double
        """
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

    def motor_position_by_name(self, scan_index, name):
        """Return motor position

        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: Specified motor position
        :rtype: double
        """
        cdef:
            int error = SF_ERR_NO_ERRORS

        name = string_to_char_star(name)

        motor_position = SfMotorPosByName(self.handle,
                                          scan_index + 1,
                                          name,
                                          &error)
        self._handle_error(error)

        return motor_position

    def number_of_mca(self, scan_index):
        """Return number of mca spectra in a scan.

        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: Number of mca spectra.
        :rtype: int
        """
        cdef:
            int error = SF_ERR_NO_ERRORS

        num_mca = SfNoMca(self.handle,
                        scan_index + 1,
                        &error)
        # error code updating isn't implemented in SfMcaCalib
        if num_mca == -1:
            raise SfNoMcaError("Failed to retrieve number of MCA " +
                             "(SfNoMca returned -1)")


        return num_mca

    def mca_calibration(self, scan_index):
        """Return MCA calibration in the form :math:`a + b x + c x²`

        Raise a KeyError if there is no ``@CALIB`` line in the scan header.

        :param scan_index: Unique scan index between ``0`` and
            ``len(self)-1``.
        :type scan_index: int

        :return: MCA calibration as a list of 3 values math:`(a, b, c)`
        :rtype: list of floats
        """
        cdef:
            int error = SF_ERR_NO_ERRORS
            double* mca_calib

        mca_calib_error = SfMcaCalib(self.handle,
                                     scan_index + 1,
                                     &mca_calib,
                                     &error)

        # error code updating isn't implemented in SfMcaCalib
        if mca_calib_error:
            raise KeyError("MCA calibration line (@CALIB) not found")

        mca_calib_list = []
        for i in range(3):
            mca_calib_list.append(mca_calib[i])

        free(mca_calib)
        return mca_calib_list


    def get_mca(self, scan_index, mca_index):
        """Return MCA data

        :param scan_index: Unique scan index between ``0`` and ``len(self)-1``.
        :type scan_index: int
        :param mca_index: Index of MCA in the scan
        :type mca_index: int

        :return: MCA data
        :rtype: numpy array
        """
        cdef:
            int error = SF_ERR_NO_ERRORS
            double* mca_data
            long  len_mca

        len_mca = SfGetMca(self.handle,
                           scan_index + 1,
                           mca_index + 1,
                           &mca_data,
                           &error)
        self._handle_error(error)


        cdef numpy.ndarray ret_array = numpy.empty((len_mca,),
                                                   dtype=numpy.double)
        for i in range(len_mca):
            ret_array[i] = mca_data[i]

        free(mca_data)
        return ret_array




