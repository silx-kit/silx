#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
__author__ = ["E. Papillon", "V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "18/04/2016"

import numpy
import sys
if sys.version < '3.0':
    import ConfigParser as configparser
else:
    import configparser


string_types = (basestring,) if sys.version_info[0] == 2 else (str,)


def _boolean(sstr):
    """Coerce a string to a boolean following the same convention as
    :meth:`configparser.ConfigParser.getboolean`:
     - '1', 'yes', 'true' and 'on' cause this function to return ``True``
     - '0', 'no', 'false' and 'off' cause this function to return ``False``

    :param sstr: String representation of a boolean
    :return: ``True`` or ``False``
    :raise: ``ValueError`` if ``sstr`` is not a valid string representation 
        of a boolean
    """
    if sstr.lower() in ['1', 'yes', 'true', 'on']:
        return True
    if sstr.lower() in ['0', 'no', 'false', 'off']:
        return False
    msg = "Cannot coerce string '%s' to a boolean value. " % sstr
    msg += "Valid boolean strings: '1', 'yes', 'true', 'on',  "
    msg += "'0', 'no', 'false', 'off'"
    raise ValueError(msg)


def _parse_simple_types(sstr):
    """Coerce a string representation of a value to the most appropriate data
    type, by trial and error.

    Typecasting is attempted to following data types (in this order):
    `int`, `float`, `boolean`. If all of these conversions fail, ``sstr``
    is assumed to be a string.

    :param sstr: String representation of an unknown data type
    :return: Value coerced into the most appropriate data type
    """
    try:
        return int(sstr)
    except ValueError:
        try:
            return float(sstr)
        except ValueError:
            try:
                return _boolean(sstr)
            except ValueError:
                # un-escape string
                sstr = sstr.lstrip("\\")
                # un-escape commas
                sstr = sstr.replace("\,", ",").replace("^@", ",")
                return sstr


def _parse_container(sstr):
    """Parse a string representation of a list or a numpy array.

    A string such as ``"-1, Hello World, 3.0"`` is interpreted as the list
    ``[-1, "Hello World", 3.0]``. ``"-1, "no", 3.0\n\t1, 2"`` is interpreted
    a list of 2 lists ``[[-1, False, 3.0], [1, 2]]``

    Strings such as ``"[ [ 1.  2.  3.] [ 4.  5.  6.] ]"`` or 
    ``[ 1.0 2.0 3.0 ]`` are interpreted as numpy arrays. Only 1D and 2D
    arrays are permitted.

    :param sstr: String representation of an container type
    :return: List or array
    :raise: ``ValueError`` if string is not a list or an array
    """
    sstr = sstr.strip()

    if sstr.find(',') == -1:
        # it is not a list
        if (sstr[0] == '[') and (sstr[-1] == ']'):
            # this looks like an array
            try:
                # try parsing as a 1D array
                return numpy.array([float(x) for x in sstr[1:-1].split()])
            except ValueError:
                # try parsing as a 2D array
                if (sstr[2] == '[') and (sstr[-3] == ']'):
                    nrows = len(sstr[3:-3].split('] ['))
                    data = sstr[3:-3].replace('] [', ' ')
                    data = numpy.array([float(x) for x in
                                          data.split()])
                    data.shape = nrows, -1
                    return data
        # not a list and not an array
        raise ValueError
    else:
        # if all commas are escaped, it is a strinq, not a list
        if sstr.count(",") == sstr.count("\,"):
            raise ValueError

        dataline = [line for line in sstr.splitlines()]
        if len(dataline) == 1:
            return _parse_list_line(dataline[0])
        else:
            return [_parse_list_line(line) for line in dataline]


def _parse_list_line(sstr):
    sstr = sstr.strip()

    # preserve escaped commas in strings before splitting list
    # (_parse_simple_types recognizes ^@ as a comma)
    sstr.replace("\,", "^@")
    # it is a list
    if sstr.endswith(','):
        if ',' in sstr[:-1]:
            return [_parse_simple_types(sstr2.strip())
                    for sstr2 in sstr[:-1].split(',')]
        else:
            return [_parse_simple_types(sstr[:-1].strip())]
    else:
        return [_parse_simple_types(sstr2.strip())
                for sstr2 in sstr.split(',')]


class OptionStr(str):
    """String class providing typecasting methods to parse values in a 
    :class:``ConfigDict`` generated configuration file.
    """
    def toint(self):
        """
        :return: integer
        :raise: ``ValueError`` if conversion to ``int`` failed
        """
        return int(self)

    def tofloat(self):
        """
        :return: ``float``
        :raise: ``ValueError`` if conversion to ``float`` failed
        """
        return float(self)

    def toboolean(self):
        """
        '1', 'yes', 'true' and 'on' are interpreted as ``True``

        '0', 'no', 'false' and 'off' are interpreted as ``False``

        :return: Boolean
        :raise: ``ValueError`` if conversion to ``bool`` failed
        """
        return _boolean(self)

    def tostr(self):
        """Return string after replacing escaped commas ``\,`` with regular
        commas ``,`` and removing leading backslash.

        :return: str(self)
        """
        return str(self.replace("\,", ",").lstrip("\\"))

    def tocontainer(self):
        """Return a list or a numpy array.

        Any string containing a comma (``,``) character will be interpreted
        as a list: for instance ``"-1, Hello World, 3.0"``, or ``"2.0,``

        The format for numpy arrays is a blank space delimited list of values
        between square brackets: ``"[ 1.3 2.2 3.1 ]"``, or 
        ``"[ [ 1 2 3 ] [ 1 4 9 ] ]"``"""   
        return _parse_container(self)

    def tobestguess(self):
        """Parse string without prior knowledge of type.

        Conversion to following types is attempted, in this order:
        `list`, `numpy array`, `int`, `float`, `boolean`. 
        If all of these conversions fail, the string is returned unchanged.
        """
        try:
            return _parse_container(self)
        except ValueError:
            return _parse_simple_types(self)


class ConfigDict(dict):
    """Store configuration parameters as a dictionary. 

    Parameters can be grouped into sections, by storing them as
    sub-dictionaries.

    Keys must be strings. Values can be: integers, booleans, lists,
    numpy arrays, floats, strings.

    Methods are provided to write a configuration file in a variant of INI
    format. A :class:`ConfigDict` can load (or be initialized from) a list of files.

    The main differences between files written/read by this class and standard
    ``ConfigParser`` files are:

        - sections can be nested to any depth
        - value types are guessed when the file is read back
        - to prevent strings from being interpreted as lists, commas are
          escaped with a backslash (``\,``)
        - strings are prefixed with a leading backslash (``\,``) to prevent
          conversion to numeric or boolean values
    """
    def __init__(self, defaultdict=None, initdict=None, filelist=None):
        if defaultdict is None:
            defaultdict = {}
        dict.__init__(self, defaultdict)
        self.default = defaultdict
        self.filelist = []

        if initdict is not None:
            self.update(initdict)
        if filelist is not None:
            self.read(filelist)

    def reset(self):
        """ Revert to default values
        """
        self.clear()
        self.update(self.default)

    def clear(self):
        """ Clear dictionnary
        """
        dict.clear(self)
        self.filelist = []

    def _check(self):
        pass

    def __tolist(self, mylist):
        if mylist is None:
            return None
        if not isinstance(mylist, list):
            return [mylist]
        else:
            return mylist

    def getfiles(self):
        return self.filelist

    def getlastfile(self):
        return self.filelist[len(self.filelist) - 1]

    def __convert(self, option):
        return option

    def read(self, filelist, sections=None):
        """
        read the input filename into the internal dictionary
        """
        filelist = self.__tolist(filelist)
        sections = self.__tolist(sections)
        cfg = configparser.ConfigParser()
        cfg.optionxform = self.__convert
        cfg.read(filelist)
        self.__read(cfg, sections)

        for ffile in filelist:
            self.filelist.append([ffile, sections])
        self._check()

    def __read(self, cfg, sections=None):
        cfgsect = cfg.sections()

        if sections is None:
            readsect = cfgsect
        else:
            readsect = [sect for sect in cfgsect if sect in sections]

        for sect in readsect:
            ddict = self
            for subsectw in sect.split('.'):
                subsect = subsectw.replace("_|_", ".")
                if not (subsect in ddict):
                    ddict[subsect] = {}
                ddict = ddict[subsect]
            for opt in cfg.options(sect):
                ddict[opt] = self.__parse_data(cfg.get(sect, opt))

    def __parse_data(self, data):
        return OptionStr(data).tobestguess()

    def tostring(self):
        """Return INI file content generated by :meth:`write` as a string
        """
        import StringIO
        tmp = StringIO.StringIO()
        self.__write(tmp, self)
        return tmp.getvalue()

    def write(self, ffile):
        """Write the current dictionary to the given filename or
        file handle.

        :param ffile: Output file name or file handle. If a file name is
            provided, the method opens it, writes it and closes it again.
        """
        if not hasattr(ffile, "write"):
            fp = open(ffile, "w")
        else:
            fp = ffile

        self.__write(fp, self)

        if not hasattr(ffile, "write"):
            fp.close()

    def _escape_str(self, sstr):
        # Escape strings with a leading \
        sstr = "\\" + sstr
        # Escape commas
        sstr = sstr.replace(",", "\,")
        return sstr

    def __write(self, fp, ddict, secthead=None):
        dictkey = []
        listkey = []
        valkey = []
        strkey = []

        for key in ddict.keys():
            if isinstance(ddict[key], list):
                listkey.append(key)
            elif hasattr(ddict[key], 'keys'):
                dictkey.append(key)
            elif isinstance(ddict[key], string_types):
                strkey.append(key)
            else:
                valkey.append(key)

        for key in valkey:
            if isinstance(ddict[key], numpy.ndarray):
                fp.write('%s =' % key + ' [ ' +
                         ' '.join([str(val) for val in ddict[key]]) +
                         ' ]\n')
            else:
                fp.write('%s = %s\n' % (key, ddict[key]))
 
        for key in strkey:
            fp.write('%s = \%s\n' % (key, self._escape_str(ddict[key])))

        for key in listkey:
            fp.write('%s = ' % key)
            llist = []
            sep = ', '
            for item in ddict[key]:
                if isinstance(item, list):
                    if len(item) == 1:
                        if isinstance(item[0], string_types):
                            self._escape_str(item[0])
                            llist.append('%s,' % self._escape_str(item[0]))
                        else:
                            llist.append('%s,' % item[0])
                    else:
                        item2 = []
                        for val in item:
                            if isinstance(val, string_types):
                                val = self._escape_str(val)
                            item2.append(val)
                        llist.append(', '.join([str(val) for val in item2]))
                    sep = '\n\t'
                elif isinstance(item, string_types):
                    llist.append(self._escape_str(item))
                else:
                    llist.append(str(item))
            fp.write('%s\n' % (sep.join(llist)))
 
        for key in dictkey:
            if secthead is None:
                newsecthead = key.replace(".", "_|_")
            else:
                newsecthead = '%s.%s' % (secthead, key.replace(".", "_|_"))

            fp.write('\n[%s]\n' % newsecthead)
            self.__write(fp, ddict[key], key, newsecthead)