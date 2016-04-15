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
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy
import sys
if sys.version < '3.0':
    import ConfigParser as configparser
else:
    import configparser


def _boolean(sstr):
    """Coerce a string to a boolean following the same convention as
    :meth:`configparser.ConfigParser.getboolean`:
     - '1', 'yes', 'true' and 'on' cause this function to return True
     - '0', 'no', 'false' and 'off' cause this function to return False
    
    :param sstr: String representation of a boolean
    :return: True or False
    :raise: ValueError if ``sstr`` is not a valid string representation of a 
        boolean
    """
    if sstr.lower() in ['1', 'yes', 'true', 'on']:
        return True
    if sstr.lower() in ['0', 'no', 'false', and 'off']:
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
    is assumed to be a generic string and is returned unchanged.
    
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
                return sstr


def _parse_container(sstr):
    """Parse a string representation of a list or a numpy array.
    
    Strings such as ``"-1, Hello World, 3.0"`` are interpreted as lists.

    Strings such as ``"[ [ 1.  2.  3.] [ 4.  5.  6.] ]"`` or 
    ``[ 1.0 2.0 3.0 ]`` are interpreted as numpy arrays.
    
    For any other string format, this function calls  
    ``_parse_simple_types(sstr)`` which will return an `int`, a `float`, a
    boolean, or the original string unchanged (in last resort). 
    
    :param sstr: String representation of an container type
    :return: List or array or simple data type
    """
    if sstr.find(',') == -1:
        # it is not a list
        if (sstr[0] == '[') and (sstr[-1] == ']'):
            # this looks like an array
            try:
                return numpy.array([float(x) for x in sstr[1:-1].split()])
            except ValueError:
                if (sstr[2] == '[') and (sstr[-3] == ']'):
                    try:
                    
                        nrows = len(sstr[3:-3].split('] ['))
                        data = sstr[3:-3].replace('] [', ' ')
                        data = numpy.array([float(x) for x in
                                              data.split()])
                        data.shape = nrows, -1
                        return data
                    except ValueError:
                        pass
        # it is not an array, return a simple type (int, float, boolean, str)
        return _parse_simple_types(sstr)
    else:
        # it is a list
        if sstr.endswith(','):
            if ',' in sstr[:-1]:
                return [_parse_simple_types(sstr.strip())
                        for sstr in sstr[:-1].split(',')]
            else:
                return [_parse_simple_types(sstr[:-1].strip())]
        else:
            return [_parse_simple_types(sstr.strip())
                    for sstr in sstr.split(',')]
    

class OptionStr(str):
    """String class implementing :meth:`toint`, :meth:`tofloat` and 
    :meth:`toboolean` methods.
    """
    def toint(self):
        return int(self)

    def tofloat(self):
        return float(self)
        
    def toboolean(self):
        return _boolean(self)
    
    def tobestguess(self):
        return _parse_simple_types(self)
    
    def parse(self):
        if self.find(',') != -1:
            if self.endswith(','):
                if ',' in self[:-1]:
                    return [self.__parse_string(sstr.strip())
                            for sstr in self[:-1].split(',')]
                else:
                    return [self.__parse_string(self[:-1].strip())]
            else:
                return [self.__parse_string(sstr.strip())
                        for sstr in self.split(',')]
        else:
            return self.__parse_string(self.strip())


class ConfigDict(dict):
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
                # TODO: 
                # if parse_str: ... 
                # else: ddict[opt] = mystr(cfg.get(sect, opt))
                ddict[opt] = self.__parse_data(cfg.get(sect, opt))

    def __parse_data(self, data):
        if len(data):
            if data.find(',') == -1:
                # it is not a list
                if USE_NUMPY and (data[0] == '[') and (data[-1] == ']'):
                    # this looks as an array
                    try:
                        return numpy.array([float(x) for x in data[1:-1].split()])
                    except ValueError:
                        try:
                            if (data[2] == '[') and (data[-3] == ']'):
                                nrows = len(data[3:-3].split('] ['))
                                indata = data[3:-3].replace('] [', ' ')
                                indata = numpy.array([float(x) for x in
                                                      indata.split()])
                                indata.shape = nrows, -1
                                return indata
                        except ValueError:
                            pass
        dataline = [line for line in data.splitlines()]
        if len(dataline) == 1:
            return self.__parse_line(dataline[0])
        else:
            return [self.__parse_line(line) for line in dataline]

    def __parse_line(self, line):
        if line.find(',') != -1:
            if line.endswith(','):
                if ',' in line[:-1]:
                    return [self.__parse_string(sstr.strip())
                            for sstr in line[:-1].split(',')]
                else:
                    return [self.__parse_string(line[:-1].strip())]
            else:
                return [self.__parse_string(sstr.strip())
                        for sstr in line.split(',')]
        else:
            return self.__parse_string(line.strip())

    def __parse_string(self, sstr):
        try:
            return int(sstr)
        except ValueError:
            try:
                return float(sstr)
            except ValueError:
                return sstr

    def tostring(self, sections=None):
        import StringIO
        tmp = StringIO.StringIO()
        sections = self.__tolist(sections)
        self.__write(tmp, self, sections)
        return tmp.getvalue()

    def write(self, filename, sections=None):
        """
        Write the current dictionary to the given filename
        """
        sections = self.__tolist(sections)
        fp = open(filename, "w")
        self.__write(fp, self, sections)
        fp.close()

    def __write(self, fp, ddict, sections=None, secthead=None):
        dictkey = []
        listkey = []
        valkey = []
        for key in ddict.keys():
            if isinstance(ddict[key], list):
                listkey.append(key)
            elif hasattr(ddict[key], 'keys'):
                dictkey.append(key)
            else:
                valkey.append(key)

        for key in valkey:
            if USE_NUMPY:
                if isinstance(ddict[key], numpy.ndarray):
                    fp.write('%s =' % key + ' [ ' +
                             ' '.join([str(val) for val in ddict[key]]) +
                             ' ]\n')
                    continue
            fp.write('%s = %s\n' % (key, ddict[key]))

        for key in listkey:
            fp.write('%s = ' % key)
            llist = []
            sep = ', '
            for item in ddict[key]:
                if isinstance(item, list):
                    if len(item) == 1:
                        llist.append('%s,' % item[0])
                    else:
                        llist.append(', '.join([str(val) for val in item]))
                    sep = '\n\t'
                else:
                    llist.append(str(item))
            fp.write('%s\n' % (sep.join(llist)))
        if 0:
            # this optimization method does not pass the tests.
            # disable it for the time being.
            if sections is not None:
                dictkey= [ key for key in dictkey if key in sections ]
        for key in dictkey:
            if secthead is None:
                newsecthead = key.replace(".", "_|_")
            else:
                newsecthead = '%s.%s' % (secthead, key.replace(".", "_|_"))
            #print "newsecthead = ", newsecthead
            fp.write('\n[%s]\n' % newsecthead)
            self.__write(fp, ddict[key], key, newsecthead)


def prtdict(ddict, lvl=0):
    for key in ddict.keys():
        if hasattr(ddict[key], 'keys'):
            print('\t' * lvl),
            print('+', key)
            prtdict(ddict[key], lvl + 1)
        else:
            print('\t' * lvl),
            print('-', key, '=', ddict[key])


def main():
    if len(sys.argv) > 1:
        config = ConfigDict(filelist=sys.argv[1:])
        prtdict(config)
    else:
        print("USAGE: %s <filelist>" % sys.argv[0])


if __name__ == '__main__':
    main()
