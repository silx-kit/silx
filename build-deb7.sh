#!/bin/sh
#
#    Project: Silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2015-2.16 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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
# THE SOFTWARE

# Script that builds a debian package from this library 

PROJECT=silx
 
if [ -d /usr/lib/ccache ];
then 
   CCPATH=/usr/lib/ccache:$PATH 
else  
   CCPATH=$PATH
fi
export PYBUILD_DISABLE_python2=test
export PYBUILD_DISABLE_python3=test
export DEB_BUILD_OPTIONS=nocheck
rm -rf dist
python setup.py sdist --no-cython
cd dist
tar -xzf ${PROJECT}-*.tar.gz
cd ${PROJECT}*

if [ $1 = 3 ]
then
  echo Using Python 2+3 
  PATH=$CCPATH  python3 setup.py --command-packages=stdeb.command sdist_dsc --with-python2=True --with-python3=True --no-python3-scripts=True bdist_deb --no-cython
  sudo dpkg -i deb_dist/python3-${PROJECT}*.deb
else
  echo Using Python 2
  PATH=$CCPATH python setup.py --command-packages=stdeb.command bdist_deb --no-cython
fi

sudo su -c  "dpkg -i deb_dist/python-${PROJECT}*.deb"
cd ../..

