#!/bin/sh
#
#    Project: Silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2015-2016 European Synchrotron Radiation Facility, Grenoble, France
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

project=silx 
debian=$(grep -o '[0-9]*' /etc/issue)
version=$(python -c"import version; print(version.version)")
strictversion=$(python -c"import version; print(version.strictversion)") 
tarname=${project}_${strictversion}.orig.tar.gz

if [ -d /usr/lib/ccache ];
then 
   export PATH=/usr/lib/ccache:$PATH 
fi

export PYBUILD_DISABLE_python2=test
export PYBUILD_DISABLE_python3=test
export DEB_BUILD_OPTIONS=nocheck

python setup.py debian_src
cp dist/${tarname} package
cd package
tar -xzf ${tarname}
newname=${project}_${strictversion}.orig.tar.gz
directory=${project}-${strictversion}
echo tarname $tarname newname $newname
if [ $tarname != $newname ]
then
    ln -s ${tarname} ${newname}
fi
cd ${directory}
cp -r ../debian .
cp ../../copyright debian
dch -v ${strictversion}-1 "upstream development build of silx ${version}"
dch --bpo "silx snapshot version ${version} built for debian ${debian}"
dpkg-buildpackage -r
rc=$?
if [ $rc -eq 0 ]
then
  cd ..
  sudo su -c  "dpkg -i *.deb"
  #rm -rf ${directory}
  cd ..
else
  echo Build failed, please investigate ...
  cd ../..
fi

