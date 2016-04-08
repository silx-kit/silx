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


python setup.py debian_src
cp -f dist/${tarname} package

if [ -f dist/${project}-testimages.tar.gz ]
then
  cp -f dist/${project}-testimages.tar.gz package
fi

cd package
tar -xzf ${tarname}
newname=${project}_${strictversion}.orig.tar.gz
directory=${project}-${strictversion}
echo tarname $tarname newname $newname
if [ $tarname != $newname ]
then
  if [ -h $newname ]
  then
    rm ${newname}
  fi
    ln -s ${tarname} ${newname}
fi

if [ -f ${project}-testimages.tar.gz ]
then
  if [ ! -h  python-${project}_${strictversion}.orig-testimages.tar.gz ]
  then
    ln -s ${project}-testimages.tar.gz python-${project}_${strictversion}.orig-testimages.tar.gz
  fi
fi

cd ${directory}
cp -r ../debian .
cp ../../copyright debian

#handle test images
if [ -f ../python-${project}_${strictversion}.orig-testimages.tar.gz ]
then
  if [ ! -d testimages ]
  then
    mkdir testimages
  fi
  cd testimages
  tar -xzf  ../../python-${project}_${strictversion}.orig-testimages.tar.gz
  cd ..
else
  # Disable to skip tests during build
  export PYBUILD_DISABLE_python2=test
  export PYBUILD_DISABLE_python3=test
  export DEB_BUILD_OPTIONS=nocheck
fi

dch -v ${strictversion}-1 "upstream development build of ${project} ${version}"
dch --bpo "${project} snapshot ${version} built for debian ${debian}"
dpkg-buildpackage -r
rc=$?
if [ $rc -eq 0 ]
then
  cd ..
  if [ -z $1 ]
  #Provide an option name for avoiding auto-install
  then
    sudo su -c  "dpkg -i *.deb"
  fi
  #rm -rf ${directory}
  cd ..
else
  echo Build failed, please investigate ...
  cd ../..
fi

