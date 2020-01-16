#!/bin/sh
#
#    Project: Silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2015-2020 European Synchrotron Radiation Facility, Grenoble, France
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
source_project=silx
version=$(python3 -c"import version; print(version.version)")
strictversion=$(python3 -c"import version; print(version.strictversion)")
debianversion=$(python3 -c"import version; print(version.debianversion)")

deb_name=$(echo "$source_project" | tr '[:upper:]' '[:lower:]')

# target system
if [ -f /etc/debian_version ]
then 
    debian_version=$(cat /etc/debian_version | cut -d. -f1 | grep -o '[0-9]*')
    if [ -z $debian_version ]
    then
    #we are probably on a ubuntu platform
        debian_version=$(cat /etc/debian_version | cut -d/ -f1)
        case $debian_version in
            stretch)
                debian_version=9
                ;;
            buster)
                debian_version=10
                ;;
            bullseye)
                debian_version=11
                ;;
        esac
    fi

else
    debian_version=0
fi
target_system=debian${debian_version}

project_directory="`dirname \"$0\"`"
project_directory="`( cd \"$project_directory\" && pwd )`" # absolutized
dist_directory=${project_directory}/dist/${target_system}
build_directory=${project_directory}/build/${target_system}

if [ -d /usr/lib/ccache ];
then
   export PATH=/usr/lib/ccache:$PATH
fi

usage="usage: $(basename "$0") [options]

Build the Debian ${debian_version} package of the ${project} library.

If the build succeed the directory dist/debian${debian_version} will
contains the packages.

optional arguments:
    --help          Show this help text
    --install       Install the packages generated at the end of
                    the process using 'sudo dpkg'
    --stdeb-py3     Build using stdeb for python3
    --stdeb-py2.py3 Build using stdeb for python2 and python3
    --debian9       Simulate a debian 9 Stretch system
    --debian10      Simulate a debian 10 Buster system
    --debian11      Simulate a debian 11 Bullseye system
"

install=0
use_stdeb=0
stdeb_all_python=0

while :
do
    case "$1" in
      -h | --help)
          echo "$usage"
          exit 0
          ;;
      --install)
          install=1
          shift
          ;;
      --stdeb-py2)
          use_stdeb=1
          stdeb_all_python=0
          shift
          ;;
      --stdeb-py2.py3)
          use_stdeb=1
          stdeb_all_python=1
          shift
          ;;
      --debian9)
          debian_version=9
          target_system=debian${debian_version}
          dist_directory=${project_directory}/dist/${target_system}
          build_directory=${project_directory}/build/${target_system}
          shift
          ;;
      --debian10)
          debian_version=10
          target_system=debian${debian_version}
          dist_directory=${project_directory}/dist/${target_system}
          build_directory=${project_directory}/build/${target_system}
          shift
          ;;
      --debian11)
          debian_version=11
          target_system=debian${debian_version}
          dist_directory=${project_directory}/dist/${target_system}
          build_directory=${project_directory}/build/${target_system}
          shift
          ;;
      -*)
          echo "Error: Unknown option: $1" >&2
          echo "$usage"
          exit 1
          ;;
      *)  # No more options
          break
          ;;
    esac
done

clean_up()
{
    echo "Clean working dir:"
    # clean up previous build
    rm -rf ${build_directory}
    # create the build context
    mkdir -p ${build_directory}
}

build_deb() {
    tarname=${project}_${debianversion}.orig.tar.gz
    clean_up
    python3 setup.py debian_src
    cp -f dist/${tarname} ${build_directory}
    if [ -f dist/${project}-testimages.tar.gz ]
    then
      cp -f dist/${project}-testimages.tar.gz ${build_directory}
    fi
    
    cd ${build_directory}
    tar -xzf ${tarname}
    
    directory=${project}-${strictversion}
    newname=${deb_name}_${debianversion}.orig.tar.gz
    
    #echo tarname $tarname newname $newname
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
      if [ ! -h  ${deb_name}_${debianversion}.orig-testimages.tar.gz ]
      then
        ln -s ${project}-testimages.tar.gz ${deb_name}_${debianversion}.orig-testimages.tar.gz
      fi
    fi
    
    cd ${directory}
    cp -r ${project_directory}/package/${target_system} debian
    cp ${project_directory}/copyright debian
    
    #handle test images
    if [ -f ../${deb_name}_${debianversion}.orig-testimages.tar.gz ]
    then
      if [ ! -d testimages ]
      then
        mkdir testimages
      fi
      cd testimages
      tar -xzf  ../${deb_name}_${debianversion}.orig-testimages.tar.gz
      cd ..
    else
      # Disable to skip tests during build
      echo No test data
      #export PYBUILD_DISABLE_python2=test
      #export PYBUILD_DISABLE_python3=test
      #export DEB_BUILD_OPTIONS=nocheck
    fi

    case $debian_version in
        9)
            debian_name=stretch
            ;;
        10)
            debian_name=buster
            ;;
        11)
            debian_name=bullseye
            ;;
    esac

    dch -v ${debianversion}-1 "upstream development build of ${project} ${version}"
    dch -D ${debian_name}-backports -l~bpo${debian_version}+ "${project} snapshot ${version} built for ${target_system}"
    #dch --bpo "${project} snapshot ${version} built for ${target_system}"
    dpkg-buildpackage -r
    rc=$?
    
    if [ $rc -eq 0 ]; then
      # move packages to dist directory
      echo Build succeeded...
      rm -rf ${dist_directory}
      mkdir -p ${dist_directory}
      mv ${build_directory}/*.deb ${dist_directory}
      mv ${build_directory}/*.x* ${dist_directory}
      mv ${build_directory}/*.dsc ${dist_directory}
      mv ${build_directory}/*.changes ${dist_directory}
      cd ../../..
    else
      echo Build failed, please investigate ...
      exit "$rc"
    fi
}


build_stdeb () {
    echo "Build for debian using stdeb"
    tarname=${project}-${strictversion}.tar.gz
    clean_up

    python setup.py sdist
    cp -f dist/${tarname} ${build_directory}
    cd ${build_directory}
    tar -xzf ${tarname}
    cd ${project}-${strictversion}

    if [ $stdeb_all_python -eq 1 ]; then
      echo Using Python 2+3
      python3 setup.py --command-packages=stdeb.command sdist_dsc --with-python2=True --with-python3=True --no-python3-scripts=True build --no-cython bdist_deb
      rc=$?
    else
      echo Using Python 3
      # bdist_deb feed /usr/bin using setup.py entry-points
      python3 setup.py --command-packages=stdeb.command build --no-cython bdist_deb
      rc=$?
    fi

    # move packages to dist directory
    rm -rf ${dist_directory}
    mkdir -p ${dist_directory}
    mv -f deb_dist/*.deb ${dist_directory}

    # back to the root
    cd ../../..
}


if [ $use_stdeb -eq 1 ]; then
    build_stdeb
else
    build_deb
fi


if [ $install -eq 1 ]; then
  sudo -v su -c  "dpkg -i ${dist_directory}/*.deb"
fi

exit "$rc"
