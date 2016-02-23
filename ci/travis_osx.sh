#!/bin/bash
# Script for travis-CI Mac OS X specific setup.
#
# It provides 4 functions:
# - travis_osx_install_begin: Install pip and setup a virtualenv for the build
# - travis_osx_install_end: Deactivate the build virtualenv
# - travis_osx_run_begin: Setup a virtualenv for the tests
# - travis_osx_run_end: Deactivate test virtualenv
#
# On Linux, those functions do nothing.

# Directory where to create build virtualenv
VENV_BUILD_DIR=./venv_build

# Directory qhere to create test virtualenv
VENV_TEST_DIR=./venv_test


if [ "$TRAVIS_OS_NAME" != "osx" ]; then
    function travis_osx_install_begin {
        echo NoOp
    }
    function travis_osx_install_end {
        echo NoOp
    }
    function travis_osx_run_begin {
        echo NoOp
    }
    function travis_osx_run_end {
        echo NoOp
    }

else
    function travis_osx_install_begin {
        echo Mac OS X install begin: Install pip and setup build venv
        set -x  # echo on
        curl -O https://bootstrap.pypa.io/get-pip.py
        python get-pip.py --user

        pip install virtualenv --user
        virtualenv --version

        virtualenv --clear $VENV_BUILD_DIR

        set +x  # echo off
        echo "Activate virtualenv $VENV_BUILD_DIR"
        source $VENV_BUILD_DIR/bin/activate
    }

    function travis_osx_install_end {
        echo Mac OS X install end: Deactivate and delete virtualenv
        echo deactivate
        deactivate
        set -x  # echo on
        rm -rf $VENV_BUILD_DIR
        set +x  # echo off
    }


    function travis_osx_run_begin {
        echo Mac OS X run begin: Setup test venv

        set -x  # echo on
        virtualenv --clear $VENV_TEST_DIR
        set +x  # echo off
        echo "Activate virtualenv $VENV_TEST_DIR"
        source $VENV_TEST_DIR/bin/activate
    }

    function travis_osx_run_end {
        echo Mac OS X run end: Deactivate and delete virtualenv
        echo deactivate
        deactivate
        set -x  # echo on
        rm -rf $VENV_TEST_DIR
        set +x  # echo off
    }

fi
