#!/usr/bin/env sh

#
# Called by setup.py when using build_man command.
# The build_man command takes care of the environment.
# PYTHON is passed using the environment to use the
# right executable.
#
# help2man will pass --help and --version to this script
#
# Use python setup.py build_man to use this script.
# It should generate a file ./build/man/silx.1
# 

if [ -z "$STATE" ]; then
    PYTHON=python
fi
$PYTHON scripts/silx-launcher.py $*

