#!/usr/bin/env sh

#
# It uses bootstrap.py to execute silx application
#
# The sed command will filter all the lines before ######
# 
# help2man will pass --help and --version to the script
#
# It must be used like that from root of the project:
# > mkdir -p build/man
# > help2man doc/man/wrapper.sh -o build/man/silx.l
# 
./bootstrap.py silx-launcher.py $* 2>/dev/null | sed -e '1,/######/d'
