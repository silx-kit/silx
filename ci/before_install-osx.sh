# Script for travis-CI Mac OS X specific setup.
# source this script with PYTHON_VERSION env variable set

VENV_DIR=./venv

# Use brew for python3
if [ "$PYTHON_VERSION" == "3" ];
then
    brew update;
    brew install python3;
    PYTHON_EXE=`brew list python3 | grep "bin/python3$" | head -n 1`;
    # Create virtual env
    $PYTHON_EXE -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
fi

# Alternative python installation using miniconda
#curl -o miniconda_installer.sh "https://repo.continuum.io/miniconda/Miniconda$PYTHON_VERSION-latest-MacOSX-x86_64.sh"
#bash miniconda_installer.sh -b -p miniconda
#export PATH="`pwd`/miniconda/bin":$PATH
