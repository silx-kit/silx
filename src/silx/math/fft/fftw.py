#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2018-2022 European Synchrotron Radiation Facility
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
# ###########################################################################*/

import os
from sys import executable as sys_executable
from socket import gethostname
from tempfile import gettempdir
from pathlib import Path
import numpy as np
from .basefft import BaseFFT, check_version

try:
    import pyfftw

    __have_fftw__ = True
except ImportError:
    __have_fftw__ = False


# Check pyfftw version
__required_pyfftw_version__ = "0.10.0"
if __have_fftw__:
    __have_fftw__ = check_version(pyfftw, __required_pyfftw_version__)


class FFTW(BaseFFT):
    """Initialize a FFTW plan.

    Please see FFT class for parameters help.

    FFTW-specific parameters
    -------------------------

    :param bool check_alignment:
        If set to True and "data" is provided, this will enforce the input data
        to be "byte aligned", which might imply extra memory usage.
    :param int num_threads:
        Number of threads for computing FFT.
    """

    def __init__(
        self,
        shape=None,
        dtype=None,
        template=None,
        shape_out=None,
        axes=None,
        normalize="rescale",
        check_alignment=False,
        num_threads=1,
    ):
        if not (__have_fftw__):
            raise ImportError(
                "Please install pyfftw >= %s to use the FFTW back-end"
                % __required_pyfftw_version__
            )
        super().__init__(
            shape=shape,
            dtype=dtype,
            template=template,
            shape_out=shape_out,
            axes=axes,
            normalize=normalize,
        )
        self.check_alignment = check_alignment
        self.num_threads = num_threads
        self.backend = "fftw"

        self.allocate_arrays()
        self.set_fftw_flags()
        self.compute_forward_plan()
        self.compute_inverse_plan()
        self.refs = {
            "data_in": self.data_in,
            "data_out": self.data_out,
        }

    # About normalization with norm="none", issues about pyfftw version :
    # --------------- pyfftw 0.12 ---------------
    # FFT :
    # normalise_idft --> 1
    # not normalise_idft --> 1
    # IFFT :
    # normalise_idft --> 1 / N
    # not normalise_idft --> 1
    # --------------- pyfftw 0.13 ---------------
    # FFT :
    # normalise_idft --> 1
    # not normalise_idft --> 1 / N (this normalization is incorrect, doc says contrary)
    # IFFT :
    # normalise_idft --> 1 / N
    # not normalise_idft --> 1

    # Solution :
    # select 'normalise_idft' for FFT and 'not normalise_idft' for IFFT
    # => behavior is the same in both version :)

    def set_fftw_flags(self):
        self.fftw_flags = ("FFTW_MEASURE",)  # TODO
        self.fftw_planning_timelimit = None  # TODO

        # To skip normalization on norm="none", we should
        # flip 'normalise_idft' to normalize no-where (see comments up):
        #
        # and :
        # ortho (orthogonal normalization)
        # ortho = True : forward -> 1/sqrt(N), backward -> 1/sqrt(N)

        self.fftw_norm_modes = {
            "rescale": (
                {"ortho": False, "normalise_idft": True},  # fft
                {"ortho": False, "normalise_idft": True},  # ifft
            ),
            "ortho": (
                {"ortho": True, "normalise_idft": False},  # fft
                {"ortho": True, "normalise_idft": False},  # ifft
            ),
            "none": (
                {"ortho": False, "normalise_idft": True},  # fft
                {"ortho": False, "normalise_idft": False},  # ifft
            ),
        }
        if self.normalize not in self.fftw_norm_modes:
            raise ValueError(
                "Unknown normalization mode %s. Possible values are %s"
                % (self.normalize, self.fftw_norm_modes.keys())
            )
        self.fftw_norm_mode = self.fftw_norm_modes[self.normalize]

    def _allocate(self, shape, dtype):
        return pyfftw.zeros_aligned(shape, dtype=dtype)

    def check_array(self, array, shape, dtype, copy=True):
        if array.shape != shape:
            raise ValueError(
                "Invalid data shape: expected %s, got %s" % (shape, array.shape)
            )
        if array.dtype != dtype:
            raise ValueError(
                "Invalid data type: expected %s, got %s" % (dtype, array.dtype)
            )

    def set_data(self, self_array, array, shape, dtype, copy=True, name=None):
        """
        :param self_array: array owned by the current instance
                           (either self.data_in or self.data_out).
        :type: numpy.ndarray
        :param self_array: data to set
        :type: numpy.ndarray
        :type tuple shape: shape of the array
        :param dtype: type of the array
        :type: numpy.dtype
        :param bool copy: should we copy the array
        :param str name: name of the array

        Copies are avoided when possible.
        """
        self.check_array(array, shape, dtype)
        if id(self.refs[name]) == id(array):
            # nothing to do: fft is performed on self.data_in or self.data_out
            arr_to_use = self.refs[name]
        if self.check_alignment and not (pyfftw.is_byte_aligned(array)):
            # If the array is not properly aligned,
            # create a temp. array copy it to self.data_in or self.data_out
            self_array[:] = array[:]
            arr_to_use = self_array
        else:
            # If the array is properly aligned, use it directly
            if copy:
                arr_to_use = np.copy(array)
            else:
                arr_to_use = array
        return arr_to_use

    def compute_forward_plan(self):
        self.plan_forward = pyfftw.FFTW(
            self.data_in,
            self.data_out,
            axes=self.axes,
            direction="FFTW_FORWARD",
            flags=self.fftw_flags,
            threads=self.num_threads,
            planning_timelimit=self.fftw_planning_timelimit,
        )

    def compute_inverse_plan(self):
        self.plan_inverse = pyfftw.FFTW(
            self.data_out,
            self.data_in,
            axes=self.axes,
            direction="FFTW_BACKWARD",
            flags=self.fftw_flags,
            threads=self.num_threads,
            planning_timelimit=self.fftw_planning_timelimit,
        )

    def fft(self, array, output=None):
        """
        Perform a (forward) Fast Fourier Transform.

        :param numpy.ndarray array:
            Input data. Must be consistent with the current context.
        :param numpy.ndarray output:
            Optional output data.
        """
        data_in = self.set_input_data(array, copy=False)
        data_out = self.set_output_data(output, copy=False)
        self.plan_forward.update_arrays(data_in, data_out)
        # execute.__call__ does both update_arrays() and normalization
        self.plan_forward(  # [0] --> fft
            ortho=self.fftw_norm_mode[0]["ortho"],
            normalise_idft=self.fftw_norm_mode[0]["normalise_idft"],
        )
        self.plan_forward.update_arrays(self.refs["data_in"], self.refs["data_out"])
        return data_out

    def ifft(self, array, output=None):
        """
        Perform a (inverse) Fast Fourier Transform.

        :param numpy.ndarray array:
            Input data. Must be consistent with the current context.
        :param numpy.ndarray output:
            Optional output data.
        """
        data_in = self.set_output_data(array, copy=False)
        data_out = self.set_input_data(output, copy=False)
        self.plan_inverse.update_arrays(
            data_in, data_out
        )  # TODO why in/out when it is out/in everywhere else in the function
        # execute.__call__ does both update_arrays() and normalization
        self.plan_inverse(  # [1] --> ifft
            ortho=self.fftw_norm_mode[1]["ortho"],
            normalise_idft=self.fftw_norm_mode[1]["normalise_idft"],
        )
        self.plan_inverse.update_arrays(self.refs["data_out"], self.refs["data_in"])
        return data_out



def get_wisdom_metadata():
    """
    Get metadata on the current platform.
    FFTW wisdom works with varying performance depending on whether the plans are re-used
    on the same machine/architecture/etc.
    For more information: https://www.fftw.org/fftw3_doc/Caveats-in-Using-Wisdom.html
    """
    return {
        # "venv"
        "executable":  sys_executable,
        # encapsulates sys.platform, platform.machine(), platform.architecture(), platform.libc_ver(), ...
        "hostname": gethostname(),
        "available_threads": len(os.sched_getaffinity(0)),
    }


def export_wisdom(fname, on_existing="overwrite"):
    """
    Export the current FFTW wisdom to a file.

    :param str fname:
        Path to the file where the wisdom is to be exported
    :param str on_existing:
        What do do when the target file already exists.
        Possible options are:
           - raise: raise an error and exit
           - overwrite: overwrite the file with the current wisdom
           - append: Import the already existing wisdom, and dump the newly combined wisdom to this file
    """
    if os.path.isfile(fname):
        if on_existing == "raise":
            raise ValueError("File already exists: %s" % fname)
        if on_existing == "append":
            import_wisdom(fname, on_mismatch="ignore") # ?
    current_wisdom = pyfftw.export_wisdom()
    res = get_wisdom_metadata()
    for i, w in enumerate(current_wisdom):
        res[str(i)] = np.array(w)
    np.savez_compressed(fname, **res)


def import_wisdom(fname, match=["hostname"], on_mismatch="warn"):
    """
    Import FFTW wisdom for a .npz file.

    :param str fname:
        Path to the .npz file containing FFTW wisdom
    :param list match:
        List of elements that must match when importing wisdom.
        If match=["hostname"] (default), this class will only load wisdom that was saved
        on the current machine, and discard everything else.
        If match=["hostname", "executable"], wisdom will only be loaded if the file was
        created on the same machine and by the same python executable.
    :param str on_mismatch:
        What to do when the file wisdom does not match the current platform.
        Available options:
          - "raise": raise an error (crash)
          - "warn": print a warning, don't crash
          - "ignore": do nothing
    """
    def handle_mismatch(item, loaded_value, current_value):
        msg = "Platform configuration mismatch: %s: currently have '%s', loaded '%s'" % (item, current_value, loaded_value)
        if on_mismatch == "raise":
            raise ValueError(msg)
        if on_mismatch == "warn":
            print(msg)

    wis_metadata = get_wisdom_metadata()
    loaded_wisdom = np.load(fname)
    for metadata_name in match:
        if metadata_name not in wis_metadata:
            raise ValueError(
                "Cannot match metadata '%s'. Available are: %s" % (match, str(wis_metadata.keys()))
            )
        if loaded_wisdom[metadata_name] != wis_metadata[metadata_name]:
            handle_mismatch(metadata_name, loaded_wisdom[metadata_name], wis_metadata[metadata_name])
            return
    w = tuple(loaded_wisdom[k][()] for k in loaded_wisdom.keys() if k not in wis_metadata.keys())
    pyfftw.import_wisdom(w)


def get_wisdom_file(directory=None, name_template="fftw_wisdom_{whoami}_{hostname}.npz", create_dirs=True):
    """
    Get a file path for storing FFTW wisdom.

    :param str directory:
        Directory where the file is created. By default, files are written in a temporary directory.
    :param str name_template:
        File name pattern. The following patterns can be used:
           - {whoami}: current username
           - {hostname}: machine name
    :param bool create_dirs:
        Whether to create (possibly nested) directories if needed.
    """
    directory = directory or gettempdir()
    file_basename = name_template.format(
        whoami=os.getlogin(),
        hostname=gethostname()
    )
    out_file = os.path.join(directory, file_basename)
    if create_dirs:
        Path(os.path.dirname(out_file)).mkdir(parents=True, exist_ok=True)
    return out_file
