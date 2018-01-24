# coding: utf-8
# /*##########################################################################
# Copyright (C) 2017 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Convert silx supported data files into HDF5 files"""

import ast
import os
import argparse
from glob import glob
import logging
import numpy
import re
import time

import silx.io
from silx.io.specfile import is_specfile

try:
    from silx.io import fabioh5
except ImportError:
    fabioh5 = None


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "12/09/2017"


_logger = logging.getLogger(__name__)
"""Module logger"""


def c_format_string_to_re(pattern_string):
    """

    :param pattern_string: C style format string with integer patterns
        (e.g. "%d", "%04d").
        Not supported: fixed length padded with whitespaces (e.g "%4d", "%-4d")
    :return: Equivalent regular expression (e.g. "\d+", "\d{4}")
    """
    # escape dots and backslashes
    pattern_string = pattern_string.replace(".", "\.")
    pattern_string = pattern_string.replace("\\", "\\\\")

    # %d
    pattern_string = pattern_string.replace("%d", "([-+]?\d+)")

    # %0nd
    for sub_pattern in re.findall("%0\d+d", pattern_string):
        n = int(re.search("%0(\d+)d", sub_pattern).group(1))
        if n == 1:
            re_sub_pattern = "([+-]?\d)"
        else:
            re_sub_pattern = "([\d+-]\d{%d})" % (n - 1)
        pattern_string = pattern_string.replace(sub_pattern, re_sub_pattern, 1)

    return pattern_string


def drop_indices_before_begin(filenames, regex, begin):
    """

    :param List[str] filenames: list of filenames
    :param str regex: Regexp used to find indices in a filename
    :param str begin: Comma separated list of begin indices
    :return: List of filenames with only indices >= begin
    """
    begin_indices = list(map(int, begin.split(",")))
    output_filenames = []
    for fname in filenames:
        m = re.match(regex, fname)
        file_indices = list(map(int, m.groups()))
        if len(file_indices) != len(begin_indices):
            raise IOError("Number of indices found in filename "
                          "does not match number of parsed end indices.")
        good_indices = True
        for i, fidx in enumerate(file_indices):
            if fidx < begin_indices[i]:
                good_indices = False
        if good_indices:
            output_filenames.append(fname)
    return output_filenames


def drop_indices_after_end(filenames, regex, end):
    """

    :param List[str] filenames: list of filenames
    :param str regex: Regexp used to find indices in a filename
    :param str end: Comma separated list of end indices
    :return: List of filenames with only indices <= end
    """
    end_indices = list(map(int, end.split(",")))
    output_filenames = []
    for fname in filenames:
        m = re.match(regex, fname)
        file_indices = list(map(int, m.groups()))
        if len(file_indices) != len(end_indices):
            raise IOError("Number of indices found in filename "
                          "does not match number of parsed end indices.")
        good_indices = True
        for i, fidx in enumerate(file_indices):
            if fidx > end_indices[i]:
                good_indices = False
        if good_indices:
            output_filenames.append(fname)
    return output_filenames


def are_files_missing_in_series(filenames, regex):
    """Return True if any file is missing in a list of filenames
    that are supposed to follow a pattern.

    :param List[str] filenames: list of filenames
    :param str regex: Regexp used to find indices in a filename
    :return: boolean
    :raises AssertionError: if a filename does not match the regexp
    """
    previous_indices = None
    for fname in filenames:
        m = re.match(regex, fname)
        assert m is not None, \
            "regex %s does not match filename %s" % (fname, regex)
        new_indices = list(map(int, m.groups()))
        if previous_indices is not None:
            for old_idx, new_idx in zip(previous_indices, new_indices):
                if (new_idx - old_idx) > 1:
                    _logger.error("Index increment > 1 in file series: "
                                  "previous idx %d, next idx %d",
                                  old_idx, new_idx)
                    return True
        previous_indices = new_indices
    return False


def are_all_specfile(filenames):
    """Return True if all files in a list are SPEC files.
    :param List[str] filenames: list of filenames
    """
    for fname in filenames:
        if not is_specfile(fname):
            return False
    return True


def contains_specfile(filenames):
    """Return True if any file in a list are SPEC files.
    :param List[str] filenames: list of filenames
    """
    for fname in filenames:
        if is_specfile(fname):
            return True
    return False


def main(argv):
    """
    Main function to launch the converter as an application

    :param argv: Command line arguments
    :returns: exit status
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_files',
        nargs="*",
        help='Input files (EDF, TIFF, SPEC...). When specifying multiple '
             'files, you cannot specify both fabio images and SPEC files. '
             'Multiple SPEC files will simply be concatenated, with one '
             'entry per scan. Multiple image files will be merged into '
             'a single entry with a stack of images.')
    # input_files and --filepattern are mutually exclusive
    parser.add_argument(
        '--file-pattern',
        help='File name pattern for loading a series of indexed image files '
             '(toto_%%04d.edf). This argument is incompatible with argument '
             'input_files. If an output URI with a HDF5 path is provided, '
             'only the content of the NXdetector group will be copied there. '
             'If no HDF5 path, or just "/", is given, a complete NXdata '
             'structure will be created.')
    parser.add_argument(
        '-o', '--output-uri',
        default=time.strftime("%Y%m%d-%H%M%S") + '.h5',
        help='Output file name (HDF5). An URI can be provided to write'
             ' the data into a specific group in the output file: '
             '/path/to/file::/path/to/group. '
             'If not provided, the filename defaults to a timestamp:'
             ' YYYYmmdd-HHMMSS.h5')
    parser.add_argument(
        '-m', '--mode',
        default="w-",
        help='Write mode: "r+" (read/write, file must exist), '
             '"w" (write, existing file is lost), '
             '"w-" (write, fail if file exists) or '
             '"a" (read/write if exists, create otherwise)')
    parser.add_argument(
        '--begin',
        help='First file index, or first file indices to be considered. '
             'This argument only makes sense when used together with '
             '--file-pattern. Provide as many start indices as there '
             'are indices in the file pattern, separated by commas. '
             'Examples: "--filepattern toto_%%d.edf --begin 100", '
             ' "--filepattern toto_%%d_%%04d_%%02d.edf --begin 100,2000,5".')
    parser.add_argument(
        '--end',
        help='Last file index, or last file indices to be considered. '
             'The same rules as with argument --begin apply. '
             'Example: "--filepattern toto_%%d_%%d.edf --end 199,1999"')
    parser.add_argument(
        '--add-root-group',
        action="store_true",
        help='This option causes each input file to be written to a '
             'specific root group with the same name as the file. When '
             'merging multiple input files, this can help preventing conflicts'
             ' when datasets have the same name (see --overwrite-data). '
             'This option is ignored when using --file-pattern.')
    parser.add_argument(
        '--overwrite-data',
        action="store_true",
        help='If the output path exists and an input dataset has the same'
             ' name as an existing output dataset, overwrite the output '
             'dataset (in modes "r+" or "a").')
    parser.add_argument(
        '--min-size',
        type=int,
        default=500,
        help='Minimum number of elements required to be in a dataset to '
             'apply compression or chunking (default 500).')
    parser.add_argument(
        '--chunks',
        nargs="?",
        const="auto",
        help='Chunk shape. Provide an argument that evaluates as a python '
             'tuple (e.g. "(1024, 768)"). If this option is provided without '
             'specifying an argument, the h5py library will guess a chunk for '
             'you. Note that if you specify an explicit chunking shape, it '
             'will be applied identically to all datasets with a large enough '
             'size (see --min-size). ')
    parser.add_argument(
        '--compression',
        nargs="?",
        const="gzip",
        help='Compression filter. By default, the datasets in the output '
             'file are not compressed. If this option is specified without '
             'argument, the GZIP compression is used. Additional compression '
             'filters may be available, depending on your HDF5 installation.')

    def check_gzip_compression_opts(value):
        ivalue = int(value)
        if ivalue < 0 or ivalue > 9:
            raise argparse.ArgumentTypeError(
                "--compression-opts must be an int from 0 to 9")
        return ivalue

    parser.add_argument(
        '--compression-opts',
        type=check_gzip_compression_opts,
        help='Compression options. For "gzip", this may be an integer from '
             '0 to 9, with a default of 4. This is only supported for GZIP.')
    parser.add_argument(
        '--shuffle',
        action="store_true",
        help='Enables the byte shuffle filter. This may improve the compression '
             'ratio for block oriented compressors like GZIP or LZF.')
    parser.add_argument(
        '--fletcher32',
        action="store_true",
        help='Adds a checksum to each chunk to detect data corruption.')
    parser.add_argument(
        '--debug',
        action="store_true",
        default=False,
        help='Set logging system in debug mode')

    options = parser.parse_args(argv[1:])

    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    # Import after parsing --debug
    try:
        # it should be loaded before h5py
        import hdf5plugin  # noqa
    except ImportError:
        _logger.debug("Backtrace", exc_info=True)
        hdf5plugin = None

    try:
        import h5py
        from silx.io.convert import write_to_h5
    except ImportError:
        _logger.debug("Backtrace", exc_info=True)
        h5py = None
        write_to_h5 = None

    if h5py is None:
        message = "Module 'h5py' is not installed but is mandatory."\
            + " You can install it using \"pip install h5py\"."
        _logger.error(message)
        return -1

    if hdf5plugin is None:
        message = "Module 'hdf5plugin' is not installed. It supports additional hdf5"\
            + " compressions. You can install it using \"pip install hdf5plugin\"."
        _logger.debug(message)

    # Process input arguments (mutually exclusive arguments)
    if bool(options.input_files) == bool(options.file_pattern is not None):
        if not options.input_files:
            message = "You must specify either input files (at least one), "
            message += "or a file pattern."
        else:
            message = "You cannot specify input files and a file pattern"
            message += " at the same time."
        _logger.error(message)
        return -1
    elif options.input_files:
        # some shells (windows) don't interpret wildcard characters (*, ?, [])
        old_input_list = list(options.input_files)
        options.input_files = []
        for fname in old_input_list:
            globbed_files = glob(fname)
            if not globbed_files:
                # no files found, keep the name as it is, to raise an error later
                options.input_files += [fname]
            else:
                # glob does not sort files, but the bash shell does
                options.input_files += sorted(globbed_files)
    else:
        # File series
        dirname = os.path.dirname(options.file_pattern)
        file_pattern_re = c_format_string_to_re(options.file_pattern) + "$"
        files_in_dir = glob(os.path.join(dirname, "*"))
        _logger.debug("""
            Processing file_pattern
            dirname: %s
            file_pattern_re: %s
            files_in_dir: %s
            """, dirname, file_pattern_re, files_in_dir)

        options.input_files = sorted(list(filter(lambda name: re.match(file_pattern_re, name),
                                                 files_in_dir)))
        _logger.debug("options.input_files: %s", options.input_files)

        if options.begin is not None:
            options.input_files = drop_indices_before_begin(options.input_files,
                                                            file_pattern_re,
                                                            options.begin)
            _logger.debug("options.input_files after applying --begin: %s",
                          options.input_files)

        if options.end is not None:
            options.input_files = drop_indices_after_end(options.input_files,
                                                         file_pattern_re,
                                                         options.end)
            _logger.debug("options.input_files after applying --end: %s",
                          options.input_files)

        if are_files_missing_in_series(options.input_files,
                                       file_pattern_re):
            _logger.error("File missing in the file series. Aborting.")
            return -1

        if not options.input_files:
            _logger.error("No file matching --file-pattern found.")
            return -1

    # Test that the output path is writeable
    if "::" in options.output_uri:
        output_name, hdf5_path = options.output_uri.split("::")
    else:
        output_name, hdf5_path = options.output_uri, "/"

    if os.path.isfile(output_name):
        if options.mode == "w-":
            _logger.error("Output file %s exists and mode is 'w-' (default)."
                          " Aborting. To append data to an existing file, "
                          "use 'a' or 'r+'.",
                          output_name)
            return -1
        elif not os.access(output_name, os.W_OK):
            _logger.error("Output file %s exists and is not writeable.",
                          output_name)
            return -1
        elif options.mode == "w":
            _logger.info("Output file %s exists and mode is 'w'. "
                         "Overwriting existing file.", output_name)
        elif options.mode in ["a", "r+"]:
            _logger.info("Appending data to existing file %s.",
                         output_name)
    else:
        if options.mode == "r+":
            _logger.error("Output file %s does not exist and mode is 'r+'"
                          " (append, file must exist). Aborting.",
                          output_name)
            return -1
        else:
            _logger.info("Creating new output file %s.",
                         output_name)

    # Test that all input files exist and are readable
    bad_input = False
    for fname in options.input_files:
        if not os.access(fname, os.R_OK):
            _logger.error("Cannot read input file %s.",
                          fname)
            bad_input = True
    if bad_input:
        _logger.error("Aborting.")
        return -1

    # create_dataset special args
    create_dataset_args = {}
    if options.chunks is not None:
        if options.chunks.lower() in ["auto", "true"]:
            create_dataset_args["chunks"] = True
        else:
            try:
                chunks = ast.literal_eval(options.chunks)
            except (ValueError, SyntaxError):
                _logger.error("Invalid --chunks argument %s", options.chunks)
                return -1
            if not isinstance(chunks, (tuple, list)):
                _logger.error("--chunks argument str does not evaluate to a tuple")
                return -1
            else:
                nitems = numpy.prod(chunks)
                nbytes = nitems * 8
                if nbytes > 10**6:
                    _logger.warning("Requested chunk size might be larger than"
                                    " the default 1MB chunk cache, for float64"
                                    " data. This can dramatically affect I/O "
                                    "performances.")
                create_dataset_args["chunks"] = chunks

    if options.compression is not None:
        create_dataset_args["compression"] = options.compression

    if options.compression_opts is not None:
        create_dataset_args["compression_opts"] = options.compression_opts

    if options.shuffle:
        create_dataset_args["shuffle"] = True

    if options.fletcher32:
        create_dataset_args["fletcher32"] = True

    if (len(options.input_files) > 1 and
            not contains_specfile(options.input_files) and
            not options.add_root_group) or options.file_pattern is not None:
        # File series -> stack of images
        if fabioh5 is None:
            # return a helpful error message if fabio is missing
            try:
                import fabio
            except ImportError:
                _logger.error("The fabio library is required to convert"
                              " edf files. Please install it with 'pip "
                              "install fabio` and try again.")
            else:
                # unexpected problem in silx.io.fabioh5
                raise
            return -1
        input_group = fabioh5.File(file_series=options.input_files)
        if hdf5_path != "/":
            # we want to append only data and headers to an existing file
            input_group = input_group["/scan_0/instrument/detector_0"]
        with h5py.File(output_name, mode=options.mode) as h5f:
            write_to_h5(input_group, h5f,
                        h5path=hdf5_path,
                        overwrite_data=options.overwrite_data,
                        create_dataset_args=create_dataset_args,
                        min_size=options.min_size)

    elif len(options.input_files) == 1 or \
            are_all_specfile(options.input_files) or\
            options.add_root_group:
        # single file, or spec files
        h5paths_and_groups = []
        for input_name in options.input_files:
            hdf5_path_for_file = hdf5_path
            if options.add_root_group:
                hdf5_path_for_file = hdf5_path.rstrip("/") + "/" + os.path.basename(input_name)
            try:
                h5paths_and_groups.append((hdf5_path_for_file,
                                           silx.io.open(input_name)))
            except IOError:
                _logger.error("Cannot read file %s. If this is a file format "
                              "supported by the fabio library, you can try to"
                              " install fabio (`pip install fabio`)."
                              " Aborting conversion.",
                              input_name)
                return -1

        with h5py.File(output_name, mode=options.mode) as h5f:
            for hdf5_path_for_file, input_group in h5paths_and_groups:
                write_to_h5(input_group, h5f,
                            h5path=hdf5_path_for_file,
                            overwrite_data=options.overwrite_data,
                            create_dataset_args=create_dataset_args,
                            min_size=options.min_size)

    else:
        # multiple file, SPEC and fabio images mixed
        _logger.error("Multiple files with incompatible formats specified. "
                      "You can provide multiple SPEC files or multiple image "
                      "files, but not both.")
        return -1

    with h5py.File(output_name, mode="r+") as h5f:
        # append "silx convert" to the creator attribute, for NeXus files
        previous_creator = h5f.attrs.get("creator", b"").decode()
        creator = "silx convert (v%s)" % silx.version
        # only if it not already there
        if creator not in previous_creator:
            h5f.attrs["creator"] = \
                numpy.string_(previous_creator + "; " + creator)

    return 0
