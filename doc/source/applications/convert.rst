
silx convert
============

Purpose
-------

The *silx convert* command is provided to help with archiving legacy file
formats into HDF5 files.

You can refer to following tutorials for additional information
about the output format:

 - :doc:`../Tutorials/io`
 - :doc:`../Tutorials/convert`
 - :doc:`../Tutorials/specfile_to_hdf5`

Usage
-----

::

    silx convert [-h] [--file-pattern FILE_PATTERN] [-o OUTPUT_URI]
                    [-m MODE] [--begin BEGIN] [--end END] [--add-root-group]
                    [--overwrite-data] [--min-size MIN_SIZE]
                    [--chunks [CHUNKS]] [--compression [COMPRESSION]]
                    [--compression-opts COMPRESSION_OPTS] [--shuffle]
                    [--fletcher32] [--debug]
                    [input_files [input_files ...]]



Options
-------

::

  input_files           Input files (EDF, TIFF, SPEC...). When specifying
                        multiple files, you cannot specify both fabio images
                        and SPEC files. Multiple SPEC files will simply be
                        concatenated, with one entry per scan. Multiple image
                        files will be merged into a single entry with a stack
                        of images.


  -h, --help            show this help message and exit
  --file-pattern FILE_PATTERN
                        File name pattern for loading a series of indexed
                        image files (toto_%04d.edf). This argument is
                        incompatible with argument input_files. If an output
                        URI with a HDF5 path is provided, only the content of
                        the NXdetector group will be copied there. If no HDF5
                        path, or just "/", is given, a complete NXdata
                        structure will be created.
  -o OUTPUT_URI, --output-uri OUTPUT_URI
                        Output file name (HDF5). An URI can be provided to
                        write the data into a specific group in the output
                        file: /path/to/file::/path/to/group. If not provided,
                        the filename defaults to a timestamp: YYYYmmdd-
                        HHMMSS.h5
  -m MODE, --mode MODE  Write mode: "r+" (read/write, file must exist), "w"
                        (write, existing file is lost), "w-" (write, fail if
                        file exists) or "a" (read/write if exists, create
                        otherwise)
  --begin BEGIN         First file index, or first file indices to be
                        considered. This argument only makes sense when used
                        together with --file-pattern. Provide as many start
                        indices as there are indices in the file pattern, separated
                        by commas. Examples: "--filepattern toto_%d.edf
                        --begin 100", "--filepattern toto_%d_%04d_%02d.edf
                        --begin 100,2000,5".
  --end END             Last file index, or last file indices to be
                        considered. The same rules as with argument --begin
                        apply. Example: "--filepattern toto_%d_%d.edf --end
                        199,1999"
  --add-root-group      This option causes each input file to be written to a
                        specific root group with the same name as the file.
                        When merging multiple input files, this can help
                        preventing conflicts when datasets have the same name
                        (see --overwrite-data). This option is ignored when
                        using --file-pattern.
  --overwrite-data      If the output path exists and an input dataset has the
                        same name as an existing output dataset, overwrite the
                        output dataset (in modes "r+" or "a").
  --min-size MIN_SIZE   Minimum number of elements required to be in a dataset
                        to apply compression or chunking (default 500).
  --chunks <CHUNKS>     Chunk shape. Provide an argument that evaluates as a
                        python tuple (e.g. "(1024, 768)"). If this option is
                        provided without specifying an argument, the h5py
                        library will guess a chunk for you. Note that if you
                        specify an explicit chunking shape, it will be applied
                        identically to all datasets with a large enough size
                        (see --min-size).
  --compression <COMPRESSION>
                        Compression filter. By default, the datasets in the
                        output file are not compressed. If this option is
                        specified without argument, the GZIP compression is
                        used. Additional compression filters may be available,
                        depending on your HDF5 installation.
  --compression-opts COMPRESSION_OPTS
                        Compression options. For "gzip", this may be an
                        integer from 0 to 9, with a default of 4. This is only
                        supported for GZIP.
  --shuffle             Enables the byte shuffle filter. This may improve the
                        compression ratio for block oriented compressors like
                        GZIP or LZF.
  --fletcher32          Adds a checksum to each chunk to detect data
                        corruption.
  --debug               Set logging system in debug mode


Examples of usage
-----------------


Simple single file conversion to new output file::

    silx convert 31oct98.dat -o 31oct98.h5

Concatenation of all SPEC files in the current directory::

    silx convert *.dat -o all_SPEC.h5

Appending a file to an existing output file::

    silx convert ch09__mca_0005_0000_0008.edf -o archive.h5::/ch09__mca_0005_0000_0008 -m a --compression

Merging a list of single frame EDF files into a multiframe HDF5 file::

    silx convert --file-pattern ch09__mca_0005_0000_%d.edf -o ch09__mca_0005_0000_multiframe.h5
