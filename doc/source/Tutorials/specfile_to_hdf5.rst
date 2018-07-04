
SpecFile as HDF5
================

Introduction to SPEC data files
-------------------------------

SPEC data files are ASCII files.
They contain two general types of line blocks:

 - header lines starting with a ``#`` immediately followed by one or more characters
   identifying the information that follows
 - data lines

Header lines
++++++++++++

There are two types of headers. The first type is the *file header*. File headers always start
with a ``#F`` line.
The metadata stored in a file header refers to all the content of the data file, until a
new file header is encountered. There can be more than one file header, but a file with
multiple headers can be treated as a set of multiple SPEC files concatenated into a single one.
File headers are sometimes missing.

A file header contains general information:

 - ``#F`` - file name
 - ``#E`` - epoch
 - ``#D`` - file time and date
 - ``#C`` - First comment (SPEC title, SPEC user)
 - ``#O`` - Motor names (separated by at least two blank spaces)

The second type of header is the *scan header*. A scan header must start with a ``#S`` line
and must be preceded by an empty line. This also applies to files without file headers: in
such a case, the file must start with an empty line.
The metadata stored in scan headers refers to a single block of data lines.

A scan header contains the following information:

 - ``#S`` - Mandatory first line showing the scan number and the
   command that was used to record the scan
 - ``#D`` - scan time and date
 - ``#Q`` - *H, K, L* values
 - ``#P`` - Motor positions (the corresponding motor names are in the file header ``#O``)
 - ``#N`` - Number of data columns in the following data block
 - ``#L`` - Column labels (``#N`` labels separated by two blank spaces)

Users can also define their own type of header lines in their macros.

There can sometimes be a block of scan header lines after a data block, but before the ``#S`` of the next
scan.

Data lines
++++++++++

Data blocks are structured as 2D arrays. Each line contains ``#N`` values, each value
corresponding to the label with the same position in the ``#L`` scan header line.
This implies that each column corresponds to one series of measurements.

A column typically includes motor positions for a given positioner, a timestamp or the measurement
of a sensor.

MCA data
++++++++

More recent SPEC files can also comprise multi-channel analyser data, between *normal* data lines.
A multichannel analyser records multiple values per single measurement.
This is typically a histogram of number of counts against channels (*MCA spectrum*), to analyse the energy distribution
of a process.

SPEC data files containing MCA data have additional scan header lines:

 - ``#@MCA %16C`` - a spectrum will usually extend over more than one line.
   This indicates a number of 16 values per line.
 - ``#@CHANN`` - contains 4 values:

   - the number of channels per spectrum
   - the first channel number
   - the last channel number
   - the increment between two channel numbers (usually 1)
 - ``#@CALIB`` - 3 polynomial calibration values a, b, c. ( i.e. energy = a + b * channel + c * channel ^ 2)
 - ``#@CTIME`` - 3 values: preset time, life time, elapsed time

The actual MCA data for a single spectrum usually spans over multiple lines.
A spectrum starts on a new line with ``@A``, and when it spans over multiple lines, all
lines except the last one end by a continuation character ``\``.

Example of SPEC files
+++++++++++++++++++++

Example of file header::

    #F ./data/binary_mixtures_mca1.100211
    #E 1295362398
    #D Thu Feb 10 22:43:43 2011
    #C id10b  User = opid10
    #O0    delta     gamma     omega     theta        mu     sigma    sigmat        xt
    #O1       zt       zt1       thd      chid      rhod        xd        yd        zd
    #O2     att0      arcf        zf      PhiD     phigH     chigH       ygH
    #O3      zgH     phigV     chigV       xgV       ygV       zgV   gslithg   gslitho
    #O4  gslitvo   gslitvg    slit1T    slit1B    slit1F    slit1R   slit1hg   slit1ho
    #O5  slit1vg   slit1vo       s0T       s0B       s0R       s0F
    #O6     s0hg      s0ho      s0vg      s0vo       TRT
    #O7       pi    trough       hv1    mpxthl    apdwin    apdthl     apdhv     xcrl2
    #O8   thcrl2     zcrl2     picou     picod    vdrift    vmulti      vglo      vghi
    #O9     rien

Example of scan and data block, without MCA::

    #S 30  ascan  tz3 29.35 29.75  100 0.5
    #D Sat Oct 31 15:43:21 1998
    #T 0.5  (Seconds)
    #G0 0
    #G1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #G2 0
    #Q
    #P0 40.135381 40.262001 65.6 70 35 -1.83 0 -36.1
    #P1 0 0 -1.98 0 0 35.6 86.2 -29.5
    #P2 3.0688882 24.893749 295.98749 28 -27.249938
    #N 22
    #L TZ3    Epoch  Seconds  If2  If3  If5  If6  If7  If8  I0  It  ItdI0  If1dI0  If2dI0  If3dI0  If4dI0  If5dI0  If6dI0  If7dI0  If8dI0  If1  If4
    29.35  45246 0.000264 478 302 206 201 209 264 177860 646 0.00363207 0.00468346 0.00268751 0.00169796 0.00146745 0.00115821 0.0011301 0.00117508 0.00148431 833 261
    29.353976  45249 0.000295 549 330 219 208 227 295 178021 684 0.00384224 0.00537577 0.00308391 0.00185371 0.00158408 0.00123019 0.0011684 0.00127513 0.00165711 957 282
    29.357952  45251 0.000313 604 368 231 215 229 313 178166 686 0.00385034 0.00603931 0.0033901 0.00206549 0.00166698 0.00129654 0.00120674 0.00128532 0.00175679 1076 297
    29.362028  45253 0.000333 671 390 237 226 236 333 178387 672 0.00376709 0.00683346 0.00376148 0.00218626 0.00176582 0.00132857 0.00126691 0.00132297 0.00186673 1219 315
    29.366004  45256 0.000343 734 419 248 229 236 343 178082 664 0.00372862 0.00765939 0.0041217 0.00235285 0.00185308 0.00139262 0.00128592 0.00132523 0.00192608 1364 330
    29.36998  45258 0.00036 847 448 254 229 248 360 178342 668 0.00374561 0.00857342 0.0047493 0.00251203 0.00194009 0.00142423 0.00128405 0.00139059 0.00201859 1529 346

Synthetic example of a file with 3 scans. The last scan includes MCA data.

::

    #F /tmp/sf.dat
    #E 1455180875
    #D Thu Feb 11 09:54:35 2016
    #C imaging  User = opid17
    #O0 Pslit HGap  MRTSlit UP  MRTSlit DOWN
    #O1 Sslit1 VOff  Sslit1 HOff  Sslit1 VGap
    #o0 pshg mrtu mrtd
    #o2 ss1vo ss1ho ss1vg

    #S 1  ascan  ss1vo -4.55687 -0.556875  40 0.2
    #D Thu Feb 11 09:55:20 2016
    #T 0.2  (Seconds)
    #P0 180.005 -0.66875 0.87125
    #P1 14.74255 16.197579 12.238283
    #N 3
    #L MRTSlit UP  second column  3rd_col
    -1.23 5.89  8
    8.478100E+01  5 1.56
    3.14 2.73 -3.14
    1.2 2.3 3.4

    #S 25  ascan  c3th 1.33245 1.52245  40 0.15
    #D Sat 2015/03/14 03:53:50
    #P0 80.005 -1.66875 1.87125
    #P1 4.74255 6.197579 2.238283
    #N 4
    #L column0  column1  col2  col3
    0.0 0.1 0.2 0.3
    1.0 1.1 1.2 1.3
    2.0 2.1 2.2 2.3
    3.0 3.1 3.2 3.3

    #S 1 aaaaaa
    #D Thu Feb 11 10:00:32 2016
    #@MCA %16C
    #@CHANN 20 0 19 1
    #@CALIB 1.2 2.3 3.4
    #@CTIME 123.4 234.5 345.6
    #N 2
    #L uno  duo
    1 2
    @A 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\
    16 17 18 19
    3 4
    @A 0 0 2 4 15 10 5 1 0 0 0 0 1 0 0 0\
    0 0 0 0
    5 6
    @A 0 0 0 0 5 7 2 0 0 0 0 0 1 0 0 0\
    0 0 0 1

Reading a SpecFile as an HDF5 file
----------------------------------

Introduction to the spech5 module
+++++++++++++++++++++++++++++++++

The *silx* module :mod:`silx.io.spech5` can be used to expose SPEC files in a hierarchical tree structure
and access them through an API that mimics the *h5py* Python library used to read HDF5 files.

The structure exposed is as follows::

  /
      1.1/
          title = "…"
          start_time = "…"
          instrument/
              specfile/
                  file_header = "…"
                  scan_header = "…"
              positioners/
                  motor_name = value
                  …
              mca_0/
                  data = …
                  calibration = …
                  channels = …
                  preset_time = …
                  elapsed_time = …
                  live_time = …

              mca_1/
                  …
              …
          measurement/
              colname0 = …
              colname1 = …
              …
              mca_0/
                   data -> /1.1/instrument/mca_0/data
                   info -> /1.1/instrument/mca_0/
              …
          sample/
              ub_matrix = …
              unit_cell = …
              unit_cell_abc = …
              unit_cell_alphabetagamma = …
      2.1/
          …

Scans appear as *Groups* at the root level. The name of a scan group is
composed of two numbers, the first one being the *scan number* from the ``#S``
header line, and the second one being the *scan order*.
If a scan number appears multiple times in a SPEC file, the scan order is incremented by one.
For example, the scan *3.2* designates the second occurence of scan number 3 in a given file.

Data is stored in the ``measurement`` subgroup, one dataset per column. The dataset name
is the column label as it appears in the ``#L`` header line.

The ``instrument`` subgroup contains following subgroups:

    - ``specfile`` - contains two datasets, ``file_header`` and ``scan_header``,
      containing all header lines as a long string. Lines are delimited by the ``\n`` character.
    - ``positioners`` - contains one dataset per motor (positioner), including
      either the single motor position from the ``#P`` header line or a complete 1D array
      of positions if the motor names correspond to a data column (i.e. if the motor name
      from the ``#O`` header line is identical to a label in the ``#L`` header line)
    - one subgroup per MCA analyser/device containing a 2D ``data`` array with all spectra
      recorded by this analyser, as well as datasets for the various MCA metadata
      (``#@`` header lines). The first dimension of the ``data`` array corresponds to the number
      of points and the second one to the spectrum length.


In addition to the data columns, this group contains one subgroup per MCA analyser/device
with links to the data already comprised in  ``instrument/mca_...``

spech5 examples
+++++++++++++++

Accessing groups and datasets:

.. code-block:: python

    from silx.io.spech5 import SpecH5

    # Open a SpecFile
    sfh5 = SpecH5("test.dat")

    # using SpecH5 as a regular group to access scans
    scan1group = sfh5["1.1"]   # This retrieves scan 1.1
    scan1group = sfh5[0]       # This retrieves the first scan irrespectively of its number.
    instrument_group = scan1group["instrument"]

    # alternative: full path access
    measurement_group = sfh5["/1.1/measurement"]

    # accessing a scan data column by name as a 1D numpy array
    data_array = measurement_group["Pslit HGap"]

    # accessing all mca-spectra for one MCA device as a 2D array
    mca_0_spectra = measurement_group["mca_0/data"]


Files and groups can be treated as iterators, allowing looping through them.

.. code-block:: python

    # get all column names (labels) in all scans in a file
    for scan_group in SpecH5("test.dat"):
        dataset_names = [item.name in scan_group["measurement"] if not
                         item.name.startswith("mca")]
        print("Found labels in scan " + scan_group.name + " :")
        print(", ".join(dataset_names))


.. note::

    A :class:`SpecH5` object is also returned when one open a SPEC file
    through :meth:`silx.io.open`. See ":doc:`io`" for additional information.

Converting SPEC data to HDF5
++++++++++++++++++++++++++++

See :doc:`convert`.
