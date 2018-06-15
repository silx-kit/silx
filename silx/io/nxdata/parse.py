# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This package provides a collection of functions to work with h5py-like
groups following the NeXus *NXdata* specification.

See http://download.nexusformat.org/sphinx/classes/base_classes/NXdata.html

The main class is :class:`NXdata`.
You can also fetch the default NXdata in a NXroot or a NXentry with function
:func:`get_default`.


Other public functions:

 - :func:`is_valid_nxdata`
 - :func:`is_NXroot_with_default_NXdata`
 - :func:`is_NXentry_with_default_NXdata`

"""

import numpy
from silx.io.utils import is_group, is_file, is_dataset

from ._utils import get_attr_as_unicode, INTERPDIM, nxdata_logger, \
    get_uncertainties_names, get_signal_name, \
    get_auxiliary_signals_names, validate_auxiliary_signals, validate_number_of_axes
from silx.third_party import six


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/04/2018"


class InvalidNXdataError(Exception):
    pass


class NXdata(object):
    """NXdata parser.

    .. note::

        Before attempting to access any attribute or property,
        you should check that :attr:`is_valid` is *True*.

    :param group: h5py-like group following the NeXus *NXdata* specification.
    :param boolean validate: Set this parameter to *False* to skip the initial
        validation. This option is provided for optimisation purposes, for cases
        where :meth:`silx.io.nxdata.is_valid_nxdata` has already been called
        prior to instantiating this :class:`NXdata`.
    """
    def __init__(self, group, validate=True):
        super(NXdata, self).__init__()

        self.group = group
        """h5py-like group object with @NX_class=NXdata.
        """

        self.issues = []
        """List of error messages for malformed NXdata."""

        if validate:
            self._validate()
        self.is_valid = not self.issues
        """Validity status for this NXdata.
        If False, all properties and attributes will be None.
        """

        self._is_scatter = None
        self._axes = None

        self.signal = None
        """Main signal dataset in this NXdata group.
        In case more than one signal is present in this group,
        the other ones can be found in :attr:`auxiliary_signals`.
        """

        self.signal_name = None
        """Signal long name, as specified in the @long_name attribute of the
        signal dataset. If not specified, the dataset name is used."""

        self.signal_ndim = None
        self.signal_is_0d = None
        self.signal_is_1d = None
        self.signal_is_2d = None
        self.signal_is_3d = None

        self.axes_names = None
        """List of axes names in a NXdata group.

        This attribute is similar to :attr:`axes_dataset_names` except that
        if an axis dataset has a "@long_name" attribute, it will be used
        instead of the dataset name.
        """

        if not self.is_valid:
            nxdata_logger.debug("%s", self.issues)
        else:
            self.signal = self.group[self.signal_dataset_name]
            self.signal_name = get_attr_as_unicode(self.signal, "long_name")

            if self.signal_name is None:
                self.signal_name = self.signal_dataset_name

            # ndim will be available in very recent h5py versions only
            self.signal_ndim = getattr(self.signal, "ndim",
                                       len(self.signal.shape))

            self.signal_is_0d = self.signal_ndim == 0
            self.signal_is_1d = self.signal_ndim == 1
            self.signal_is_2d = self.signal_ndim == 2
            self.signal_is_3d = self.signal_ndim == 3

            self.axes_names = []
            # check if axis dataset defines @long_name
            for i, dsname in enumerate(self.axes_dataset_names):
                if dsname is not None and "long_name" in self.group[dsname].attrs:
                    self.axes_names.append(get_attr_as_unicode(self.group[dsname], "long_name"))
                else:
                    self.axes_names.append(dsname)

            # excludes scatters
            self.signal_is_1d = self.signal_is_1d and len(self.axes) <= 1  # excludes n-D scatters

    def _validate(self):
        """Fill :attr:`issues` with error messages for each error found."""
        if not is_group(self.group):
            raise TypeError("group must be a h5py-like group")
        if get_attr_as_unicode(self.group, "NX_class") != "NXdata":
            self.issues.append("Group has no attribute @NX_class='NXdata'")

        signal_name = get_signal_name(self.group)
        if signal_name is None:
            self.issues.append("No @signal attribute on the NXdata group, "
                               "and no dataset with a @signal=1 attr found")
            # very difficult to do more consistency tests without signal
            return

        elif signal_name not in self.group or not is_dataset(self.group[signal_name]):
            self.issues.append("Cannot find signal dataset '%s'" % signal_name)
            return

        auxiliary_signals_names = get_auxiliary_signals_names(self.group)
        self.issues += validate_auxiliary_signals(self.group,
                                                  signal_name,
                                                  auxiliary_signals_names)

        if "axes" in self.group.attrs:
            axes_names = get_attr_as_unicode(self.group, "axes")
            if isinstance(axes_names, (six.text_type, six.binary_type)):
                axes_names = [axes_names]

            self.issues += validate_number_of_axes(self.group, signal_name,
                                                   num_axes=len(axes_names))

            # Test consistency of @uncertainties
            uncertainties_names = get_uncertainties_names(self.group, signal_name)
            if uncertainties_names is not None:
                if len(uncertainties_names) != len(axes_names):
                    self.issues.append("@uncertainties does not define the same " +
                                       "number of fields than @axes")

            # Test individual axes
            is_scatter = True  # true if all axes have the same size as the signal
            signal_size = 1
            for dim in self.group[signal_name].shape:
                signal_size *= dim
            polynomial_axes_names = []
            for i, axis_name in enumerate(axes_names):

                if axis_name == ".":
                    continue
                if axis_name not in self.group or not is_dataset(self.group[axis_name]):
                    self.issues.append("Could not find axis dataset '%s'" % axis_name)
                    continue

                axis_size = 1
                for dim in self.group[axis_name].shape:
                    axis_size *= dim

                if len(self.group[axis_name].shape) != 1:
                    # I don't know how to interpret n-D axes
                    self.issues.append("Axis %s is not 1D" % axis_name)
                    continue
                else:
                    # for a  1-d axis,
                    fg_idx = self.group[axis_name].attrs.get("first_good", 0)
                    lg_idx = self.group[axis_name].attrs.get("last_good", len(self.group[axis_name]) - 1)
                    axis_len = lg_idx + 1 - fg_idx

                if axis_len != signal_size:
                    if axis_len not in self.group[signal_name].shape + (1, 2):
                        self.issues.append(
                                "Axis %s number of elements does not " % axis_name +
                                "correspond to the length of any signal dimension,"
                                " it does not appear to be a constant or a linear calibration," +
                                " and this does not seem to be a scatter plot.")
                        continue
                    elif axis_len in (1, 2):
                        polynomial_axes_names.append(axis_name)
                    is_scatter = False
                else:
                    if not is_scatter:
                        self.issues.append(
                                "Axis %s number of elements is equal " % axis_name +
                                "to the length of the signal, but this does not seem" +
                                " to be a scatter (other axes have different sizes)")
                        continue

                # Test individual uncertainties
                errors_name = axis_name + "_errors"
                if errors_name not in self.group and uncertainties_names is not None:
                    errors_name = uncertainties_names[i]
                    if errors_name in self.group and axis_name not in polynomial_axes_names:
                        if self.group[errors_name].shape != self.group[axis_name].shape:
                            self.issues.append(
                                    "Errors '%s' does not have the same " % errors_name +
                                    "dimensions as axis '%s'." % axis_name)

        # test dimensions of errors associated with signal
        if "errors" in self.group and is_dataset(self.group["errors"]):
            if self.group["errors"].shape != self.group[signal_name].shape:
                self.issues.append(
                        "Dataset containing standard deviations must " +
                        "have the same dimensions as the signal.")

    @property
    def signal_dataset_name(self):
        """Name of the main signal dataset."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")
        signal_dataset_name = get_attr_as_unicode(self.group, "signal")
        if signal_dataset_name is None:
            # find a dataset with @signal == 1
            for dsname in self.group:
                signal_attr = self.group[dsname].attrs.get("signal")
                if signal_attr in [1, b"1", u"1"]:
                    # This is the main (default) signal
                    signal_dataset_name = dsname
                    break
        assert signal_dataset_name is not None
        return signal_dataset_name

    @property
    def auxiliary_signals_dataset_names(self):
        """Sorted list of names of the auxiliary signals datasets.

        These are the names provided by the *@auxiliary_signals* attribute
        on the NXdata group.

        In case the NXdata group does not specify a *@signal* attribute
        but has a dataset with an attribute *@signal=1*,
        we look for datasets with attributes *@signal=2, @signal=3...*
        (deprecated NXdata specification)."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")
        signal_dataset_name = get_attr_as_unicode(self.group, "signal")
        if signal_dataset_name is not None:
            auxiliary_signals_names = get_attr_as_unicode(self.group, "auxiliary_signals")
            if auxiliary_signals_names is not None:
                if not isinstance(auxiliary_signals_names,
                                  (tuple, list, numpy.ndarray)):
                    # tolerate a single string, but coerce into a list
                    return [auxiliary_signals_names]
                return list(auxiliary_signals_names)
            return []

        # try old spec, @signal=1 (2, 3...) on dataset
        numbered_names = []
        for dsname in self.group:
            if dsname == self.signal_dataset_name:
                # main signal, not auxiliary
                continue
            ds = self.group[dsname]
            signal_attr = ds.attrs.get("signal")
            if signal_attr is not None and not is_dataset(ds):
                nxdata_logger.warning("Item %s with @signal=%s is not a dataset (%s)",
                                      dsname, signal_attr, type(ds))
                continue
            if signal_attr is not None:
                try:
                    signal_number = int(signal_attr)
                except (ValueError, TypeError):
                    nxdata_logger.warning("Could not parse attr @signal=%s on "
                                          "dataset %s as an int",
                                          signal_attr, dsname)
                    continue
                numbered_names.append((signal_number, dsname))
        return [a[1] for a in sorted(numbered_names)]

    @property
    def auxiliary_signals_names(self):
        """List of names of the auxiliary signals.

        Similar to :attr:`auxiliary_signals_dataset_names`, but the @long_name
        is used when this attribute is present, instead of the dataset name.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        signal_names = []
        for asdn in self.auxiliary_signals_dataset_names:
            if "long_name" in self.group[asdn].attrs:
                signal_names.append(self.group[asdn].attrs["long_name"])
            else:
                signal_names.append(asdn)
        return signal_names

    @property
    def auxiliary_signals(self):
        """List of all auxiliary signal datasets."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        return [self.group[dsname] for dsname in self.auxiliary_signals_dataset_names]

    @property
    def interpretation(self):
        """*@interpretation* attribute associated with the *signal*
        dataset of the NXdata group. ``None`` if no interpretation
        attribute is present.

        The *interpretation* attribute provides information about the last
        dimensions of the signal. The allowed values are:

             - *"scalar"*: 0-D data to be plotted
             - *"spectrum"*: 1-D data to be plotted
             - *"image"*: 2-D data to be plotted
             - *"vertex"*: 3-D data to be plotted

        For example, a 3-D signal with interpretation *"spectrum"* should be
        considered to be a 2-D array of 1-D data. A 3-D signal with
        interpretation *"image"* should be interpreted as a 1-D array (a list)
        of 2-D images. An n-D array with interpretation *"image"* should be
        interpreted as an (n-2)-D array of images.

        A warning message is logged if the returned interpretation is not one
        of the allowed values, but no error is raised and the unknown
        interpretation is returned anyway.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        allowed_interpretations = [None, "scalar", "spectrum", "image",
                                   "rgba-image",  # "hsla-image", "cmyk-image"
                                   "vertex"]

        interpretation = get_attr_as_unicode(self.signal, "interpretation")
        if interpretation is None:
            interpretation = get_attr_as_unicode(self.group, "interpretation")

        if interpretation not in allowed_interpretations:
            nxdata_logger.warning("Interpretation %s is not valid." % interpretation +
                                  " Valid values: " + ", ".join(allowed_interpretations))
        return interpretation

    @property
    def axes(self):
        """List of the axes datasets.

        The list typically has as many elements as there are dimensions in the
        signal dataset, the exception being scatter plots which use a 1D
        signal and multiple 1D axes of the same size.

        If an axis dataset applies to several dimensions of the signal, it
        will be repeated in the list.

        If a dimension of the signal has no dimension scale, `None` is
        inserted in its position in the list.

        .. note::

            The *@axes* attribute should define as many entries as there
            are dimensions in the signal, to avoid  any ambiguity.
            If this is not the case, this implementation relies on the existence
            of an *@interpretation* (*spectrum* or *image*) attribute in the
            *signal* dataset.

        .. note::

            If an axis dataset defines attributes @first_good or @last_good,
            the output will be a numpy array resulting from slicing that
            axis (*axis[first_good:last_good + 1]*).

        :rtype: List[Dataset or 1D array or None]
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        if self._axes is not None:
            # use cache
            return self._axes
        axes = []
        for axis_name in self.axes_dataset_names:
            if axis_name is None:
                axes.append(None)
            else:
                axes.append(self.group[axis_name])

        # keep only good range of axis data
        for i, axis in enumerate(axes):
            if axis is None:
                continue
            if "first_good" not in axis.attrs and "last_good" not in axis.attrs:
                continue
            fg_idx = axis.attrs.get("first_good", 0)
            lg_idx = axis.attrs.get("last_good", len(axis) - 1)
            axes[i] = axis[fg_idx:lg_idx + 1]

        self._axes = axes
        return self._axes

    @property
    def axes_dataset_names(self):
        """List of axes dataset names.

        If an axis dataset applies to several dimensions of the signal, its
        name will be repeated in the list.

        If a dimension of the signal has no dimension scale (i.e. there is a
        "." in that position in the *@axes* array), `None` is inserted in the
        output list in its position.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        numbered_names = []     # used in case of @axis=0 (old spec)
        axes_dataset_names = get_attr_as_unicode(self.group, "axes")
        if axes_dataset_names is None:
            # try @axes on signal dataset (older NXdata specification)
            axes_dataset_names = get_attr_as_unicode(self.signal, "axes")
            if axes_dataset_names is not None:
                # we expect a comma separated string
                if hasattr(axes_dataset_names, "split"):
                    axes_dataset_names = axes_dataset_names.split(":")
            else:
                # try @axis on the individual datasets (oldest NXdata specification)
                for dsname in self.group:
                    if not is_dataset(self.group[dsname]):
                        continue
                    axis_attr = self.group[dsname].attrs.get("axis")
                    if axis_attr is not None:
                        try:
                            axis_num = int(axis_attr)
                        except (ValueError, TypeError):
                            nxdata_logger.warning("Could not interpret attr @axis as"
                                                  "int on dataset %s", dsname)
                            continue
                        numbered_names.append((axis_num, dsname))

        ndims = len(self.signal.shape)
        if axes_dataset_names is None:
            if numbered_names:
                axes_dataset_names = []
                numbers = [a[0] for a in numbered_names]
                names = [a[1] for a in numbered_names]
                for i in range(ndims):
                    if i in numbers:
                        axes_dataset_names.append(names[numbers.index(i)])
                    else:
                        axes_dataset_names.append(None)
                return axes_dataset_names
            else:
                return [None] * ndims

        if isinstance(axes_dataset_names, (six.text_type, six.binary_type)):
            axes_dataset_names = [axes_dataset_names]

        for i, axis_name in enumerate(axes_dataset_names):
            if hasattr(axis_name, "decode"):
                axis_name = axis_name.decode()
            if axis_name == ".":
                axes_dataset_names[i] = None

        if len(axes_dataset_names) != ndims:
            if self.is_scatter and ndims == 1:
                # case of a 1D signal with arbitrary number of axes
                return list(axes_dataset_names)
            if self.interpretation != "rgba-image":
                # @axes may only define 1 or 2 axes if @interpretation=spectrum/image.
                # Use the existing names for the last few dims, and prepend with Nones.
                assert len(axes_dataset_names) == INTERPDIM[self.interpretation]
                all_dimensions_names = [None] * (ndims - INTERPDIM[self.interpretation])
                for axis_name in axes_dataset_names:
                    all_dimensions_names.append(axis_name)
            else:
                # 2 axes applying to the first two dimensions.
                # The 3rd signal dimension is expected to contain 3(4) RGB(A) values.
                assert len(axes_dataset_names) == 2
                all_dimensions_names = [axn for axn in axes_dataset_names]
                all_dimensions_names.append(None)
            return all_dimensions_names

        return list(axes_dataset_names)

    @property
    def title(self):
        """Plot title. If not found, returns an empty string.

        This attribute does not appear in the NXdata specification, but it is
        implemented in *nexpy* as a dataset named "title" inside the NXdata
        group. This dataset is expected to contain text.

        Because the *nexpy* approach could cause a conflict if the signal
        dataset or an axis dataset happened to be called "title", we also
        support providing the title as an attribute of the NXdata group.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        title = self.group.get("title")
        data_dataset_names = [self.signal_name] + self.axes_dataset_names
        if (title is not None and is_dataset(title) and
                "title" not in data_dataset_names):
            return str(title[()])

        title = self.group.attrs.get("title")
        if title is None:
            return ""
        return str(title)

    def get_axis_errors(self, axis_name):
        """Return errors (uncertainties) associated with an axis.

        If the axis has attributes @first_good or @last_good, the output
        is trimmed accordingly (a numpy array will be returned rather than a
        dataset).

        :param str axis_name: Name of axis dataset. This dataset **must exist**.
        :return: Dataset with axis errors, or None
        :raise KeyError: if this group does not contain a dataset named axis_name
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        # ensure axis_name is decoded, before comparing it with decoded attributes
        if hasattr(axis_name, "decode"):
            axis_name = axis_name.decode("utf-8")
        if axis_name not in self.group:
            # tolerate axis_name given as @long_name
            for item in self.group:
                long_name = get_attr_as_unicode(self.group[item], "long_name")
                if long_name is not None and long_name == axis_name:
                    axis_name = item
                    break

        if axis_name not in self.group:
            raise KeyError("group does not contain a dataset named '%s'" % axis_name)

        len_axis = len(self.group[axis_name])

        fg_idx = self.group[axis_name].attrs.get("first_good", 0)
        lg_idx = self.group[axis_name].attrs.get("last_good", len_axis - 1)

        # case of axisname_errors dataset present
        errors_name = axis_name + "_errors"
        if errors_name in self.group and is_dataset(self.group[errors_name]):
            if fg_idx != 0 or lg_idx != (len_axis - 1):
                return self.group[errors_name][fg_idx:lg_idx + 1]
            else:
                return self.group[errors_name]
        # case of uncertainties dataset name provided in @uncertainties
        uncertainties_names = get_attr_as_unicode(self.group, "uncertainties")
        if uncertainties_names is None:
            uncertainties_names = get_attr_as_unicode(self.signal, "uncertainties")
        if isinstance(uncertainties_names, six.text_type):
            uncertainties_names = [uncertainties_names]
        if uncertainties_names is not None:
            # take the uncertainty with the same index as the axis in @axes
            axes_ds_names = get_attr_as_unicode(self.group, "axes")
            if axes_ds_names is None:
                axes_ds_names = get_attr_as_unicode(self.signal, "axes")
            if isinstance(axes_ds_names, six.text_type):
                axes_ds_names = [axes_ds_names]
            elif isinstance(axes_ds_names, numpy.ndarray):
                # transform numpy.ndarray into list
                axes_ds_names = list(axes_ds_names)
            assert isinstance(axes_ds_names, list)
            if hasattr(axes_ds_names[0], "decode"):
                axes_ds_names = [ax_name.decode("utf-8") for ax_name in axes_ds_names]
            if axis_name not in axes_ds_names:
                raise KeyError("group attr @axes does not mention a dataset " +
                               "named '%s'" % axis_name)
            errors = self.group[uncertainties_names[list(axes_ds_names).index(axis_name)]]
            if fg_idx == 0 and lg_idx == (len_axis - 1):
                return errors      # dataset
            else:
                return errors[fg_idx:lg_idx + 1]    # numpy array
        return None

    @property
    def errors(self):
        """Return errors (uncertainties) associated with the signal values.

        :return: Dataset with errors, or None
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        if "errors" not in self.group:
            return None
        return self.group["errors"]

    @property
    def is_scatter(self):
        """True if the signal is 1D and all the axes have the
        same size as the signal."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        if self._is_scatter is not None:
            return self._is_scatter
        if not self.signal_is_1d:
            self._is_scatter = False
        else:
            self._is_scatter = True
            sigsize = 1
            for dim in self.signal.shape:
                sigsize *= dim
            for axis in self.axes:
                if axis is None:
                    continue
                axis_size = 1
                for dim in axis.shape:
                    axis_size *= dim
                self._is_scatter = self._is_scatter and (axis_size == sigsize)
        return self._is_scatter

    @property
    def is_x_y_value_scatter(self):
        """True if this is a scatter with a signal and two axes."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        return self.is_scatter and len(self.axes) == 2

    # we currently have no widget capable of plotting 4D data
    @property
    def is_unsupported_scatter(self):
        """True if this is a scatter with a signal and more than 2 axes."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        return self.is_scatter and len(self.axes) > 2

    @property
    def is_curve(self):
        """This property is True if the signal is 1D or :attr:`interpretation` is
        *"spectrum"*, and there is at most one axis with a consistent length.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        if self.signal_is_0d or self.interpretation not in [None, "spectrum"]:
            return False
        # the axis, if any, must be of the same length as the last dimension
        # of the signal, or of length 2 (a + b *x scale)
        if self.axes[-1] is not None and len(self.axes[-1]) not in [
                self.signal.shape[-1], 2]:
            return False
        if self.interpretation is None:
            # We no longer test whether x values are monotonic
            # (in the past, in that case, we used to consider it a scatter)
            return self.signal_is_1d
        # everything looks good
        return True

    @property
    def is_image(self):
        """True if the signal is 2D, or 3D with last dimension of length 3 or 4
        and interpretation *rgba-image*, or >2D with interpretation *image*.
        The axes (if any) length must also be consistent with the signal shape.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        if self.interpretation in ["scalar", "spectrum", "scaler"]:
            return False
        if self.signal_is_0d or self.signal_is_1d:
            return False
        if not self.signal_is_2d and \
                self.interpretation not in ["image", "rgba-image"]:
            return False
        if self.signal_is_3d and self.interpretation == "rgba-image":
            if self.signal.shape[-1] not in [3, 4]:
                return False
            img_axes = self.axes[0:2]
            img_shape = self.signal.shape[0:2]
        else:
            img_axes = self.axes[-2:]
            img_shape = self.signal.shape[-2:]
        for i, axis in enumerate(img_axes):
            if axis is not None and len(axis) not in [img_shape[i], 2]:
                return False

        return True

    @property
    def is_stack(self):
        """True in the signal is at least 3D and interpretation is not
        "scalar", "spectrum", "image" or "rgba-image".
        The axes length must also be consistent with the last 3 dimensions
        of the signal.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        if self.signal_ndim < 3 or self.interpretation in [
                "scalar", "scaler", "spectrum", "image", "rgba-image"]:
            return False
        stack_shape = self.signal.shape[-3:]
        for i, axis in enumerate(self.axes[-3:]):
            if axis is not None and len(axis) not in [stack_shape[i], 2]:
                return False
        return True


def is_valid_nxdata(group):   # noqa
    """Check if a h5py group is a **valid** NX_data group.

    :param group: h5py-like group
    :return: True if this NXdata group is valid.
    :raise TypeError: if group is not a h5py group, a spech5 group,
        or a fabioh5 group
    """
    nxd = NXdata(group)
    return nxd.is_valid


def is_NXentry_with_default_NXdata(group, validate=True):
    """Return True if group is a valid NXentry defining a valid default
    NXdata.

    :param group: h5py-like object.
    :param bool validate: Set this to False if you are sure that the target group
        is valid NXdata (i.e. :func:`silx.io.nxdata.is_valid_nxdata(target_group)`
        returns True). Parameter provided for optimisation purposes."""
    if not is_group(group):
        return False

    if get_attr_as_unicode(group, "NX_class") != "NXentry":
        return False

    default_nxdata_name = group.attrs.get("default")
    if default_nxdata_name is None or default_nxdata_name not in group:
        return False

    default_nxdata_group = group.get(default_nxdata_name)

    if not is_group(default_nxdata_group):
        return False

    if not validate:
        return True
    else:
        return is_valid_nxdata(default_nxdata_group)


def is_NXroot_with_default_NXdata(group, validate=True):
    """Return True if group is a valid NXroot defining a default NXentry
    defining a valid default NXdata.

    :param group: h5py-like object.
    :param bool validate: Set this to False if you are sure that the target group
        is valid NXdata (i.e. :func:`silx.io.nxdata.is_valid_nxdata(target_group)`
        returns True). Parameter provided for optimisation purposes.
    """
    if not is_group(group):
        return False

    # A NXroot is supposed to be at the root of a data file, and @NX_class
    # is therefore optional. We accept groups that are not located at the root
    # if they have @NX_class=NXroot (use case: several nexus files archived
    # in a single HDF5 file)
    if get_attr_as_unicode(group, "NX_class") != "NXroot" and not is_file(group):
        return False

    default_nxentry_name = group.attrs.get("default")
    if default_nxentry_name is None or default_nxentry_name not in group:
        return False

    default_nxentry_group = group.get(default_nxentry_name)
    return is_NXentry_with_default_NXdata(default_nxentry_group,
                                          validate=validate)


def get_default(group, validate=True):
    """Return a :class:`NXdata` object corresponding to the default NXdata group
    in the group specified as parameter.

    This function can find the NXdata if the group is already a NXdata, or
    if it is a NXentry defining a default NXdata, or if it is a NXroot
    defining such a default valid NXentry.

    Return None if no valid NXdata could be found.

    :param group: h5py-like group following the Nexus specification
        (NXdata, NXentry or NXroot).
    :param bool validate: Set this to False if you are sure that group
        is valid NXdata (i.e. :func:`silx.io.nxdata.is_valid_nxdata(group)`
        returns True). Parameter provided for optimisation purposes.
    :return: :class:`NXdata` object or None
    :raise TypeError: if group is not a h5py-like group
    """
    if not is_group(group):
        raise TypeError("Provided parameter is not a h5py-like group")

    if is_NXroot_with_default_NXdata(group, validate=validate):
        default_entry = group[group.attrs["default"]]
        default_data = default_entry[default_entry.attrs["default"]]
    elif is_NXentry_with_default_NXdata(group, validate=validate):
        default_data = group[group.attrs["default"]]
    elif not validate or is_valid_nxdata(group):
        default_data = group
    else:
        return None

    return NXdata(default_data, validate=False)
