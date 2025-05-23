# /*##########################################################################
#
# Copyright (c) 2017-2022 European Synchrotron Radiation Facility
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
 - :func:`is_group_with_default_NXdata`

"""

import json
from typing import Any

import h5py
import numpy

from silx.io.utils import is_group, is_file, is_dataset, h5py_read_dataset

from ._utils import (
    Interpretation,
    get_attr_as_unicode,
    INTERPDIM,
    get_dataset_name,
    nxdata_logger,
    get_uncertainties_names,
    get_signal_name,
    get_auxiliary_signals_names,
    validate_auxiliary_signals,
    validate_number_of_axes,
)

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "24/03/2020"


class InvalidNXdataError(Exception):
    pass


class _SilxStyle:
    """NXdata@SILX_style parser.

    :param NXdata nxdata:
        NXdata description for which to extract silx_style information.
    """

    def __init__(self, nxdata):
        naxes = len(nxdata.axes)
        self._axes_scale_types = [None] * naxes
        self._signal_scale_type = None

        stylestr = get_attr_as_unicode(nxdata.group, "SILX_style")
        if stylestr is None:
            return

        try:
            style = json.loads(stylestr)
        except json.JSONDecodeError:
            nxdata_logger.error("Ignoring SILX_style, cannot parse: %s", stylestr)
            return

        if not isinstance(style, dict):
            nxdata_logger.error("Ignoring SILX_style, cannot parse: %s", stylestr)

        if "axes_scale_types" in style:
            axes_scale_types = style["axes_scale_types"]

            if isinstance(axes_scale_types, str):
                # Convert single argument to list
                axes_scale_types = [axes_scale_types]

            if not isinstance(axes_scale_types, list):
                nxdata_logger.error("Ignoring SILX_style:axes_scale_types, not a list")
            else:
                for scale_type in axes_scale_types:
                    if scale_type not in ("linear", "log"):
                        nxdata_logger.error(
                            "Ignoring SILX_style:axes_scale_types, invalid value: %s",
                            str(scale_type),
                        )
                        break
                else:  # All values are valid
                    if len(axes_scale_types) > naxes:
                        nxdata_logger.error(
                            "Clipping SILX_style:axes_scale_types, too many values"
                        )
                        axes_scale_types = axes_scale_types[:naxes]
                    elif len(axes_scale_types) < naxes:
                        # Extend axes_scale_types with None to match number of axes
                        axes_scale_types = [None] * (
                            naxes - len(axes_scale_types)
                        ) + axes_scale_types
                    self._axes_scale_types = tuple(axes_scale_types)

        if "signal_scale_type" in style:
            scale_type = style["signal_scale_type"]
            if scale_type not in ("linear", "log"):
                nxdata_logger.error(
                    "Ignoring SILX_style:signal_scale_type, invalid value: %s",
                    str(scale_type),
                )
            else:
                self._signal_scale_type = scale_type

    axes_scale_types = property(
        lambda self: self._axes_scale_types,
        doc="Tuple of NXdata axes scale types (None, 'linear' or 'log'). List[str]",
    )

    signal_scale_type = property(
        lambda self: self._signal_scale_type,
        doc="NXdata signal scale type (None, 'linear' or 'log'). str",
    )


class NXdata:
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

    def __init__(self, group: h5py.Group, validate: bool = True):
        super().__init__()
        self._plot_style = None

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
            self.signal_name = get_dataset_name(self.group, self.signal_dataset_name)

            # ndim will be available in very recent h5py versions only
            self.signal_ndim = getattr(self.signal, "ndim", len(self.signal.shape))

            self.signal_is_0d = self.signal_ndim == 0
            self.signal_is_1d = self.signal_ndim == 1
            self.signal_is_2d = self.signal_ndim == 2
            self.signal_is_3d = self.signal_ndim == 3

            self.axes_names = [
                get_dataset_name(self.group, dsname)
                for dsname in self.axes_dataset_names
            ]

            # excludes scatters
            self.signal_is_1d = (
                self.signal_is_1d and len(self.axes) <= 1
            )  # excludes n-D scatters

            self._plot_style = _SilxStyle(self)

    def _validate(self):
        """Fill :attr:`issues` with error messages for each error found."""
        if not is_group(self.group):
            raise TypeError("group must be a h5py-like group")
        if get_attr_as_unicode(self.group, "NX_class") != "NXdata":
            self.issues.append("Group has no attribute @NX_class='NXdata'")
            return

        signal_name = get_signal_name(self.group)
        if signal_name is None:
            self.issues.append(
                "No @signal attribute on the NXdata group, "
                "and no dataset with a @signal=1 attr found"
            )
            # very difficult to do more consistency tests without signal
            return

        elif signal_name not in self.group or not is_dataset(self.group[signal_name]):
            self.issues.append("Cannot find signal dataset '%s'" % signal_name)
            return

        auxiliary_signals_names = get_auxiliary_signals_names(self.group)
        self.issues += validate_auxiliary_signals(
            self.group, signal_name, auxiliary_signals_names
        )

        axes_names = get_attr_as_unicode(self.group, "axes")
        if axes_names is None:
            # try @axes on signal dataset (older NXdata specification)
            axes_names = get_attr_as_unicode(self.group[signal_name], "axes")
            if axes_names is not None:
                # we expect a comma separated string
                if hasattr(axes_names, "split"):
                    axes_names = axes_names.split(":")

        if isinstance(axes_names, (str, bytes)):
            axes_names = [axes_names]

        if axes_names:
            self.issues += validate_number_of_axes(
                self.group, signal_name, num_axes=len(axes_names)
            )

            # Test consistency of @uncertainties
            uncertainties_names = get_uncertainties_names(self.group, signal_name)
            if uncertainties_names is not None:
                if len(uncertainties_names) != len(axes_names):
                    if len(uncertainties_names) < len(axes_names):
                        # ignore the field to avoid index error in the axes loop
                        uncertainties_names = None
                        self.issues.append(
                            "@uncertainties does not define the same "
                            + "number of fields than @axes. Field ignored"
                        )
                    else:
                        self.issues.append(
                            "@uncertainties does not define the same "
                            + "number of fields than @axes"
                        )

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
                    lg_idx = self.group[axis_name].attrs.get(
                        "last_good", len(self.group[axis_name]) - 1
                    )
                    axis_len = lg_idx + 1 - fg_idx

                if axis_len != signal_size:
                    if axis_len not in self.group[signal_name].shape + (1, 2):
                        self.issues.append(
                            "Axis %s number of elements does not " % axis_name
                            + "correspond to the length of any signal dimension,"
                            " it does not appear to be a constant or a linear calibration,"
                            + " and this does not seem to be a scatter plot."
                        )
                        continue
                    elif axis_len in (1, 2):
                        polynomial_axes_names.append(axis_name)
                    is_scatter = False

                # Test individual uncertainties
                errors_name = axis_name + "_errors"
                if errors_name not in self.group and uncertainties_names is not None:
                    errors_name = uncertainties_names[i]
                    if (
                        errors_name in self.group
                        and axis_name not in polynomial_axes_names
                    ):
                        if self.group[errors_name].shape != self.group[axis_name].shape:
                            self.issues.append(
                                "Errors '%s' does not have the same " % errors_name
                                + "dimensions as axis '%s'." % axis_name
                            )

        # test dimensions of errors associated with signal

        signal_errors = signal_name + "_errors"
        if "errors" in self.group and is_dataset(self.group["errors"]):
            errors = "errors"
        elif signal_errors in self.group and is_dataset(self.group[signal_errors]):
            errors = signal_errors
        else:
            errors = None
        if errors:
            if self.group[errors].shape != self.group[signal_name].shape:
                # In principle just the same size should be enough but
                # NeXus documentation imposes to have the same shape
                self.issues.append(
                    "Dataset containing standard deviations must "
                    + "have the same dimensions as the signal."
                )

    @property
    def signal_dataset_name(self) -> str:
        """Name of the main signal dataset."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")
        signal_dataset_name = get_attr_as_unicode(self.group, "signal")
        if signal_dataset_name is None:
            # find a dataset with @signal == 1
            for dsname in self.group:
                signal_attr = self.group[dsname].attrs.get("signal")
                if signal_attr in [1, b"1", "1"]:
                    # This is the main (default) signal
                    signal_dataset_name = dsname
                    break
        assert signal_dataset_name is not None
        return signal_dataset_name

    @property
    def auxiliary_signals_dataset_names(self) -> list[str]:
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
            auxiliary_signals_names = get_attr_as_unicode(
                self.group, "auxiliary_signals"
            )
            if auxiliary_signals_names is not None:
                if not isinstance(
                    auxiliary_signals_names, (tuple, list, numpy.ndarray)
                ):
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
                nxdata_logger.warning(
                    "Item %s with @signal=%s is not a dataset (%s)",
                    dsname,
                    signal_attr,
                    type(ds),
                )
                continue
            if signal_attr is not None:
                try:
                    signal_number = int(signal_attr)
                except (ValueError, TypeError):
                    nxdata_logger.warning(
                        "Could not parse attr @signal=%s on " "dataset %s as an int",
                        signal_attr,
                        dsname,
                    )
                    continue
                numbered_names.append((signal_number, dsname))
        return [a[1] for a in sorted(numbered_names)]

    @property
    def auxiliary_signals_names(self) -> list[str]:
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
    def auxiliary_signals(self) -> list[h5py.Dataset]:
        """List of all auxiliary signal datasets."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        return [self.group[dsname] for dsname in self.auxiliary_signals_dataset_names]

    @property
    def interpretation(self) -> Interpretation:
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

        allowed_interpretations = [
            None,
            "scaler",  # TODO: Is this part of the spec?
            "scalar",
            "spectrum",
            "image",
            "rgba-image",  # "hsla-image", "cmyk-image"
            "vertex",
        ]

        interpretation = get_attr_as_unicode(self.signal, "interpretation")
        if interpretation is None:
            interpretation = get_attr_as_unicode(self.group, "interpretation")

        if interpretation not in allowed_interpretations:
            nxdata_logger.warning(
                "Interpretation %s is not valid." % interpretation
                + " Valid values: "
                + ", ".join(str(s) for s in allowed_interpretations)
            )
        return interpretation

    @property
    def axes(self) -> list[h5py.Dataset]:
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
            axes[i] = axis[fg_idx : lg_idx + 1]

        self._axes = axes
        return self._axes

    @property
    def axes_dataset_names(self) -> list[str | None]:
        """List of axes dataset names.

        If an axis dataset applies to several dimensions of the signal, its
        name will be repeated in the list.

        If a dimension of the signal has no dimension scale (i.e. there is a
        "." in that position in the *@axes* array), `None` is inserted in the
        output list in its position.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        numbered_names = []  # used in case of @axis=0 (old spec)
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
                            nxdata_logger.warning(
                                "Could not interpret attr @axis as" "int on dataset %s",
                                dsname,
                            )
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

        if isinstance(axes_dataset_names, (str, bytes)):
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
    def title(self) -> str:
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
        if (
            title is not None
            and is_dataset(title)
            and "title" not in data_dataset_names
        ):
            return str(h5py_read_dataset(title))

        title = self.group.attrs.get("title")
        if title is None:
            return ""
        return str(title)

    def get_axis_errors(self, axis_name: str) -> h5py.Dataset | numpy.ndarray | None:
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
                return self.group[errors_name][fg_idx : lg_idx + 1]
            else:
                return self.group[errors_name]
        # case of uncertainties dataset name provided in @uncertainties
        uncertainties_names = get_attr_as_unicode(self.group, "uncertainties")
        if isinstance(uncertainties_names, str):
            uncertainties_names = [uncertainties_names]
        if uncertainties_names is not None:
            # take the uncertainty with the same index as the axis in @axes
            axes_ds_names = get_attr_as_unicode(self.group, "axes")
            if axes_ds_names is None:
                axes_ds_names = get_attr_as_unicode(self.signal, "axes")
            if isinstance(axes_ds_names, str):
                axes_ds_names = [axes_ds_names]
            elif isinstance(axes_ds_names, numpy.ndarray):
                # transform numpy.ndarray into list
                axes_ds_names = list(axes_ds_names)
            assert isinstance(axes_ds_names, list)
            if hasattr(axes_ds_names[0], "decode"):
                axes_ds_names = [ax_name.decode("utf-8") for ax_name in axes_ds_names]
            if axis_name not in axes_ds_names:
                raise KeyError(
                    "group attr @axes does not mention a dataset "
                    + "named '%s'" % axis_name
                )
            errors = self.group[
                uncertainties_names[list(axes_ds_names).index(axis_name)]
            ]
            if fg_idx == 0 and lg_idx == (len_axis - 1):
                return errors  # dataset
            else:
                return errors[fg_idx : lg_idx + 1]  # numpy array
        return None

    @property
    def errors(self) -> h5py.Dataset | None:
        """Return errors (uncertainties) associated with the signal values.

        :return: Dataset with errors, or None
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        dataset_names = [
            # From NXData:
            "errors",
            # Not Nexus (VARIABLE_errors is only for axes), but supported anyway
            self.signal_dataset_name + "_errors",
        ]
        for name in dataset_names:
            entity = self.group.get(name)
            if entity is not None and is_dataset(entity):
                return entity

        return None

    @property
    def plot_style(self) -> _SilxStyle | None:
        """Information extracted from the optional SILX_style attribute

        :raises: InvalidNXdataError
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        return self._plot_style

    @property
    def is_scatter(self) -> bool:
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
    def is_x_y_value_scatter(self) -> bool:
        """True if this is a scatter with a signal and two axes."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        return self.is_scatter and len(self.axes) == 2

    # we currently have no widget capable of plotting 4D data
    @property
    def is_unsupported_scatter(self) -> bool:
        """True if this is a scatter with a signal and more than 2 axes."""
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        return self.is_scatter and len(self.axes) > 2

    @property
    def is_curve(self) -> bool:
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
            self.signal.shape[-1],
            2,
        ]:
            return False
        if self.interpretation is None:
            # We no longer test whether x values are monotonic
            # (in the past, in that case, we used to consider it a scatter)
            return self.signal_is_1d
        # everything looks good
        return True

    @property
    def is_image(self) -> bool:
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
        if not self.signal_is_2d and self.interpretation not in ["image", "rgba-image"]:
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
    def is_stack(self) -> bool:
        """True in the signal is at least 3D and interpretation is not
        "scalar", "spectrum", "image" or "rgba-image".
        The axes length must also be consistent with the last 3 dimensions
        of the signal.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        if self.signal_ndim < 3 or self.interpretation in [
            "scalar",
            "scaler",
            "spectrum",
            "image",
            "rgba-image",
        ]:
            return False
        stack_shape = self.signal.shape[-3:]
        for i, axis in enumerate(self.axes[-3:]):
            if axis is not None and len(axis) not in [stack_shape[i], 2]:
                return False
        return True

    @property
    def is_volume(self) -> bool:
        """True in the signal is exactly 3D and interpretation
            "scalar", or nothing.

        The axes length must also be consistent with the 3 dimensions
        of the signal.
        """
        if not self.is_valid:
            raise InvalidNXdataError("Unable to parse invalid NXdata")

        if self.signal_ndim != 3:
            return False
        if self.interpretation not in [None, "scalar", "scaler"]:
            # 'scaler' and 'scalar' for a three dimensional array indicate a scalar field in 3D
            return False
        volume_shape = self.signal.shape[-3:]
        for i, axis in enumerate(self.axes[-3:]):
            if axis is not None and len(axis) not in [volume_shape[i], 2]:
                return False
        return True


def is_valid_nxdata(group: h5py.Group) -> bool:  # noqa
    """Check if a h5py group is a **valid** NX_data group.

    :param group: h5py-like group
    :return: True if this NXdata group is valid.
    :raise TypeError: if group is not a h5py group, a spech5 group,
        or a fabioh5 group
    """
    nxd = NXdata(group)
    return nxd.is_valid


def is_group_with_default_NXdata(group: h5py.Group, validate: bool = True) -> bool:
    """Return True if group defines a valid default
    NXdata.

    .. note::

        See https://github.com/silx-kit/silx/issues/2215

    :param group: h5py-like object.
    :param bool validate: Set this to skip the NXdata validation, and only
        check the existence of the group.
        Parameter provided for optimisation purposes, to avoid double
        validation if the validation is already performed separately."""
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


def is_NXentry_with_default_NXdata(group: Any, validate: bool = True) -> bool:
    """Return True if group is a valid NXentry defining a valid default
    NXdata.

    :param group: h5py-like object.
    :param bool validate: Set this to skip the NXdata validation, and only
        check the existence of the group.
        Parameter provided for optimisation purposes, to avoid double
        validation if the validation is already performed separately."""
    if not is_group(group):
        return False

    if get_attr_as_unicode(group, "NX_class") != "NXentry":
        return False

    return is_group_with_default_NXdata(group, validate)


def is_NXroot_with_default_NXdata(group: Any, validate=True) -> bool:
    """Return True if group is a valid NXroot defining a default NXentry
    defining a valid default NXdata.

    .. note::

        A NXroot group cannot directly define a default NXdata. If a
        *@default* argument is present, it must point to a NXentry group.
        This NXentry must define a valid NXdata for this function to return
        True.

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
    return is_NXentry_with_default_NXdata(default_nxentry_group, validate=validate)


def _get_default(
    group: Any,
    validate: bool,
    traversed: list,
) -> NXdata | None:
    if not is_group(group):
        raise TypeError("Provided parameter is not a h5py-like group")

    if get_attr_as_unicode(group, "NX_class") == "NXdata":
        nxdata = NXdata(group, validate=validate)
        return nxdata if nxdata.is_valid else None

    default_name = get_attr_as_unicode(group, "default")
    if default_name is None:
        return None

    default_entity = group.get(default_name)
    if default_entity is None or default_entity in traversed:
        return None

    try:
        return _get_default(default_entity, validate, traversed + [default_entity])
    except TypeError:
        return None


def get_default(group: Any, validate: bool = True) -> NXdata | None:
    """Find the default :class:`NXdata` group in given group.

    `@default` attributes are recursively followed until finding a group with
    NX_class="NXdata".
    Return None if no valid NXdata group could be found.

    :param group: h5py-like group to look for @default NXdata.
        In cas it is a NXdata group, it is returned.
    :param validate: False to disable checking the returned NXdata group.
    :raise TypeError: if group is not a h5py-like group
    """
    return _get_default(group, validate, [])
