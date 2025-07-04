# Copyright 1999-2025 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple

import numpy as np

from maxframe import tensor as mt

from ...udf import builtin_function
from .core import is_scalar_nan


@builtin_function
def _cut_before(c, idx):
    return c[: idx + 1]


@builtin_function
def _map_inverse_encodes(c, idx):
    if c.flags.writeable:
        c[c > idx] = idx
    else:  # pragma: no cover
        # If c is got from the shared memory, it is immutable.
        c = np.select([c <= idx], [c], idx)
    return c


def _unique(values, *, return_inverse=False):
    """Helper function to find unique values with support for python objects.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : ndarray
        Values to check for unknowns.

    return_inverse : bool, default=False
        If True, also return the indices of the unique values.

    Returns
    -------
    unique : ndarray
        The sorted unique values.

    unique_inverse : ndarray
        The indices to reconstruct the original array from the unique array.
        Only provided if `return_inverse` is True.
    """
    if values.dtype == object:
        # FIXME workaround for LabelEncoder
        #  remove if-block below when Series.unique implemented
        # series_unique = md.Series(md.Series(values).unique()).sort_values().values
        # if return_inverse:
        #     return series_unique, _map_to_integer(values, series_unique)
        # else:
        #     return series_unique
        return mt.unique(values, return_inverse=return_inverse)

    out = mt.unique(values, return_inverse=return_inverse)

    if return_inverse:
        uniques, inverse = out
    else:
        uniques = out

    # np.unique will have duplicate missing values at the end of `uniques`
    # here we clip the nans and remove it from uniques
    uniques = uniques.rechunk(tuple((s,) for s in uniques.shape))
    nan_idx = mt.searchsorted(uniques, mt.nan)
    uniques = uniques.mf.apply_chunk(
        _cut_before,
        args=(nan_idx,),
        dtype=uniques.dtype,
        shape=(np.nan,) * uniques.ndim,
    )
    if return_inverse:
        inverse = inverse.mf.apply_chunk(
            _map_inverse_encodes,
            args=(nan_idx,),
            dtype=inverse.dtype,
            shape=((np.nan,),) * inverse.ndim,
        )

        return uniques, inverse
    return uniques


class _nandict(dict):  # pragma: no cover
    """Dictionary with support for nans."""

    def __init__(self, mapping):
        super().__init__(mapping)
        for key, value in mapping.items():
            if is_scalar_nan(key):
                self.nan_value = value
                break

    def __missing__(self, key):
        if hasattr(self, "nan_value") and is_scalar_nan(key):
            return self.nan_value
        raise KeyError(key)


class MissingValues(NamedTuple):  # pragma: no cover
    """Data class for missing data information"""

    nan: bool
    none: bool

    def to_list(self):
        """Convert tuple to a list where None is always first."""
        output = []
        if self.none:
            output.append(None)
        if self.nan:
            output.append(np.nan)
        return output


def _extract_missing(values):  # pragma: no cover
    """Extract missing values from `values`.

    Parameters
    ----------
    values: set
        Set of values to extract missing from.

    Returns
    -------
    output: set
        Set with missing values extracted.

    missing_values: MissingValues
        Object with missing value information.
    """
    missing_values_set = {
        value for value in values if value is None or is_scalar_nan(value)
    }

    if not missing_values_set:
        return values, MissingValues(nan=False, none=False)

    if None in missing_values_set:
        if len(missing_values_set) == 1:
            output_missing_values = MissingValues(nan=False, none=True)
        else:
            # If there is more than one missing value, then it has to be
            # float('nan') or np.nan
            output_missing_values = MissingValues(nan=True, none=True)
    else:
        output_missing_values = MissingValues(nan=True, none=False)

    # create set without the missing values
    output = values - missing_values_set
    return output, output_missing_values


@builtin_function
def _value_to_int_mapper(values_data, uniques_data, check_unknown=True):
    if values_data.dtype.kind in "OUS":
        try:
            table = _nandict({val: i for i, val in enumerate(uniques_data)})
            return np.array([table[v] for v in values_data])
        except KeyError as e:
            raise ValueError(f"y contains previously unseen labels: {str(e)}")
    else:
        if check_unknown:
            diff = _check_unknown(values_data, uniques_data)
            if diff:
                raise ValueError(f"y contains previously unseen labels: {str(diff)}")
        return np.searchsorted(uniques_data, values_data)


def _map_to_integer(values, uniques, check_unknown=True):
    """Map values based on its position in uniques."""

    return values.mf.apply_chunk(
        _value_to_int_mapper,
        args=(uniques,),
        dtype=np.dtype(np.int64),
        shape=values.shape,
        check_unknown=check_unknown,
    )


def _check_unknown(values, known_values, return_mask=False):  # pragma: no cover
    """
    Helper function to check for unknowns in values to be encoded.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : array
        Values to check for unknowns.
    known_values : array
        Known values. Must be unique.
    return_mask : bool, default=False
        If True, return a mask of the same shape as `values` indicating
        the valid values.

    Returns
    -------
    diff : list
        The unique values present in `values` and not in `know_values`.
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.

    """
    valid_mask = None

    if values.dtype.kind in "OUS":
        values_set = set(values)
        values_set, missing_in_values = _extract_missing(values_set)

        uniques_set = set(known_values)
        uniques_set, missing_in_uniques = _extract_missing(uniques_set)
        diff = values_set - uniques_set

        nan_in_diff = missing_in_values.nan and not missing_in_uniques.nan
        none_in_diff = missing_in_values.none and not missing_in_uniques.none

        def is_valid(value):
            return (
                value in uniques_set
                or missing_in_uniques.none
                and value is None
                or missing_in_uniques.nan
                and is_scalar_nan(value)
            )

        if return_mask:
            if diff or nan_in_diff or none_in_diff:
                valid_mask = np.array([is_valid(value) for value in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)

        diff = list(diff)
        if none_in_diff:
            diff.append(None)
        if nan_in_diff:
            diff.append(np.nan)
    else:
        unique_values = np.unique(values)
        diff = np.setdiff1d(unique_values, known_values, assume_unique=True)
        if return_mask:
            if diff.size:
                valid_mask = np.in1d(values, known_values)
            else:
                valid_mask = np.ones(len(values), dtype=bool)

        # check for nans in the known_values
        if np.isnan(known_values).any():
            diff_is_nan = np.isnan(diff)
            if diff_is_nan.any():
                # removes nan from valid_mask
                if diff.size and return_mask:
                    is_nan = np.isnan(values)
                    valid_mask[is_nan] = 1

                # remove nan from diff
                diff = diff[~diff_is_nan]
        diff = list(diff)

    if return_mask:
        return diff, valid_mask
    return diff


def _encode(values, *, uniques, check_unknown=True):
    """Helper function to encode values into [0, n_uniques - 1].

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    the case. The calling method needs to ensure this for all non-object
    values.

    Parameters
    ----------
    values : tensor
        Values to encode.
    uniques : tensor
        The unique values in `values`. If the dtype is not object, then
        `uniques` needs to be sorted.
    check_unknown : bool, default=True
        If True, check for values in `values` that are not in `unique`
        and raise an error. This is ignored for object dtype, and treated as
        True in this case. This parameter is useful for
        _BaseEncoder._transform() to avoid calling _check_unknown()
        twice.

    Returns
    -------
    encoded : tensor
        Encoded values
    """
    return _map_to_integer(values, uniques, check_unknown=check_unknown)
