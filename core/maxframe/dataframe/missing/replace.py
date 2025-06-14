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

from typing import List

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import AnyField, Int32Field
from ...utils import no_default
from ..operators import SERIES_TYPE, DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, build_series


class DataFrameReplace(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.REPLACE

    to_replace = AnyField("to_replace", default=None)
    value = AnyField("value", default=None)
    limit = Int32Field("limit", default=None)
    regex = AnyField("regex", default=None)
    method = AnyField("method", default=no_default)

    @classmethod
    def _set_inputs(cls, op: "DataFrameReplace", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        input_iter = iter(inputs)
        next(input_iter)
        if isinstance(op.to_replace, SERIES_TYPE):
            op.to_replace = next(input_iter)
        if isinstance(op.value, SERIES_TYPE):
            op.value = next(input_iter)

    def __call__(self, df_or_series):
        inputs = [df_or_series]
        mock_obj = (
            build_df(df_or_series)
            if df_or_series.ndim == 2
            else build_series(df_or_series)
        )

        if isinstance(self.to_replace, SERIES_TYPE):
            mock_to_replace = build_series(self.to_replace)
            inputs.append(self.to_replace)
        else:
            mock_to_replace = self.to_replace

        if isinstance(self.value, SERIES_TYPE):
            mock_value = build_series(self.value)
            inputs.append(self.value)
        else:
            mock_value = self.value

        mock_result = mock_obj.replace(
            mock_to_replace, mock_value, regex=self.regex, method=self.method
        )

        if df_or_series.ndim == 2:
            return self.new_dataframe(
                inputs,
                shape=df_or_series.shape,
                dtypes=mock_result.dtypes,
                index_value=df_or_series.index_value,
                columns_value=df_or_series.columns_value,
            )
        else:
            return self.new_series(
                inputs,
                shape=df_or_series.shape,
                dtype=mock_result.dtype,
                index_value=df_or_series.index_value,
                name=df_or_series.name,
            )


_fun_doc = """
Replace values given in `to_replace` with `value`.

Values of the #obj_type# are replaced with other values dynamically.
This differs from updating with ``.loc`` or ``.iloc``, which require
you to specify a location to update with some value.

Parameters
----------
to_replace : str, regex, list, dict, Series, int, float, or None
    How to find the values that will be replaced.

    * numeric, str or regex:

        - numeric: numeric values equal to `to_replace` will be
          replaced with `value`
        - str: string exactly matching `to_replace` will be replaced
          with `value`
        - regex: regexs matching `to_replace` will be replaced with
          `value`

    * list of str, regex, or numeric:

        - First, if `to_replace` and `value` are both lists, they
          **must** be the same length.
        - Second, if ``regex=True`` then all of the strings in **both**
          lists will be interpreted as regexs otherwise they will match
          directly. This doesn't matter much for `value` since there
          are only a few possible substitution regexes you can use.
        - str, regex and numeric rules apply as above.

    * dict:

        - Dicts can be used to specify different replacement values
          for different existing values. For example,
          ``{'a': 'b', 'y': 'z'}`` replaces the value 'a' with 'b' and
          'y' with 'z'. To use a dict in this way the `value`
          parameter should be `None`.
        - For a DataFrame a dict can specify that different values
          should be replaced in different columns. For example,
          ``{'a': 1, 'b': 'z'}`` looks for the value 1 in column 'a'
          and the value 'z' in column 'b' and replaces these values
          with whatever is specified in `value`. The `value` parameter
          should not be ``None`` in this case. You can treat this as a
          special case of passing two lists except that you are
          specifying the column to search in.
        - For a DataFrame nested dictionaries, e.g.,
          ``{'a': {'b': np.nan}}``, are read as follows: look in column
          'a' for the value 'b' and replace it with NaN. The `value`
          parameter should be ``None`` to use a nested dict in this
          way. You can nest regular expressions as well. Note that
          column names (the top-level dictionary keys in a nested
          dictionary) **cannot** be regular expressions.

    * None:

        - This means that the `regex` argument must be a string,
          compiled regular expression, or list, dict, ndarray or
          Series of such elements. If `value` is also ``None`` then
          this **must** be a nested dictionary or Series.

    See the examples section for examples of each of these.
value : scalar, dict, list, str, regex, default None
    Value to replace any values matching `to_replace` with.
    For a DataFrame a dict of values can be used to specify which
    value to use for each column (columns not in the dict will not be
    filled). Regular expressions, strings and lists or dicts of such
    objects are also allowed.
inplace : bool, default False
    If True, in place. Note: this will modify any
    other views on this object (e.g. a column from a DataFrame).
    Returns the caller if this is True.
limit : int, default None
    Maximum size gap to forward or backward fill.
regex : bool or same types as `to_replace`, default False
    Whether to interpret `to_replace` and/or `value` as regular
    expressions. If this is ``True`` then `to_replace` *must* be a
    string. Alternatively, this could be a regular expression or a
    list, dict, or array of regular expressions in which case
    `to_replace` must be ``None``.
method : {'pad', 'ffill', 'bfill', `None`}
    The method to use when for replacement, when `to_replace` is a
    scalar, list or tuple and `value` is ``None``.

Returns
-------
#obj_type#
    Object after replacement.

Raises
------
AssertionError
    * If `regex` is not a ``bool`` and `to_replace` is not
      ``None``.
TypeError
    * If `to_replace` is a ``dict`` and `value` is not a ``list``,
      ``dict``, ``ndarray``, or ``Series``
    * If `to_replace` is ``None`` and `regex` is not compilable
      into a regular expression or is a list, dict, ndarray, or
      Series.
    * When replacing multiple ``bool`` or ``datetime64`` objects and
      the arguments to `to_replace` does not match the type of the
      value being replaced
ValueError
    * If a ``list`` or an ``ndarray`` is passed to `to_replace` and
      `value` but they are not the same length.

See Also
--------
#obj_type#.fillna : Fill NA values.
#obj_type#.where : Replace values based on boolean condition.
Series.str.replace : Simple string replacement.

Notes
-----
* Regex substitution is performed under the hood with ``re.sub``. The
  rules for substitution for ``re.sub`` are the same.
* Regular expressions will only substitute on strings, meaning you
  cannot provide, for example, a regular expression matching floating
  point numbers and expect the columns in your frame that have a
  numeric dtype to be matched. However, if those floating point
  numbers *are* strings, then you can do this.
* This method has *a lot* of options. You are encouraged to experiment
  and play with this method to gain intuition about how it works.
* When dict is used as the `to_replace` value, it is like
  key(s) in the dict are the to_replace part and
  value(s) in the dict are the value parameter.

Examples
--------

**Scalar `to_replace` and `value`**

>>> import maxframe.tensor as mt
>>> import maxframe.dataframe as md
>>> s = md.Series([0, 1, 2, 3, 4])
>>> s.replace(0, 5).execute()
0    5
1    1
2    2
3    3
4    4
dtype: int64

>>> df = md.DataFrame({'A': [0, 1, 2, 3, 4],
...                    'B': [5, 6, 7, 8, 9],
...                    'C': ['a', 'b', 'c', 'd', 'e']})
>>> df.replace(0, 5).execute()
   A  B  C
0  5  5  a
1  1  6  b
2  2  7  c
3  3  8  d
4  4  9  e

**List-like `to_replace`**

>>> df.replace([0, 1, 2, 3], 4).execute()
   A  B  C
0  4  5  a
1  4  6  b
2  4  7  c
3  4  8  d
4  4  9  e

>>> df.replace([0, 1, 2, 3], [4, 3, 2, 1]).execute()
   A  B  C
0  4  5  a
1  3  6  b
2  2  7  c
3  1  8  d
4  4  9  e

>>> s.replace([1, 2], method='bfill').execute()
0    0
1    3
2    3
3    3
4    4
dtype: int64

**dict-like `to_replace`**

>>> df.replace({0: 10, 1: 100}).execute()
     A  B  C
0   10  5  a
1  100  6  b
2    2  7  c
3    3  8  d
4    4  9  e

>>> df.replace({'A': 0, 'B': 5}, 100).execute()
     A    B  C
0  100  100  a
1    1    6  b
2    2    7  c
3    3    8  d
4    4    9  e

>>> df.replace({'A': {0: 100, 4: 400}}).execute()
     A  B  C
0  100  5  a
1    1  6  b
2    2  7  c
3    3  8  d
4  400  9  e

**Regular expression `to_replace`**

>>> df = md.DataFrame({'A': ['bat', 'foo', 'bait'],
...                    'B': ['abc', 'bar', 'xyz']})
>>> df.replace(to_replace=r'^ba.$', value='new', regex=True).execute()
      A    B
0   new  abc
1   foo  new
2  bait  xyz

>>> df.replace({'A': r'^ba.$'}, {'A': 'new'}, regex=True).execute()
      A    B
0   new  abc
1   foo  bar
2  bait  xyz

>>> df.replace(regex=r'^ba.$', value='new').execute()
      A    B
0   new  abc
1   foo  new
2  bait  xyz

>>> df.replace(regex={r'^ba.$': 'new', 'foo': 'xyz'}).execute()
      A    B
0   new  abc
1   xyz  new
2  bait  xyz

>>> df.replace(regex=[r'^ba.$', 'foo'], value='new').execute()
      A    B
0   new  abc
1   new  new
2  bait  xyz

Note that when replacing multiple ``bool`` or ``datetime64`` objects,
the data types in the `to_replace` parameter must match the data
type of the value being replaced:

>>> df = md.DataFrame({'A': [True, False, True],
...                    'B': [False, True, False]})
>>> df.replace({'a string': 'new value', True: False})  # raises.execute()
Traceback (most recent call last):
    ....execute()
TypeError: Cannot compare types 'ndarray(dtype=bool)' and 'str'

This raises a ``TypeError`` because one of the ``dict`` keys is not of
the correct type for replacement.

Compare the behavior of ``s.replace({'a': None})`` and
``s.replace('a', None)`` to understand the peculiarities
of the `to_replace` parameter:

>>> s = md.Series([10, 'a', 'a', 'b', 'a'])

When one uses a dict as the `to_replace` value, it is like the
value(s) in the dict are equal to the `value` parameter.
``s.replace({'a': None})`` is equivalent to
``s.replace(to_replace={'a': None}, value=None, method=None)``:

>>> s.replace({'a': None}).execute()
0      10
1    None
2    None
3       b
4    None
dtype: object

When ``value=None`` and `to_replace` is a scalar, list or
tuple, `replace` uses the method parameter (default 'pad') to do the
replacement. So this is why the 'a' values are being replaced by 10
in rows 1 and 2 and 'b' in row 4 in this case.
The command ``s.replace('a', None)`` is actually equivalent to
``s.replace(to_replace='a', value=None, method='pad')``:

>>> s.replace('a', None).execute()
0    10
1    10
2    10
3     b
4     b
dtype: object
"""


def _replace(
    df_or_series,
    to_replace=None,
    value=None,
    inplace=False,
    limit=None,
    regex=False,
    method=no_default,
):
    if not isinstance(to_replace, dict) and value is no_default and limit is not None:
        raise NotImplementedError("fill with limit not supported when value is None")

    if not isinstance(regex, bool):
        to_replace = regex
        regex = True
    op = DataFrameReplace(
        to_replace=to_replace, value=value, limit=limit, regex=regex, method=method
    )
    ret = op(df_or_series)
    if inplace:
        df_or_series.data = ret.data
    else:
        return ret


def df_replace(
    df,
    to_replace=no_default,
    value=no_default,
    inplace=False,
    limit=None,
    regex=False,
    method=no_default,
):
    return _replace(
        df,
        to_replace=to_replace,
        value=value,
        inplace=inplace,
        limit=limit,
        regex=regex,
        method=method,
    )


def series_replace(
    series,
    to_replace=no_default,
    value=no_default,
    inplace=False,
    limit=None,
    regex=False,
    method=no_default,
):
    return _replace(
        series,
        to_replace=to_replace,
        value=value,
        inplace=inplace,
        limit=limit,
        regex=regex,
        method=method,
    )


df_replace.__doc__ = _fun_doc.replace("#obj_type#", "DataFrame")
series_replace.__doc__ = _fun_doc.replace("#obj_type#", "Series")
