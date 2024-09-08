# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from typing import Callable

import numpy as np
import pandas as pd

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.dataframe.core import DataFrame, IndexValue
from maxframe.dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from maxframe.dataframe.utils import make_dtypes, parse_index
from maxframe.serialization.serializables import (
    BoolField,
    DictField,
    FunctionField,
    TupleField,
)


class DataFrameFlatMapOperator(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.FLATMAP

    func = FunctionField("func")
    raw = BoolField("raw", default=False)
    args = TupleField("args", default=())
    kwargs = DictField("kwargs", default={})

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @staticmethod
    def _gen_flattening_index_value(index_value, *args) -> IndexValue:
        pd_index = index_value.to_pandas()
        if not isinstance(pd_index, pd.MultiIndex):
            # for func return multi rows, will copy indexes
            return parse_index(pd.Index([], dtype=pd_index.dtype), *args)
        # multi index will keep the same level and types
        return parse_index(
            pd.MultiIndex.from_arrays([c[:0] for c in pd_index.levels]), *args
        )

    def _call_dataframe(self, df: DataFrame, dtypes: pd.Series):
        dtypes = make_dtypes(dtypes)
        index_value = self._gen_flattening_index_value(
            df.index_value,
            (df.key, df.index_value.key, self.func),
        )
        return self.new_dataframe(
            [df],
            shape=(np.nan, len(dtypes)),
            index_value=index_value,
            columns_value=parse_index(dtypes.index, store_data=True),
            dtypes=dtypes,
        )

    def _call_series_or_index(self, series, dtypes=None):
        index_value = self._gen_flattening_index_value(
            series.index_value,
            (series.key, series.index_value.key, self.func),
        )

        if self.output_types[0] == OutputType.series:
            name, dtype = dtypes
            return self.new_series(
                [series],
                dtype=dtype,
                shape=(np.nan,),
                index_value=index_value,
                name=name,
            )

        dtypes = make_dtypes(dtypes)
        columns_value = parse_index(dtypes.index, store_data=True)
        return self.new_dataframe(
            [series],
            shape=(np.nan, len(dtypes)),
            index_value=index_value,
            columns_value=columns_value,
            dtypes=dtypes,
        )

    def __call__(
        self,
        df_or_series,
        dtypes=None,
        output_type=None,
    ):
        if df_or_series.op.output_types[0] == OutputType.dataframe:
            return self._call_dataframe(df_or_series, dtypes=dtypes)
        else:
            return self._call_series_or_index(df_or_series, dtypes=dtypes)


def df_flatmap(dataframe, func: Callable, dtypes=None, raw=False, args=(), **kwargs):
    """
    Apply the given function to each row and then flatten results. Use this method if your transformation returns
    multiple rows for each input row.

    This function applies a transformation to each row of the DataFrame, where the transformation can return zero
    or multiple values, effectively flattening Python generators, list-like collections, and DataFrames.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to which the function will be applied.

    func : Callable
        Function to apply to each row of the DataFrame. It should accept a Series (or an array if `raw=True`)
        representing a row and return a list or iterable of values.

    dtypes : Series, dict or list
        Specify dtypes of returned DataFrame.

    raw : bool, default False
        Determines if the row is passed as a Series or as a numpy array:

        * ``False`` : passes each row as a Series to the function.
        * ``True`` : the passed function will receive numpy array objects instead.

    args : tuple
        Positional arguments to pass to `func`.

    **kwargs
        Additional keyword arguments to pass as keywords arguments to `func`.

    Returns
    -------
    DataFrame
        Return DataFrame with specified `dtypes`.

    Notes
    -----
    The `func` must return an iterable of values for each input row. The index of the resulting DataFrame will be
     repeated based on the number of output rows generated by `func`.

    Examples
    --------
    >>> import numpy as np
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df.execute()
       A  B
    0  1  4
    1  2  5
    2  3  6

    Define a function that takes a number and returns a list of two numbers:

    >>> def generate_values_array(row):
    ...     return [row['A'] * 2, row['B'] * 3]

    Define a function that takes a row and return two rows and two columns:

    >>> def generate_values_in_generator(row):
    ...     yield [row[0] * 2, row[1] * 4]
    ...     yield [row[0] * 3, row[1] * 5]

    Which equals to the following function return a dataframe:

    >>> def generate_values_in_dataframe(row):
    ...     return pd.DataFrame([[row[0] * 2, row[1] * 4], [row[0] * 3, row[1] * 5]])

    Specify `dtypes` with a function which returns a DataFrame:

    >>> df.mf.flatmap(generate_values_array, dtypes=pd.Series({'A': 'int'})).execute()
            A
        0   2
        0  12
        1   4
        1  15
        2   6
        2  18

    Specify raw=True to pass input row as array:

    >>> df.mf.flatmap(generate_values_in_generator, dtypes={"A": "int", "B": "int"}, raw=True).execute()
           A   B
        0  2  16
        0  3  20
        1  4  20
        1  6  25
        2  6  24
        2  9  30
    """
    if dtypes is None or len(dtypes) == 0:
        raise TypeError(
            "Cannot determine {dtypes} by calculating with enumerate data, "
            "please specify it as arguments"
        )

    if not isinstance(func, Callable):
        raise TypeError("function must be a callable object")

    output_types = [OutputType.dataframe]
    op = DataFrameFlatMapOperator(
        func=func, raw=raw, output_types=output_types, args=args, kwargs=kwargs
    )
    return op(
        dataframe,
        dtypes=dtypes,
    )


def series_flatmap(
    series, func: Callable, dtypes=None, dtype=None, name=None, args=(), **kwargs
):
    """
    Apply the given function to each row and then flatten results. Use this method if your transformation returns
    multiple rows for each input row.

    This function applies a transformation to each element of the Series, where the transformation can return zero
     or multiple values, effectively flattening Python generator, list-liked collections and DataFrame.

    Parameters
    ----------
    series : Series
        The series to which the function will be applied.

    func : Callable
        Function to apply to each element of the Series. It should accept a scalar value
        (or an array if `raw=True`) and return a list or iterable of values.

    dtypes : Series, default None
        Specify dtypes of returned DataFrame. Can't work with dtype.

    dtype : numpy.dtype, default None
        Specify dtype of returned Series. Can't work with dtypes.

    name : str, default None
        Specify name of the returned Series.

    args : tuple
        Positional arguments to pass to `func`.

    **kwargs
        Additional keyword arguments to pass as keywords arguments to `func`.

    Returns
    -------
    DataFrame or Series
        Result of DataFrame when dtypes specified, else Series.

    Notes
    -----
    The `func` must return an iterable of values for each input element. If `dtypes` is specified,
    `flatmap` will return a DataFrame, if `dtype` and `name` is specified, a Series will be returned. The index of
     the resulting DataFrame/Series will be repeated based on the number of output rows generated by `func`.

    Examples
    --------
    >>> import numpy as np
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df.execute()
       A  B
    0  1  4
    1  2  5
    2  3  6

    Define a function that takes a number and returns a list of two numbers:

    >>> def generate_values_array(x):
    ...     return [x * 2, x * 3]

    >>> def generate_values_in_generator(x):
    ...     yield pd.Series([x * 2, x * 4])
    ...     yield pd.Series([x * 3, x * 5])

    Specify `dtype` with a function which returns list to return more than one elements as a Series:

    >>> df['A'].mf.flatmap(generate_values_array, dtype="int", name="C").execute()
        0    2
        0    3
        1    4
        1    6
        2    6
        2    9
        Name: C, dtype: int64

    Specify `dtypes` to return multi columns as a DataFrame:

    >>> df['A'].mf.flatmap(generate_values_in_generator, dtypes={"A": "int", "B": "int"}).execute()
           A   B
        0  2   4
        0  3   5
        1  4   8
        1  6  10
        2  6  12
        2  9  15
    """

    if dtypes and dtype:
        raise ValueError("Both dtypes and dtype cannot be specified at the same time.")

    dtypes = (name, dtype) if dtype is not None else dtypes
    if dtypes is None:
        raise TypeError(
            "Cannot determine {dtypes} or {dtype} by calculating with enumerate data, "
            "please specify it as arguments"
        )

    if not isinstance(func, Callable):
        raise TypeError("function must be a callable object")

    output_type = OutputType.series if dtype is not None else OutputType.dataframe

    op = DataFrameFlatMapOperator(
        func=func, raw=False, output_types=[output_type], args=args, kwargs=kwargs
    )
    return op(
        series,
        dtypes=dtypes,
    )
