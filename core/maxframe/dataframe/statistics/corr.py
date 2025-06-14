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

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import AnyField, BoolField, Int32Field, KeyField
from ...tensor.utils import filter_inputs
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, parse_index, validate_axis


class DataFrameCorr(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.CORR

    other = KeyField("other", default=None)
    method = AnyField("method", default=None)
    min_periods = Int32Field("min_periods", default=None)
    axis = Int32Field("axis", default=None)
    drop = BoolField("drop", default=None)

    @classmethod
    def _set_inputs(cls, op: "DataFrameCorr", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)
        next(inputs_iter)
        if isinstance(op.other, ENTITY_TYPE):
            op.other = next(inputs_iter)

    def __call__(self, df_or_series):
        if isinstance(df_or_series, SERIES_TYPE):
            inputs = filter_inputs([df_or_series, self.other])
            return self.new_scalar(inputs, dtype=np.dtype(float))
        else:

            def _filter_numeric(obj):
                if not isinstance(obj, DATAFRAME_TYPE):
                    return obj
                num_dtypes = build_empty_df(obj.dtypes)._get_numeric_data().dtypes
                if len(num_dtypes) != len(obj.dtypes):
                    return obj[list(num_dtypes.index)]
                return obj

            df_or_series = _filter_numeric(df_or_series)
            self.other = _filter_numeric(self.other)

            inputs = filter_inputs([df_or_series, self.other])
            if self.axis is None:
                dtypes = pd.Series(
                    [np.dtype(float)] * len(df_or_series.dtypes),
                    index=df_or_series.dtypes.index,
                )
                return self.new_dataframe(
                    inputs,
                    shape=(df_or_series.shape[1],) * 2,
                    dtypes=dtypes,
                    index_value=df_or_series.columns_value,
                    columns_value=df_or_series.columns_value,
                )
            else:
                new_index_value = df_or_series.axes[1 - self.axis].index_value
                if isinstance(self.other, DATAFRAME_TYPE):
                    align_dtypes = pd.concat(
                        [self.other.dtypes, df_or_series.dtypes], axis=1
                    )
                    align_shape = (np.nan, align_dtypes.shape[0])
                    new_index_value = parse_index(align_dtypes.index)
                else:
                    align_shape = df_or_series.shape

                shape = (np.nan,) if self.drop else (align_shape[1 - self.axis],)
                return self.new_series(
                    inputs,
                    shape=shape,
                    dtype=np.dtype(float),
                    index_value=new_index_value,
                )


def df_corr(df, method="pearson", min_periods=1):
    """
    Compute pairwise correlation of columns, excluding NA/null values.

    Parameters
    ----------
    method : {'pearson', 'kendall', 'spearman'} or callable
        Method of correlation:

        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior.

        .. note::
            kendall, spearman and callables not supported on multiple chunks yet.

    min_periods : int, optional
        Minimum number of observations required per pair of columns
        to have a valid result. Currently only available for Pearson
        and Spearman correlation.

    Returns
    -------
    DataFrame
        Correlation matrix.

    See Also
    --------
    DataFrame.corrwith : Compute pairwise correlation with another
        DataFrame or Series.
    Series.corr : Compute the correlation between two Series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
    ...                   columns=['dogs', 'cats'])
    >>> df.corr(method='pearson').execute()
              dogs      cats
    dogs  1.000000 -0.851064
    cats -0.851064  1.000000
    """
    op = DataFrameCorr(method=method, min_periods=min_periods)
    return op(df)


def df_corrwith(df, other, axis=0, drop=False, method="pearson"):
    """
    Compute pairwise correlation.

    Pairwise correlation is computed between rows or columns of
    DataFrame with rows or columns of Series or DataFrame. DataFrames
    are first aligned along both axes before computing the
    correlations.

    Parameters
    ----------
    other : DataFrame, Series
        Object with which to compute correlations.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. 0 or 'index' to compute column-wise, 1 or 'columns' for
        row-wise.
    drop : bool, default False
        Drop missing indices from result.
    method : {'pearson', 'kendall', 'spearman'} or callable
        Method of correlation:

        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float.

        .. note::
            kendall, spearman and callables not supported on multiple chunks yet.

    Returns
    -------
    Series
        Pairwise correlations.

    See Also
    --------
    DataFrame.corr : Compute pairwise correlation of columns.
    """
    axis = validate_axis(axis, df)
    if drop:
        # TODO implement with df.align(method='inner')
        raise NotImplementedError("drop=True not implemented")
    op = DataFrameCorr(other=other, method=method, axis=axis, drop=drop)
    return op(df)


def series_corr(series, other, method="pearson", min_periods=None):
    """
    Compute correlation with `other` Series, excluding missing values.

    Parameters
    ----------
    other : Series
        Series with which to compute the correlation.
    method : {'pearson', 'kendall', 'spearman'} or callable
        Method used to compute correlation:

        - pearson : Standard correlation coefficient
        - kendall : Kendall Tau correlation coefficient
        - spearman : Spearman rank correlation
        - callable: Callable with input two 1d ndarrays and returning a float.

        .. note::
            kendall, spearman and callables not supported on multiple chunks yet.

    min_periods : int, optional
        Minimum number of observations needed to have a valid result.

    Returns
    -------
    float
        Correlation with other.

    See Also
    --------
    DataFrame.corr : Compute pairwise correlation between columns.
    DataFrame.corrwith : Compute pairwise correlation with another
        DataFrame or Series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s1 = md.Series([.2, .0, .6, .2])
    >>> s2 = md.Series([.3, .6, .0, .1])
    >>> s1.corr(s2, method='pearson').execute()
    -0.8510644963469898
    """
    op = DataFrameCorr(other=other, method=method, min_periods=min_periods)
    return op(series)


def series_autocorr(series, lag=1):
    """
    Compute the lag-N autocorrelation.

    This method computes the Pearson correlation between
    the Series and its shifted self.

    Parameters
    ----------
    lag : int, default 1
        Number of lags to apply before performing autocorrelation.

    Returns
    -------
    float
        The Pearson correlation between self and self.shift(lag).

    See Also
    --------
    Series.corr : Compute the correlation between two Series.
    Series.shift : Shift index by desired number of periods.
    DataFrame.corr : Compute pairwise correlation of columns.
    DataFrame.corrwith : Compute pairwise correlation between rows or
        columns of two DataFrame objects.

    Notes
    -----
    If the Pearson correlation is not well defined return 'NaN'.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series([0.25, 0.5, 0.2, -0.05])
    >>> s.autocorr().execute()  # doctest: +ELLIPSIS.execute()
    0.10355...
    >>> s.autocorr(lag=2).execute()  # doctest: +ELLIPSIS.execute()
    -0.99999...

    If the Pearson correlation is not well defined, then 'NaN' is returned.

    >>> s = md.Series([1, 0, 0, 0])
    >>> s.autocorr().execute()
    nan
    """
    op = DataFrameCorr(other=series.shift(lag), method="pearson")
    return op(series)
