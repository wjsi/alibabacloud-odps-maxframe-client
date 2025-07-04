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

from numbers import Integral
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import find_common_type
from pandas.core.indexing import IndexingError

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, OutputType
from ...serialization.serializables import AnyField, KeyField, ListField
from ...tensor.datasource import asarray
from ...tensor.utils import calc_sliced_size, filter_inputs
from ...utils import is_full_slice, lazy_import, pd_release_version
from ..core import DATAFRAME_TYPE, IndexValue
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index
from .iloc import DataFrameIlocSetItem

cudf = lazy_import("cudf")
with_slice_locs_kind = pd_release_version < (1, 4, 0)


def process_loc_indexes(inp, indexes, fetch_index: bool = True):
    ndim = inp.ndim

    if not isinstance(indexes, tuple):
        indexes = (indexes,)
    if len(indexes) < ndim:
        indexes += (slice(None),) * (ndim - len(indexes))
    if len(indexes) > ndim:
        raise IndexingError("Too many indexers")

    new_indexes = []
    for ax, index in enumerate(indexes):
        if isinstance(index, (list, np.ndarray, pd.Series, ENTITY_TYPE)):
            if not isinstance(index, ENTITY_TYPE):
                index = np.asarray(index)
            elif fetch_index:
                index = asarray(index)
                if ax == 1:
                    # do not support tensor index on axis 1
                    # because if so, the dtypes and columns_value would be unknown
                    try:
                        index = index.fetch()
                    except (RuntimeError, ValueError):
                        raise NotImplementedError(
                            "indexer on axis columns cannot be non-executed tensor"
                        )
        new_indexes.append(index)

    return new_indexes


class DataFrameLoc:
    def __init__(self, obj):
        self._obj = obj

    def _use_iloc(self, indexes):
        # for RangeIndex from 0, use iloc instead of loc
        index_value = self._obj.index_value.value
        if len(indexes) == 2:
            if not isinstance(indexes[1], slice):
                return False, None
            elif indexes[1] != slice(None):
                return False, None
        if not isinstance(index_value, IndexValue.RangeIndex):
            return False, None
        if index_value.slice.start != 0 and index_value.slice.start is not None:
            return False, None
        if not isinstance(indexes[0], (Integral, slice)):
            return False, None
        if isinstance(indexes[0], Integral):
            if indexes[0] < 0:
                return False, None
        else:
            index0 = indexes[0]
            for v in (index0.start, index0.stop, index0.step):
                if v is None:
                    continue
                if not isinstance(v, Integral):
                    return False, None
                if v < 0:
                    return False, None
            if index0.stop is not None:
                # adjust slice right bound
                return (
                    True,
                    [slice(index0.start, index0.stop + 1, index0.step)] + indexes[1:],
                )
        return True, None

    def __getitem__(self, indexes):
        indexes = process_loc_indexes(self._obj, indexes)

        use_iloc, new_indexes = self._use_iloc(indexes)
        if use_iloc:
            # use iloc instead
            return self._obj.iloc[tuple(new_indexes or indexes)]

        op = DataFrameLocGetItem(indexes=indexes)
        return op(self._obj)

    def __setitem__(self, indexes, value):
        if not np.isscalar(value):
            raise NotImplementedError("Only scalar value is supported to set by loc")
        if not isinstance(self._obj, DATAFRAME_TYPE):
            raise NotImplementedError("Only DataFrame is supported to set by loc")
        indexes = process_loc_indexes(self._obj, indexes, fetch_index=False)
        use_iloc, new_indexes = self._use_iloc(indexes)
        if use_iloc:
            op = DataFrameIlocSetItem(indexes=new_indexes, value=value)
            ret = op(self._obj)
            self._obj.data = ret.data
        else:
            other_indices = []
            indices_tileable = [
                idx
                for idx in indexes
                if isinstance(idx, ENTITY_TYPE) or other_indices.append(idx)
            ]
            op = DataFrameLocSetItem(indexes=other_indices, value=value)
            ret = op([self._obj] + indices_tileable)
            self._obj.data = ret.data


class DataFrameLocSetItem(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_ILOC_SETITEM

    indexes = ListField("indexes", default=None)
    value = AnyField("value", default=None)

    def __init__(self, gpu=None, sparse=False, output_types=None, **kw):
        super().__init__(
            gpu=gpu,
            sparse=sparse,
            _output_types=output_types,
            **kw,
        )
        if not self.output_types:
            self.output_types = [OutputType.dataframe]

    def __call__(self, inputs):
        df = inputs[0]
        return self.new_dataframe(
            inputs,
            shape=df.shape,
            dtypes=df.dtypes,
            index_value=df.index_value,
            columns_value=df.columns_value,
        )


class DataFrameLocGetItem(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_LOC_GETITEM

    _input = KeyField("input")
    indexes = ListField("indexes", default=None)

    def __init__(self, gpu=None, sparse=False, output_types=None, **kw):
        super().__init__(gpu=gpu, sparse=sparse, _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def can_index_miss(self):
        return False

    @classmethod
    def _set_inputs(cls, op: "DataFrameLocGetItem", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)
        op._input = next(inputs_iter)
        indexes = []
        for index in op.indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        op.indexes = list(indexes)

    @classmethod
    def _calc_slice_param(
        cls,
        input_index_value: IndexValue,
        pd_index: pd.Index,
        inp,
        index: slice,
        axis: int,
    ) -> Dict:
        param = dict()
        if is_full_slice(index):
            # full slice on this axis
            param["shape"] = inp.shape[axis]
            param["index_value"] = input_index_value
            if axis == 1:
                param["dtypes"] = inp.dtypes
        elif input_index_value.has_value():
            kw = {}
            if with_slice_locs_kind:
                kw["kind"] = "loc"
            start, end = pd_index.slice_locs(index.start, index.stop, index.step, **kw)
            slc = slice(start, end, index.step)
            size = calc_sliced_size(inp.shape[axis], slc)
            param["shape"] = size
            out_index = pd_index[slc]
            param["index_value"] = parse_index(out_index, store_data=axis == 1)
            if axis == 1:
                param["dtypes"] = inp.dtypes[slc]
        else:
            assert axis == 0
            if index.start is None and index.stop is None:
                param["shape"] = calc_sliced_size(inp.shape[axis], index)
            else:
                param["shape"] = np.nan
            param["index_value"] = parse_index(pd_index, inp, index)

        return param

    @classmethod
    def _calc_bool_index_param(
        cls, input_index_value: IndexValue, pd_index: pd.Index, inp, index, axis: int
    ) -> Dict:
        param = dict()
        if input_index_value.has_value():
            if isinstance(index, np.ndarray):
                filtered_index = pd_index[index]
                param["shape"] = len(filtered_index)
                param["index_value"] = parse_index(filtered_index, store_data=axis == 1)
                if axis == 1:
                    param["dtypes"] = inp.dtypes[index]
            else:
                # tensor, cannot be indexer on axis 1
                assert axis == 0
                param["shape"] = np.nan
                param["index_value"] = parse_index(
                    pd.Index([], dtype=pd_index.dtype), inp, index, store_data=False
                )
        else:
            assert axis == 0
            if isinstance(index, np.ndarray):
                param["shape"] = int(index.sum())
            else:
                param["shape"] = np.nan
            param["index_value"] = parse_index(pd_index, inp, index, store_data=False)

        return param

    @classmethod
    def _calc_fancy_index_param(
        cls, input_index_value: IndexValue, pd_index: pd.Index, inp, index, axis: int
    ) -> Dict:
        param = dict()
        if input_index_value.has_value():
            if isinstance(index, np.ndarray):
                if not pd_index.is_unique:
                    assert axis == 1
                    # as there's no direct method in pandas to handle fancy indexes
                    # we creates a empty
                    new_dtypes = inp.dtypes.loc[index]
                    param["shape"] = len(new_dtypes)
                    param["index_value"] = parse_index(
                        new_dtypes.index, store_data=True
                    )
                    param["dtypes"] = new_dtypes
                else:
                    for it in index:
                        if it not in pd_index:
                            axis_name = "index" if axis == 0 else "columns"
                            raise KeyError(
                                f"Label [{it}] not found in the [{axis_name}]"
                            )
                    param["shape"] = len(index)
                    param["index_value"] = parse_index(pd.Index(index), store_data=True)
                    if axis == 1:
                        param["dtypes"] = inp.dtypes[index]
            else:
                assert axis == 0
                param["shape"] = index.shape[0]
                param["index_value"] = parse_index(
                    pd.Index([], dtype=pd_index.dtype), inp, index
                )
        else:
            assert axis == 0
            param["shape"] = index.shape[0]
            param["index_value"] = parse_index(pd_index, inp, index)

        return param

    @classmethod
    def _calc_param(cls, inp, axis: int, index) -> Dict:
        input_index_value = inp.index_value if axis == 0 else inp.columns_value
        pd_index = input_index_value.to_pandas()

        if isinstance(index, slice):
            return cls._calc_slice_param(input_index_value, pd_index, inp, index, axis)
        elif hasattr(index, "dtype") and index.ndim == 1:
            if index.dtype == np.bool_:
                # bool indexing
                return cls._calc_bool_index_param(
                    input_index_value, pd_index, inp, index, axis
                )
            else:
                # fancy indexing
                return cls._calc_fancy_index_param(
                    input_index_value, pd_index, inp, index, axis
                )
        else:
            param = dict()
            if input_index_value.has_value():
                loc = pd_index.get_loc(index)
                if isinstance(loc, (slice, np.ndarray)):
                    assert axis == 1
                    new_dtypes = inp.dtypes[loc]
                    param["shape"] = len(new_dtypes)
                    param["index_value"] = parse_index(
                        new_dtypes.index, store_data=True
                    )
                    param["dtypes"] = new_dtypes
                else:
                    # append None to indicate returning Series
                    param["shape"] = None
            else:
                param["shape"] = None
            return param

    def __call__(self, inp):
        inputs = [inp] + filter_inputs(self.indexes)

        shape = []
        sizes = []
        index_value = columns_value = dtypes = None
        for ax, index in enumerate(self.indexes):
            param = self._calc_param(inp, ax, index)

            size = param.get("shape")
            sizes.append(size)
            if size is not None:
                shape.append(size)

            if ax == 0:
                index_value = param.get("index_value")
            else:
                columns_value = param.get("index_value")
                dtypes = param.get("dtypes")

        shape = tuple(shape)
        if len(shape) == 0:
            # scalar
            if isinstance(inp, DATAFRAME_TYPE):
                dtype = inp.dtypes[self.indexes[1]]
            else:
                dtype = inp.dtype
            return self.new_scalar(inputs, dtype=dtype)
        elif len(shape) == 1:
            # series
            if isinstance(inp, DATAFRAME_TYPE):
                if sizes[0] is None:
                    # label on axis 0
                    dtype = find_common_type(list(dtypes))
                    return self.new_series(
                        inputs,
                        shape=shape,
                        dtype=dtype,
                        index_value=columns_value,
                        name=self.indexes[0],
                    )
                else:
                    # label on axis 1
                    dtype = inp.dtypes[self.indexes[1]]
                    return self.new_series(
                        inputs,
                        shape=shape,
                        dtype=dtype,
                        index_value=index_value,
                        name=self.indexes[1],
                    )
            else:
                return self.new_series(
                    inputs,
                    shape=shape,
                    dtype=inp.dtype,
                    index_value=index_value,
                    name=inp.name,
                )
        else:
            # dataframe
            return self.new_dataframe(
                inputs,
                shape=shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=columns_value,
            )


def loc(a):
    return DataFrameLoc(a)
