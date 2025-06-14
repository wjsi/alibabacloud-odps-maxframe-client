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

from numbers import Number
from typing import List

import numpy as np

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import AnyField, KeyField
from ..core import Tensor
from ..datasource import tensor as astensor
from ..utils import broadcast_shape
from .core import TensorElementWise, TensorOperator, filter_inputs


class TensorClip(TensorOperator, TensorElementWise):
    _op_type_ = opcodes.CLIP

    a = KeyField("a", default=None)
    a_min = AnyField("a_min", default=None)
    a_max = AnyField("a_max", default=None)
    out = KeyField("out", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorClip", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)
        op.a = next(inputs_iter)
        if isinstance(op.a_min, ENTITY_TYPE):
            op.a_min = next(inputs_iter)
        if isinstance(op.a_max, ENTITY_TYPE):
            op.a_max = next(inputs_iter)
        if getattr(op, "_out", None) is not None:
            op.out = next(inputs_iter)

    def __call__(self, a, a_min, a_max, out=None):
        a = astensor(a)
        tensors = [a]
        sparse = a.issparse()

        if isinstance(a_min, Number):
            if a_min > 0:
                sparse = False
            a_min_dtype = np.array(a_min).dtype
        elif a_min is not None:
            a_min = astensor(a_min)
            tensors.append(a_min)
            if not a_min.issparse():
                sparse = False
            a_min_dtype = a_min.dtype
        else:
            a_min_dtype = None
        self.a_min = a_min

        if isinstance(a_max, Number):
            if a_max < 0:
                sparse = False
            a_max_dtype = np.array(a_max).dtype
        elif a_max is not None:
            a_max = astensor(a_max)
            tensors.append(a_max)
            if not a_max.issparse():
                sparse = False
            a_max_dtype = a_max.dtype
        else:
            a_max_dtype = None
        self.a_max = a_max

        if out is not None:
            if isinstance(out, Tensor):
                self.out = out
            else:
                raise TypeError(f"out should be Tensor object, got {type(out)} instead")

        dtypes = [dt for dt in [a.dtype, a_min_dtype, a_max_dtype] if dt is not None]
        dtype = np.result_type(*dtypes)
        # check broadcast
        shape = broadcast_shape(*[t.shape for t in tensors])

        setattr(self, "sparse", sparse)
        inputs = filter_inputs([a, a_min, a_max, out])
        t = self.new_tensor(inputs, shape)

        if out is None:
            setattr(self, "dtype", dtype)
            return t

        # if `out` is specified, use out's dtype and shape
        out_shape, out_dtype = out.shape, out.dtype

        if t.shape != out_shape:
            t = self.new_tensor(inputs, out_shape)
        setattr(self, "dtype", out_dtype)

        out.data = t.data
        return out


def clip(a, a_min, a_max, out=None):
    """
    Clip (limit) the values in a tensor.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : array_like
        Tensor containing elements to clip.
    a_min : scalar or array_like or `None`
        Minimum value. If `None`, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.
    a_max : scalar or array_like or `None`
        Maximum value. If `None`, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`. If `a_min` or `a_max` are array_like, then the three
        arrays will be broadcasted to match their shapes.
    out : Tensor, optional
        The results will be placed in this tensor. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.

    Returns
    -------
    clipped_array : Tensor
        An tensor with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> a = mt.arange(10)
    >>> mt.clip(a, 1, 8).execute()
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> a.execute()
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mt.clip(a, 3, 6, out=a).execute()
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a = mt.arange(10)
    >>> a.execute()
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mt.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8).execute()
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """
    op = TensorClip(a=a, a_min=a_min, a_max=a_max, out=out)
    return op(a, a_min, a_max, out=out)
