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

import functools

import numpy as np

from ... import opcodes
from ...serialization.serializables import BoolField
from ..datasource import scalar
from ..utils import infer_dtype
from .core import TensorBinOp, TensorMultiOp
from .utils import TreeReductionBuilder, arithmetic_operator


@arithmetic_operator(sparse_mode="binary_or")
class TensorMultiply(TensorBinOp):
    _op_type_ = opcodes.MUL
    _func_name = "multiply"


@infer_dtype(np.multiply)
def multiply(x1, x2, out=None, where=None, **kwargs):
    """
    Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays to be multiplied.
    out : Tensor, None, or tuple of Tensor and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated tensor is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs

    Returns
    -------
    y : Tensor
        The product of `x1` and `x2`, element-wise. Returns a scalar if
        both  `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to `x1` * `x2` in terms of array broadcasting.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.multiply(2.0, 4.0).execute()
    8.0

    >>> x1 = mt.arange(9.0).reshape((3, 3))
    >>> x2 = mt.arange(3.0)
    >>> mt.multiply(x1, x2).execute()
    array([[  0.,   1.,   4.],
           [  0.,   4.,  10.],
           [  0.,   7.,  16.]])
    """
    op = TensorMultiply(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.multiply, reverse=True)
def rmultiply(x1, x2, **kwargs):
    op = TensorMultiply(**kwargs)
    return op.rcall(x1, x2)


class TensorTreeMultiply(TensorMultiOp):
    _op_type_ = opcodes.TREE_MULTIPLY
    _func_name = "multiply"

    ignore_empty_input = BoolField("ignore_empty_input", default=False)

    def __init__(self, sparse=False, **kw):
        super().__init__(sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, *args):
        if not args or all(np.isscalar(x) for x in args):
            return False
        if all(
            np.isscalar(x) or (hasattr(x, "issparse") and x.issparse()) for x in args
        ):
            return True
        return False


@infer_dtype(lambda *args: functools.reduce(np.multiply, args))
def tree_multiply(*args, combine_size=None, **kwargs):
    class MultiplyBuilder(TreeReductionBuilder):
        def _build_reduction(self, inputs, final=False):
            op = TensorTreeMultiply(args=inputs, **kwargs)
            return op(*inputs)

    args = [scalar(a) if np.isscalar(a) else a for a in args]
    return MultiplyBuilder(combine_size).build(args)
