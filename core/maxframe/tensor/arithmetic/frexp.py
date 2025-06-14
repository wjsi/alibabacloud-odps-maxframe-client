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

import numpy as np

from ... import opcodes
from .core import TensorOutBinOp


class TensorFrexp(TensorOutBinOp):
    _op_type_ = opcodes.FREXP
    _func_name = "frexp"

    def __init__(self, casting="same_kind", dtype=None, sparse=False, **kw):
        super().__init__(casting=casting, dtype=dtype, sparse=sparse, **kw)

    @property
    def _fun(self):
        return np.frexp

    @classmethod
    def _is_sparse(cls, x):
        if hasattr(x, "issparse") and x.issparse():
            return True
        return False


def frexp(x, out1=None, out2=None, out=None, where=None, **kwargs):
    """
    Decompose the elements of x into mantissa and twos exponent.

    Returns (`mantissa`, `exponent`), where `x = mantissa * 2**exponent``.
    The mantissa is lies in the open interval(-1, 1), while the twos
    exponent is a signed integer.

    Parameters
    ----------
    x : array_like
        Tensor of numbers to be decomposed.
    out1 : Tensor, optional
        Output tensor for the mantissa. Must have the same shape as `x`.
    out2 : Tensor, optional
        Output tensor for the exponent. Must have the same shape as `x`.
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
    (mantissa, exponent) : tuple of tensors, (float, int)
        `mantissa` is a float array with values between -1 and 1.
        `exponent` is an int array which represents the exponent of 2.

    See Also
    --------
    ldexp : Compute ``y = x1 * 2**x2``, the inverse of `frexp`.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.arange(9)
    >>> y1, y2 = mt.frexp(x)

    >>> y1_result, y2_result = mt.ExecutableTuple([y1, y2]).execute()
    >>> y1_result
    array([ 0.   ,  0.5  ,  0.5  ,  0.75 ,  0.5  ,  0.625,  0.75 ,  0.875,
            0.5  ])
    >>> y2_result
    array([0, 1, 2, 2, 3, 3, 3, 3, 4])
    >>> (y1 * 2**y2).execute()
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    """
    op = TensorFrexp(**kwargs)
    return op(x, out1=out1, out2=out2, out=out, where=where)
