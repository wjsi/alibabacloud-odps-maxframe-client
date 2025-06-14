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
from ..utils import infer_dtype
from .core import TensorUnaryOp
from .utils import arithmetic_operator


@arithmetic_operator(sparse_mode="unary")
class TensorConj(TensorUnaryOp):
    _op_type_ = opcodes.CONJ
    _func_name = "conj"


@infer_dtype(np.conj)
def conj(x, out=None, where=None, **kwargs):
    """
    Return the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the
    sign of its imaginary part.

    Parameters
    ----------
    x : array_like
        Input value.
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
        The complex conjugate of `x`, with same dtype as `y`.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.conjugate(1+2j).execute()
    (1-2j)

    >>> x = mt.eye(2) + 1j * mt.eye(2)
    >>> mt.conjugate(x).execute()
    array([[ 1.-1.j,  0.-0.j],
           [ 0.-0.j,  1.-1.j]])
    """
    op = TensorConj(**kwargs)
    return op(x, out=out, where=where)


conjugate = conj
