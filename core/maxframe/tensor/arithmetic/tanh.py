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
class TensorTanh(TensorUnaryOp):
    _op_type_ = opcodes.TANH
    _func_name = "tanh"


@infer_dtype(np.tanh)
def tanh(x, out=None, where=None, **kwargs):
    """
    Compute hyperbolic tangent element-wise.

    Equivalent to ``mt.sinh(x)/np.cosh(x)`` or ``-1j * mt.tan(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input tensor.
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
        The corresponding hyperbolic tangent values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    .. [1] M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
           New York, NY: Dover, 1972, pg. 83.
           http://www.math.sfu.ca/~cbm/aands/

    .. [2] Wikipedia, "Hyperbolic function",
           http://en.wikipedia.org/wiki/Hyperbolic_function

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.tanh((0, mt.pi*1j, mt.pi*1j/2)).execute()
    array([ 0. +0.00000000e+00j,  0. -1.22460635e-16j,  0. +1.63317787e+16j])

    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = mt.zeros(1)
    >>> out2 = mt.tanh([0.1], out1)
    >>> out2 is out1
    True

    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> mt.tanh(mt.zeros((3,3)),mt.zeros((2,2)))
    Traceback (most recent call last):
    ...
    ValueError: operators could not be broadcast together with shapes (3,3) (2,2)
    """
    op = TensorTanh(**kwargs)
    return op(x, out=out, where=where)
