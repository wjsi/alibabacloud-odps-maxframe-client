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


@arithmetic_operator(sparse_mode="always_false")
class TensorRint(TensorUnaryOp):
    _op_type_ = opcodes.RINT
    _func_name = "rint"


@infer_dtype(np.rint)
def rint(x, out=None, where=None, **kwargs):
    """
    Round elements of the tensor to the nearest integer.

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
    out : Tensor or scalar
        Output array is same shape and type as `x`.

    See Also
    --------
    ceil, floor, trunc

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> a = mt.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> mt.rint(a).execute()
    array([-2., -2., -0.,  0.,  2.,  2.,  2.])
    """
    op = TensorRint(**kwargs)
    return op(x, out=out, where=where)
