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
from ...serialization.serializables import BoolField
from ..utils import infer_dtype
from .core import TensorUnaryOp
from .utils import arithmetic_operator


@arithmetic_operator(init=False, sparse_mode="unary")
class TensorAngle(TensorUnaryOp):
    _op_type_ = opcodes.ANGLE
    _func_name = "angle"

    deg = BoolField("deg", default=None)

    def __init__(self, casting="same_kind", err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super().__init__(casting=casting, err=err, dtype=dtype, sparse=sparse, **kw)


@infer_dtype(np.angle)
def angle(z, deg=False, **kwargs):
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    deg : bool, optional
        Return angle in degrees if True, radians if False (default).

    Returns
    -------
    angle : Tensor or scalar
        The counterclockwise angle from the positive real axis on
        the complex plane, with dtype as numpy.float64.

    See Also
    --------
    arctan2
    absolute

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.angle([1.0, 1.0j, 1+1j]).execute()               # in radians
    array([ 0.        ,  1.57079633,  0.78539816])
    >>> mt.angle(1+1j, deg=True).execute()                  # in degrees
    45.0

    """
    op = TensorAngle(deg=deg, **kwargs)
    return op(z)
