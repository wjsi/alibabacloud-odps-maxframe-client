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
from ...core import ExecutableTuple
from ...serialization.serializables import KeyField
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorNonzero(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.NONZERO

    _input = KeyField("input")

    @property
    def output_limit(self):
        return float("inf")

    def __call__(self, a):
        kws = [
            {"shape": (np.nan,), "order": TensorOrder.C_ORDER, "_idx_": i}
            for i in range(a.ndim)
        ]
        return ExecutableTuple(self.new_tensors([a], kws=kws, output_limit=len(kws)))


def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of tensors, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension. The values in `a` are always tested and returned.
    The corresponding non-zero
    values can be obtained with::

        a[nonzero(a)]

    To group the indices by element, rather than dimension, use::

        transpose(nonzero(a))

    The result of this is always a 2-D array, with a row for
    each non-zero element.

    Parameters
    ----------
    a : array_like
        Input tensor.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    flatnonzero :
        Return indices that are non-zero in the flattened version of the input
        tensor.
    Tensor.nonzero :
        Equivalent tensor method.
    count_nonzero :
        Counts the number of non-zero elements in the input tensor.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.array([[1,0,0], [0,2,0], [1,1,0]])
    >>> x.execute()
    array([[1, 0, 0],
           [0, 2, 0],
           [1, 1, 0]])
    >>> mt.nonzero(x).execute()
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

    >>> x[mt.nonzero(x)].execute()

    >>> mt.transpose(mt.nonzero(x)).execute()

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is True.  Given an array `a`, the condition `a` > 3 is a
    boolean array and since False is interpreted as 0, np.nonzero(a > 3)
    yields the indices of the `a` where the condition is true.

    >>> a = mt.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> (a > 3).execute()
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> mt.nonzero(a > 3).execute()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    The ``nonzero`` method of the boolean array can also be called.

    >>> (a > 3).nonzero().execute()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    """
    a = astensor(a)
    op = TensorNonzero(dtype=np.dtype(np.intp))
    return op(a)
