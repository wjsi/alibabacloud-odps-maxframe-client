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
from ..utils import gen_random_seeds
from .core import TensorDistribution, TensorRandomOperatorMixin


class TensorStandardExponential(TensorDistribution, TensorRandomOperatorMixin):
    _op_type_ = opcodes.RAND_STANDARD_EXPONENTIAL
    _func_name = "standard_exponential"
    _fields_ = ("size",)

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)


def standard_exponential(
    random_state, size=None, chunk_size=None, gpu=None, dtype=None
):
    """
    Draw samples from the standard exponential distribution.

    `standard_exponential` is identical to the exponential distribution
    with a scale parameter of 1.

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : float or Tensor
        Drawn samples.

    Examples
    --------
    Output a 3x8000 tensor:

    >>> import maxframe.tensor as mt
    >>> n = mt.random.standard_exponential((3, 8000))
    """
    if dtype is None:
        dtype = np.random.RandomState().standard_exponential(size=(0,)).dtype
    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorStandardExponential(size=size, seed=seed, gpu=gpu, dtype=dtype)
    return op(chunk_size=chunk_size)
