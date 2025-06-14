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
from ...serialization.serializables import AnyField
from ..utils import gen_random_seeds
from .core import TensorDistribution, TensorRandomOperatorMixin, handle_array


class TensorTriangular(TensorDistribution, TensorRandomOperatorMixin):
    _input_fields_ = ["left", "mode", "right"]
    _op_type_ = opcodes.RAND_TRIANGULAR

    _fields_ = "left", "mode", "right", "size"
    left = AnyField("left")
    mode = AnyField("mode")
    right = AnyField("right")
    _func_name = "triangular"

    def __call__(self, left, mode, right, chunk_size=None):
        return self.new_tensor([left, mode, right], None, raw_chunk_size=chunk_size)


def triangular(
    random_state, left, mode, right, size=None, chunk_size=None, gpu=None, dtype=None
):
    r"""
    Draw samples from the triangular distribution over the
    interval ``[left, right]``.

    The triangular distribution is a continuous probability
    distribution with lower limit left, peak at mode, and upper
    limit right. Unlike the other distributions, these parameters
    directly define the shape of the pdf.

    Parameters
    ----------
    left : float or array_like of floats
        Lower limit.
    mode : float or array_like of floats
        The value where the peak of the distribution occurs.
        The value should fulfill the condition ``left <= mode <= right``.
    right : float or array_like of floats
        Upper limit, should be larger than `left`.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``left``, ``mode``, and ``right``
        are all scalars.  Otherwise, ``mt.broadcast(left, mode, right).size``
        samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized triangular distribution.

    Notes
    -----
    The probability density function for the triangular distribution is

    .. math:: P(x;l, m, r) = \begin{cases}
              \frac{2(x-l)}{(r-l)(m-l)}& \text{for $l \leq x \leq m$},\\
              \frac{2(r-x)}{(r-l)(r-m)}& \text{for $m \leq x \leq r$},\\
              0& \text{otherwise}.
              \end{cases}

    The triangular distribution is often used in ill-defined
    problems where the underlying distribution is not known, but
    some knowledge of the limits and mode exists. Often it is used
    in simulations.

    References
    ----------
    .. [1] Wikipedia, "Triangular distribution"
           http://en.wikipedia.org/wiki/Triangular_distribution

    Examples
    --------
    Draw values from the distribution and plot the histogram:

    >>> import matplotlib.pyplot as plt
    >>> import maxframe.tensor as mt
    >>> h = plt.hist(mt.random.triangular(-3, 0, 8, 100000).execute(), bins=200,
    ...              normed=True)
    >>> plt.show()
    """
    if dtype is None:
        dtype = (
            np.random.RandomState()
            .triangular(
                handle_array(left), handle_array(mode), handle_array(right), size=(0,)
            )
            .dtype
        )
    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorTriangular(size=size, seed=seed, gpu=gpu, dtype=dtype)
    return op(left, mode, right, chunk_size=chunk_size)
