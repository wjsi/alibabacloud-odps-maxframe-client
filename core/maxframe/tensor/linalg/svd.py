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
from numpy.linalg import LinAlgError

from ... import opcodes
from ...core import ExecutableTuple
from ...serialization.serializables import StringField
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorSVD(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.SVD

    method = StringField("method")

    @property
    def output_limit(self):
        return 3

    @classmethod
    def _is_svd(cls):
        return True

    @staticmethod
    def _calc_svd_shapes(a):
        """
        Calculate output shapes of singular value decomposition.
        Follow the behavior of `numpy`:
        if a's shape is (6, 18), U's shape is (6, 6), s's shape is (6,), V's shape is (6, 18)
        if a's shape is (18, 6), U's shape is (18, 6), s's shape is (6,), V's shape is (6, 6)
        :param a: input tensor
        :return: (U.shape, s.shape, V.shape)
        """
        x, y = a.shape
        if x > y:
            return (x, y), (y,), (y, y)
        else:
            return (x, x), (x,), (x, y)

    def __call__(self, a):
        a = astensor(a)

        if a.ndim != 2:
            raise LinAlgError(
                f"{a.ndim}-dimensional tensor given. Tensor must be two-dimensional"
            )

        tiny_U, tiny_s, tiny_V = np.linalg.svd(np.ones((1, 1), dtype=a.dtype))

        # if a's shape is (6, 18), U's shape is (6, 6), s's shape is (6,), V's shape is (6, 18)
        # if a's shape is (18, 6), U's shape is (18, 6), s's shape is (6,), V's shape is (6, 6)
        U_shape, s_shape, V_shape = self._calc_svd_shapes(a)
        U, s, V = self.new_tensors(
            [a],
            order=TensorOrder.C_ORDER,
            kws=[
                {"side": "U", "dtype": tiny_U.dtype, "shape": U_shape},
                {"side": "s", "dtype": tiny_s.dtype, "shape": s_shape},
                {"side": "V", "dtype": tiny_V.dtype, "shape": V_shape},
            ],
        )
        return ExecutableTuple([U, s, V])


def svd(a, method="tsqr"):
    """
    Singular Value Decomposition.

    When `a` is a 2D tensor, it is factorized as ``u @ np.diag(s) @ vh
    = (u * s) @ vh``, where `u` and `vh` are 2D unitary tensors and `s` is a 1D
    tensor of `a`'s singular values. When `a` is higher-dimensional, SVD is
    applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) array_like
        A real or complex tensor with ``a.ndim >= 2``.
    method: {'tsqr'}, optional
        method to calculate qr factorization, tsqr as default

        TSQR is presented in:

            A. Benson, D. Gleich, and J. Demmel.
            Direct QR factorizations for tall-and-skinny matrices in
            MapReduce architectures.
            IEEE International Conference on Big Data, 2013.
            http://arxiv.org/abs/1301.1071


    Returns
    -------
    u : { (..., M, M), (..., M, K) } tensor
        Unitary tensor(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    s : (..., K) tensor
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    vh : { (..., N, N), (..., K, N) } tensor
        Unitary tensor(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Notes
    -----

    SVD is usually described for the factorization of a 2D matrix :math:`A`.
    The higher-dimensional case will be discussed below. In the 2D case, SVD is
    written as :math:`A = U S V^H`, where :math:`A = a`, :math:`U= u`,
    :math:`S= \\mathtt{np.diag}(s)` and :math:`V^H = vh`. The 1D tensor `s`
    contains the singular values of `a` and `u` and `vh` are unitary. The rows
    of `vh` are the eigenvectors of :math:`A^H A` and the columns of `u` are
    the eigenvectors of :math:`A A^H`. In both cases the corresponding
    (possibly non-zero) eigenvalues are given by ``s**2``.

    If `a` has more than two dimensions, then broadcasting rules apply, as
    explained in :ref:`routines.linalg-broadcasting`. This means that SVD is
    working in "stacked" mode: it iterates over all indices of the first
    ``a.ndim - 2`` dimensions and for each combination SVD is applied to the
    last two indices. The matrix `a` can be reconstructed from the
    decomposition with either ``(u * s[..., None, :]) @ vh`` or
    ``u @ (s[..., None] * vh)``. (The ``@`` operator can be replaced by the
    function ``mt.matmul`` for python versions below 3.5.)

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> a = mt.random.randn(9, 6) + 1j*mt.random.randn(9, 6)
    >>> b = mt.random.randn(2, 7, 8, 3) + 1j*mt.random.randn(2, 7, 8, 3)

    Reconstruction based on reduced SVD, 2D case:

    >>> u, s, vh = mt.linalg.svd(a)
    >>> u.shape, s.shape, vh.shape
    ((9, 6), (6,), (6, 6))
    >>> np.allclose(a, np.dot(u * s, vh))
    True
    >>> smat = np.diag(s)
    >>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
    True

    """
    op = TensorSVD(method=method)
    return op(a)
