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

from .dot import TensorDot, dot
from .inner import inner, innerproduct
from .inv import TensorInv, inv
from .lu import TensorLU, lu
from .matmul import TensorMatmul, matmul
from .qr import TensorQR, qr
from .solve_triangular import TensorSolveTriangular, solve_triangular
from .svd import TensorSVD, svd
from .tensordot import TensorTensorDot, tensordot
from .vdot import vdot


def _install():
    from ..core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, "__matmul__", matmul)
        setattr(cls, "dot", dot)


_install()
del _install
