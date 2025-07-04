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

from collections import defaultdict
from contextlib import contextmanager

import numpy as np

from ..lib import sparse
from ..lib.sparse.core import get_dense_module, issparse
from ..utils import lazy_import

cp = lazy_import("cupy", rename="cp")


def is_array(x):
    if isinstance(x, np.ndarray):
        return True
    elif isinstance(x, (sparse.SparseMatrix, sparse.SparseVector)):
        return True
    elif cp:  # pragma: no cover
        return isinstance(x, cp.ndarray)
    else:
        return False


def is_cupy(x):
    if cp and isinstance(x, cp.ndarray):  # pragma: no cover
        return True
    else:
        return False


def get_array_module(x, nosparse=False):
    if issparse(x):
        if nosparse:
            return get_dense_module(x)
        return sparse
    if cp:
        return cp.get_array_module(x)
    return np


def array_module(gpu):
    if gpu:
        if cp is None:
            raise ImportError("Execute on GPU requires for `cupy` library")
        return cp

    return np


def _get(x):
    m = get_array_module(x)

    if m is np:
        return x
    if m is sparse:
        return x if not hasattr(x, "get") else x.get()
    return x.get()


def move_to_device(x, device_id):
    if hasattr(x, "device") and x.device.id == device_id:
        return x

    assert device_id >= 0

    if issparse(x) and device_id > 0:
        raise NotImplementedError

    # for dense array, we currently copy from gpu to memory and then copy back to destination device
    # to avoid kernel panic
    with cp.cuda.Device(device_id):
        return cp.asarray(cp.asnumpy(x))  # remove `cp.asnumpy` call to do directly copy


def convert_order(x, order):
    xp = get_array_module(x)
    if xp.isfortran(x) != (order == "F"):
        x = xp.array(x, order=order)
    return x


def _most_nbytes_device(device_nbytes):
    device_to_nbytes = defaultdict(lambda: 0)
    for device, nbytes in device_nbytes:
        device_to_nbytes[device] += nbytes
    return max(device_to_nbytes, key=lambda i: device_to_nbytes[i])


def _is_array_writeable(a):
    if hasattr(a, "flags") and hasattr(a.flags, "writeable"):
        return a.flags.writeable
    # writeable as default
    return True


def as_same_module(inputs, ret_extra=False, copy_if_not_writeable=False):
    input_tensors = [
        i for i in inputs if hasattr(i, "ndim") and i.ndim > 0
    ]  # filter scalar
    has_sparse = any(issparse(i) for i in inputs)

    outputs = [_get(i) for i in inputs]

    if copy_if_not_writeable:
        new_outputs = []
        for out in outputs:
            if not _is_array_writeable(out):
                new_outputs.append(out.copy())
            elif isinstance(out, (sparse.SparseMatrix, sparse.SparseVector)):
                if (
                    not _is_array_writeable(out.data)
                    or not _is_array_writeable(out.indices)
                    or not _is_array_writeable(out.indptr)
                ):
                    new_outputs.append(type(out)(out.spmatrix.copy(), shape=out.shape))
                else:
                    new_outputs.append(out)
            else:
                new_outputs.append(out)
        outputs = new_outputs

    if not ret_extra:
        return outputs

    if has_sparse:
        m = sparse
    else:
        if len(input_tensors) > 0:
            m = get_array_module(input_tensors[0])
        else:
            m = np
    return outputs, m


def as_np_array(x):
    xp = get_array_module(x)
    return x if xp == np else x.get()


def is_sparse_module(xp):
    return xp is sparse


@contextmanager
def device(device_id):
    if device_id is None or device_id < 0:
        yield
    else:  # pragma: no cover
        with cp.cuda.Device(device_id) as dev:
            yield dev
