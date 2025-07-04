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
from ...lib.sparse.core import cp, get_array_module, issparse
from ...serialization.serializables import (
    AnyField,
    FieldTypes,
    NDArrayField,
    StringField,
    TupleField,
)
from ...utils import on_deserialize_shape, on_serialize_shape
from ..array_utils import is_array, is_cupy
from ..core import TENSOR_TYPE, Tensor, TensorData, TensorOrder
from .core import TensorNoInput
from .scalar import scalar


class ArrayDataSource(TensorNoInput):
    """
    Represents data from numpy or cupy array
    """

    _op_type_ = opcodes.TENSOR_DATA_SOURCE

    data = NDArrayField("data", default=None)
    chunk_size = AnyField("chunk_size", default=None)
    order = StringField("order", default=None)

    def __init__(self, data=None, dtype=None, gpu=None, **kw):
        if dtype is not None:
            dtype = np.dtype(dtype)
        elif data is not None:
            dtype = np.dtype(data.dtype)

        if gpu is None and is_cupy(data):  # pragma: no cover
            gpu = True

        super().__init__(data=data, dtype=dtype, gpu=gpu, **kw)

    def get_data(self):
        return self.data


class CSRMatrixDataSource(TensorNoInput):
    """
    Represents data from sparse array include scipy sparse or cupy sparse matrix.
    """

    _op_type_ = opcodes.SPARSE_MATRIX_DATA_SOURCE

    indices = NDArrayField("indices")
    indptr = NDArrayField("indptr")
    data = NDArrayField("data")
    shape = TupleField(
        "shape",
        FieldTypes.int64,
        on_serialize=on_serialize_shape,
        on_deserialize=on_deserialize_shape,
    )
    chunk_size = AnyField("chunk_size")

    def __init__(self, data=None, **kw):
        kw["sparse"] = True
        if is_cupy(data):  # pragma: no cover
            kw["gpu"] = True
        super().__init__(data=data, **kw)


def _from_spmatrix(spmatrix, dtype=None, chunk_size=None, gpu=None):
    if gpu is None:
        m = get_array_module(spmatrix)
        if cp is not None and m is cp:
            gpu = True
        elif cp is np:
            gpu = False
    if dtype and spmatrix.dtype != dtype:
        spmatrix = spmatrix.astype(dtype)
    spmatrix = spmatrix.tocsr()
    op = CSRMatrixDataSource(
        indices=spmatrix.indices,
        indptr=spmatrix.indptr,
        data=spmatrix.data,
        shape=spmatrix.shape,
        dtype=spmatrix.dtype,
        gpu=gpu,
        chunk_size=chunk_size,
    )
    return op(spmatrix.shape, chunk_size=chunk_size)


def tensor(
    data=None, dtype=None, order="K", chunk_size=None, gpu=None, sparse=False
) -> Tensor:
    order = order or "K"
    if isinstance(data, TENSOR_TYPE):
        if isinstance(data, TensorData):
            data = Tensor(data)
        return data.astype(dtype or data.dtype, order=order, copy=False)
    elif (
        isinstance(data, (tuple, list))
        and len(data) > 0
        and all(isinstance(d, TENSOR_TYPE) for d in data)
    ):
        from ..merge import stack

        data = stack(data)
        return data.astype(dtype or data.dtype, order=order, copy=False)
    elif np.isscalar(data):
        return scalar(data, dtype=dtype)
    elif issparse(data):
        return _from_spmatrix(data, dtype=dtype, chunk_size=chunk_size, gpu=gpu)
    elif hasattr(data, "__maxframe_tensor__"):
        return data.__maxframe_tensor__(dtype=dtype, order=order)
    else:
        m = get_array_module(data)
        try:
            data = m.asarray(data, dtype=dtype, order=order)
        except ValueError:
            if not hasattr(data, "__array__"):
                raise
            arr = data.__array__(dtype=dtype)
            if isinstance(arr, TENSOR_TYPE):
                return arr.astype(arr.dtype, order=order, copy=False)
            raise
        if gpu is None:
            if cp is not None and m is cp:
                gpu = True

    if is_array(data):
        if data.ndim == 0:
            return scalar(data.item(), dtype=dtype)
        tensor_order = (
            TensorOrder.C_ORDER if data.flags["C_CONTIGUOUS"] else TensorOrder.F_ORDER
        )
        op = ArrayDataSource(data, dtype=dtype, gpu=gpu, chunk_size=chunk_size)
        t = op(data.shape, chunk_size=chunk_size, order=tensor_order)
        if sparse and not t.issparse():
            return t.tosparse()
        return t
    else:
        raise ValueError(f"Cannot create tensor by given data: {data}")


def array(x, dtype=None, copy=True, order="K", ndmin=None, chunk_size=None):
    """
    Create a tensor.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.  This argument can only be used to 'upcast' the array.  For
        downcasting, use the .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy will
        only be made if __array__ returns a copy, if obj is a nested sequence,
        or if a copy is needed to satisfy any of the other requirements
        (`dtype`, `order`, etc.).
    order : {'K', 'A', 'C', 'F'}, optional
        Specify the memory layout of the array. If object is not an array, the
        newly created array will be in C order (row major) unless 'F' is
        specified, in which case it will be in Fortran order (column major).
        If object is an array the following holds.

        ===== ========= ===================================================
        order  no copy                     copy=True
        ===== ========= ===================================================
        'K'   unchanged F & C order preserved, otherwise most similar order
        'A'   unchanged F order if input is F and not C, otherwise C order
        'C'   C order   C order
        'F'   F order   F order
        ===== ========= ===================================================

        When ``copy=False`` and a copy is made for other reasons, the result is
        the same as if ``copy=True``, with some exceptions for `A`, see the
        Notes section. The default order is 'K'.
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be prepended to the shape as
        needed to meet this requirement.
    chunk_size: int, tuple, optional
        Specifies chunk size for each dimension.

    Returns
    -------
    out : Tensor
        An tensor object satisfying the specified requirements.

    See Also
    --------
    empty, empty_like, zeros, zeros_like, ones, ones_like, full, full_like

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.array([1, 2, 3]).execute()
    array([1, 2, 3])

    Upcasting:

    >>> mt.array([1, 2, 3.0]).execute()
    array([ 1.,  2.,  3.])

    More than one dimension:

    >>> mt.array([[1, 2], [3, 4]]).execute()
    array([[1, 2],
           [3, 4]])

    Minimum dimensions 2:

    >>> mt.array([1, 2, 3], ndmin=2).execute()
    array([[1, 2, 3]])

    Type provided:

    >>> mt.array([1, 2, 3], dtype=complex).execute()
    array([ 1.+0.j,  2.+0.j,  3.+0.j])

    """
    raw_x = x
    order = order or "K"
    x = tensor(x, dtype=dtype, order=order, chunk_size=chunk_size)
    while ndmin is not None and x.ndim < ndmin:
        x = x[np.newaxis]

    if copy and x is raw_x:
        x = x.copy(order=order)
    elif (
        not copy
        and isinstance(raw_x, TENSOR_TYPE)
        and raw_x.dtype == x.dtype
        and raw_x.order == x.order
        and raw_x.shape == x.shape
        and raw_x is not x
        and hasattr(raw_x, "data")
    ):
        raw_x.data = x.data

    return x


def asarray(x, dtype=None, order=None, chunk_size=None):
    """Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to a tensor.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and tensors.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or
        column-major (Fortran-style) memory representation.
    chunk_size: int, tuple, optional
        Specifies chunk size for each dimension.

    Returns
    -------
    out : Tensor
        Tensor interpretation of `a`.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If `a` is a
        subclass of ndarray, a base class ndarray is returned.

    See Also
    --------
    ascontiguousarray : Convert input to a contiguous tensor.
    asfortranarray : Convert input to a tensor with column-major
                     memory order.

    Examples
    --------
    Convert a list into a tensor:

    >>> import maxframe.tensor as mt

    >>> a = [1, 2]
    >>> mt.asarray(a).execute()
    array([1, 2])

    Existing arrays are not copied:

    >>> a = mt.array([1, 2])
    >>> mt.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = mt.array([1, 2], dtype=mt.float32)
    >>> mt.asarray(a, dtype=mt.float32) is a
    True
    >>> mt.asarray(a, dtype=mt.float64) is a
    False
    """
    return array(x, dtype=dtype, copy=False, order=order, chunk_size=chunk_size)


def ascontiguousarray(a, dtype=None, chunk_size=None):
    """
    Return a contiguous tensor (ndim >= 1) in memory (C order).

    Parameters
    ----------
    a : array_like
        Input tensor.
    dtype : str or dtype object, optional
        Data-type of returned tensor.
    chunk_size: int, tuple, optional
        Specifies chunk size for each dimension.

    Returns
    -------
    out : Tensor
        Contiguous tensor of same shape and content as `a`, with type `dtype`
        if specified.

    See Also
    --------
    asfortranarray : Convert input to a tensor with column-major
                     memory order.
    Tensor.flags : Information about the memory layout of the tensor.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> x = mt.arange(6).reshape(2,3)
    >>> mt.ascontiguousarray(x, dtype=mt.float32)
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]], dtype=float32)
    >>> x.flags['C_CONTIGUOUS']
    True

    Note: This function returns a tensor with at least one-dimension (1-d)
    so it will not preserve 0-d tensors.

    """

    return array(a, dtype, copy=False, order="C", ndmin=1, chunk_size=chunk_size)


def asfortranarray(a, dtype=None, chunk_size=None):
    """
    Return a tensor (ndim >= 1) laid out in Fortran order in memory.

    Parameters
    ----------
    a : array_like
        Input tensor.
    dtype : str or dtype object, optional
        By default, the data-type is inferred from the input data.
    chunk_size: int, tuple, optional
        Specifies chunk size for each dimension.

    Returns
    -------
    out : Tensor
        The input `a` in Fortran, or column-major, order.

    See Also
    --------
    ascontiguousarray : Convert input to a contiguous (C order) tensor.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> x = mt.arange(6).reshape(2,3)
    >>> y = mt.asfortranarray(x)
    >>> x.flags['F_CONTIGUOUS']
    False
    >>> y.flags['F_CONTIGUOUS']
    True

    Note: This function returns a tensor with at least one-dimension (1-d)
    so it will not preserve 0-d tensors.

    """
    return array(a, dtype, copy=False, order="F", ndmin=1, chunk_size=chunk_size)
