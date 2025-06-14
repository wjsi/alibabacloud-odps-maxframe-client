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
import pytest
import scipy.sparse as sps

from ....core import enter_mode
from ....utils import collect_leaf_operators
from ...core import SparseTensor, Tensor
from ...datasource import array, empty, ones, tensor
from .. import *  # noqa: F401
from ..core import TensorBinOp, TensorUnaryOp


def test_add():
    t1 = ones((3, 4), chunk_size=2)
    t2 = ones(4, chunk_size=2)
    t3 = t1 + t2
    assert t3.op.gpu is None
    assert t3.shape == (3, 4)
    assert t3.op.dtype == np.dtype("f8")

    t1 = ones((3, 4), chunk_size=2)
    t4 = t1 + 1
    assert t4.shape == (3, 4)

    t2 = ones(4, chunk_size=2)
    t6 = ones((3, 4), chunk_size=2, gpu=True)
    t7 = ones(4, chunk_size=2, gpu=True)
    t8 = t6 + t7
    t9 = t6 + t2
    assert t8.op.gpu is True
    assert t9.op.gpu is None

    # sparse tests
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = t1 + 1
    assert t.op.gpu is None
    assert t.issparse() is False
    assert type(t) is Tensor

    t = t1 + 0
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

    t = t1 + t2
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t3 = tensor([1, 1, 1], chunk_size=2)
    t = t1 + t3
    assert t.issparse() is False
    assert type(t) is Tensor


def test_add_order():
    raw_a = np.random.rand(4, 2)
    raw_b = np.asfortranarray(np.random.rand(4, 2))
    t1 = tensor(raw_a)
    t2 = tensor(raw_b)
    out = tensor(raw_b)

    # C + scalar
    assert (t1 + 1).flags["C_CONTIGUOUS"] == (raw_a + 1).flags["C_CONTIGUOUS"]
    assert (t1 + 1).flags["F_CONTIGUOUS"] == (raw_a + 1).flags["F_CONTIGUOUS"]
    # C + C
    assert (t1 + t1).flags["C_CONTIGUOUS"] == (raw_a + raw_a).flags["C_CONTIGUOUS"]
    assert (t1 + t1).flags["F_CONTIGUOUS"] == (raw_a + raw_a).flags["F_CONTIGUOUS"]
    # F + scalar
    assert (t2 + 1).flags["C_CONTIGUOUS"] == (raw_b + 1).flags["C_CONTIGUOUS"]
    assert (t2 + 1).flags["F_CONTIGUOUS"] == (raw_b + 1).flags["F_CONTIGUOUS"]
    # F + F
    assert (t2 + t2).flags["C_CONTIGUOUS"] == (raw_b + raw_b).flags["C_CONTIGUOUS"]
    assert (t2 + t2).flags["F_CONTIGUOUS"] == (raw_b + raw_b).flags["F_CONTIGUOUS"]
    # C + F
    assert (t1 + t2).flags["C_CONTIGUOUS"] == (raw_a + raw_b).flags["C_CONTIGUOUS"]
    assert (t1 + t2).flags["F_CONTIGUOUS"] == (raw_a + raw_b).flags["F_CONTIGUOUS"]
    # C + C + out
    assert (
        add(t1, t1, out=out).flags["C_CONTIGUOUS"]
        == np.add(raw_a, raw_a, out=np.empty((4, 2), order="F")).flags["C_CONTIGUOUS"]
    )
    assert (
        add(t1, t1, out=out).flags["F_CONTIGUOUS"]
        == np.add(raw_a, raw_a, out=np.empty((4, 2), order="F")).flags["F_CONTIGUOUS"]
    )

    with pytest.raises(TypeError):
        add(t1, 1, order="B")


def test_multiply():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = t1 * 10
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

    t = t1 * t2
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t3 = tensor([1, 1, 1], chunk_size=2)
    t = t1 * t3
    assert t.issparse() is True
    assert type(t) is SparseTensor


def test_divide():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = t1 / 10
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

    t = t1 / t2
    assert t.issparse() is False
    assert type(t) is Tensor

    t3 = tensor([1, 1, 1], chunk_size=2)
    t = t1 / t3
    assert t.issparse() is False
    assert type(t) is Tensor

    t = t3 / t1
    assert t.issparse() is False
    assert type(t) is Tensor


def test_datatime_arith():
    t1 = array([np.datetime64("2005-02-02"), np.datetime64("2005-02-03")])
    t2 = t1 + np.timedelta64(1)

    assert isinstance(t2.op, TensorAdd)

    t3 = t1 - np.datetime64("2005-02-02")

    assert isinstance(t3.op, TensorSubtract)
    assert (
        t3.dtype
        == (
            np.array(["2005-02-02", "2005-02-03"], dtype=np.datetime64)
            - np.datetime64("2005-02-02")
        ).dtype
    )

    t1 = array([np.datetime64("2005-02-02"), np.datetime64("2005-02-03")])
    subtract(t1, np.datetime64("2005-02-02"), out=empty(t1.shape, dtype=t3.dtype))

    t1 = array([np.datetime64("2005-02-02"), np.datetime64("2005-02-03")])
    add(t1, np.timedelta64(1, "D"), out=t1)


def test_add_with_out():
    t1 = ones((3, 4), chunk_size=2)
    t2 = ones(4, chunk_size=2)

    t3 = add(t1, t2, out=t1)

    assert isinstance(t1.op, TensorAdd)
    assert t1.op.out.key == t1.op.lhs.key
    assert t3 is t1
    assert t3.shape == (3, 4)
    assert t3.op.lhs.extra_params.raw_chunk_size == 2
    assert t3.op.rhs is t2.data
    assert t3.key != t3.op.lhs.key

    with pytest.raises(TypeError):
        add(t1, t2, out=1)

    with pytest.raises(ValueError):
        add(t1, t2, out=t2)

    with pytest.raises(TypeError):
        truediv(t1, t2, out=t1.astype("i8"))

    t1 = ones((3, 4), chunk_size=2, dtype=float)
    t2 = ones(4, chunk_size=2, dtype=int)

    t3 = add(t2, 1, out=t1)
    assert t3.shape == (3, 4)
    assert t3.dtype == np.float64


def test_dtype_from_out():
    x = array([-np.inf, 0.0, np.inf])
    y = array([2, 2, 2])

    t3 = isfinite(x, y)
    assert t3.dtype == y.dtype


def test_log_without_where():
    t1 = ones((3, 4), chunk_size=2)

    t2 = log(t1, out=t1)

    assert isinstance(t2.op, TensorLog)
    assert t1.op.out.key == t1.op.input.key
    assert t2 is t1
    assert t2.op.input.extra_params.raw_chunk_size == 2
    assert t2.key != t2.op.input.key

    t3 = empty((3, 4), chunk_size=2)
    t4 = log(t1, out=t3, where=t1 > 0)
    assert isinstance(t4.op, TensorLog)
    assert t4 is t3
    assert t2.op.input.extra_params.raw_chunk_size == 2
    assert t2.key != t2.op.input.key


def test_compare():
    t1 = ones(4, chunk_size=2) * 2
    t2 = ones(4, chunk_size=2)
    t3 = t1 > t2
    assert isinstance(t3.op, TensorGreaterThan)


def test_frexp():
    t1 = ones((3, 4, 5), chunk_size=2)
    t2 = empty((3, 4, 5), dtype=np.dtype(float), chunk_size=2)
    op_type = type(t1.op)

    o1, o2 = frexp(t1)

    assert o1.op is o2.op
    assert o1.dtype != o2.dtype

    o1, o2 = frexp(t1, t1)

    assert o1 is t1
    assert o1.inputs[0] is not t1
    assert isinstance(o1.inputs[0].op, op_type)
    assert o2.inputs[0] is not t1

    o1, o2 = frexp(t1, t2, where=t1 > 0)

    op_type = type(t2.op)
    assert o1 is t2
    assert o1.inputs[0] is not t1
    assert isinstance(o1.inputs[0].op, op_type)
    assert o2.inputs[0] is not t1


def test_frexp_order():
    raw1 = np.asfortranarray(np.random.rand(2, 4))
    t = tensor(raw1)
    o1 = tensor(np.random.rand(2, 4))

    o1, o2 = frexp(t, out1=o1)

    assert (
        o1.flags["C_CONTIGUOUS"]
        == np.frexp(raw1, np.empty((2, 4)))[0].flags["C_CONTIGUOUS"]
    )
    assert (
        o1.flags["F_CONTIGUOUS"]
        == np.frexp(raw1, np.empty((2, 4)))[0].flags["F_CONTIGUOUS"]
    )
    assert o2.flags["C_CONTIGUOUS"] == np.frexp(raw1)[1].flags["C_CONTIGUOUS"]
    assert o2.flags["F_CONTIGUOUS"] == np.frexp(raw1)[1].flags["F_CONTIGUOUS"]


def test_dtype():
    t1 = ones((2, 3), dtype="f4", chunk_size=2)

    t = truediv(t1, 2, dtype="f8")

    assert t.dtype == np.float64

    with pytest.raises(TypeError):
        truediv(t1, 2, dtype="i4")


def test_negative():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = negative(t1)
    assert t.op.gpu is None
    assert t.issparse() is True
    assert type(t) is SparseTensor


def test_negative_order():
    raw1 = np.random.rand(4, 2)
    raw2 = np.asfortranarray(np.random.rand(4, 2))
    t1 = tensor(raw1)
    t2 = tensor(raw2)
    t3 = tensor(raw1)
    t4 = tensor(raw2)

    # C
    assert negative(t1).flags["C_CONTIGUOUS"] == np.negative(raw1).flags["C_CONTIGUOUS"]
    assert negative(t1).flags["F_CONTIGUOUS"] == np.negative(raw1).flags["F_CONTIGUOUS"]
    # F
    assert negative(t2).flags["C_CONTIGUOUS"] == np.negative(raw2).flags["C_CONTIGUOUS"]
    assert negative(t2).flags["F_CONTIGUOUS"] == np.negative(raw2).flags["F_CONTIGUOUS"]
    # C + out
    assert (
        negative(t1, out=t4).flags["C_CONTIGUOUS"]
        == np.negative(raw1, out=np.empty((4, 2), order="F")).flags["C_CONTIGUOUS"]
    )
    assert (
        negative(t1, out=t4).flags["F_CONTIGUOUS"]
        == np.negative(raw1, out=np.empty((4, 2), order="F")).flags["F_CONTIGUOUS"]
    )
    # F + out
    assert (
        negative(t2, out=t3).flags["C_CONTIGUOUS"]
        == np.negative(raw1, out=np.empty((4, 2), order="C")).flags["C_CONTIGUOUS"]
    )
    assert (
        negative(t2, out=t3).flags["F_CONTIGUOUS"]
        == np.negative(raw1, out=np.empty((4, 2), order="C")).flags["F_CONTIGUOUS"]
    )

    with pytest.raises(TypeError):
        negative(t1, order="B")


def test_cos():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = cos(t1)
    assert t.issparse() is False
    assert type(t) is Tensor


def test_around():
    t1 = ones((2, 3), dtype="f4", chunk_size=2)

    t = around(t1, decimals=3)

    assert t.issparse() is False
    assert t.op.decimals == 3


def test_isclose():
    t1 = ones((2, 3), dtype="f4", chunk_size=2)

    atol = 1e-4
    rtol = 1e-5
    equal_nan = True

    t = isclose(t1, 2, atol=atol, rtol=rtol, equal_nan=equal_nan)

    assert isinstance(t.op, TensorIsclose)
    assert t.op.atol == atol
    assert t.op.rtol == rtol
    assert t.op.equal_nan == equal_nan

    t1 = ones((2, 3), dtype="f4", chunk_size=2)
    t2 = ones((2, 3), dtype="f4", chunk_size=2)

    atol = 1e-4
    rtol = 1e-5
    equal_nan = True

    t = isclose(t1, t2, atol=atol, rtol=rtol, equal_nan=equal_nan)

    assert isinstance(t.op, TensorIsclose)
    assert t.op.atol == atol
    assert t.op.rtol == rtol
    assert t.op.equal_nan == equal_nan


def test_get_set_real():
    a_data = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    a = tensor(a_data, chunk_size=2)

    with pytest.raises(ValueError):
        a.real = [2, 4]


def test_build_mode():
    t1 = ones((2, 3), chunk_size=2)
    assert t1 == 2

    with enter_mode(build=True):
        assert t1 != 2


def test_unary_op_func_name():
    # make sure all the unary op has defined the func name.

    results = collect_leaf_operators(TensorUnaryOp)
    for op_type in results:
        assert hasattr(op_type, "_func_name")


def test_binary_op_func_name():
    # make sure all the binary op has defined the func name.

    results = collect_leaf_operators(TensorBinOp)
    for op_type in results:
        if op_type not in (TensorSetImag, TensorSetReal):
            assert hasattr(op_type, "_func_name")


def test_tree_arithmetic():
    raws = [np.random.rand(10, 10) for _ in range(10)]
    tensors = [tensor(a, chunk_size=3) for a in raws]

    t = tree_add(*tensors, combine_size=4)
    assert isinstance(t.op, TensorTreeAdd)
    assert t.issparse() is False
    assert len(t.inputs) == 3
    assert len(t.inputs[0].inputs) == 4
    assert len(t.inputs[-1].inputs) == 2

    t = tree_multiply(*tensors, combine_size=4)
    assert isinstance(t.op, TensorTreeMultiply)
    assert t.issparse() is False
    assert len(t.inputs) == 3
    assert len(t.inputs[0].inputs) == 4
    assert len(t.inputs[-1].inputs) == 2

    raws = [sps.random(5, 9, density=0.1) for _ in range(10)]
    tensors = [tensor(a, chunk_size=3) for a in raws]

    t = tree_add(*tensors, combine_size=4)
    assert isinstance(t.op, TensorTreeAdd)
    assert t.issparse() is True
    assert len(t.inputs) == 3
    assert len(t.inputs[0].inputs) == 4
    assert len(t.inputs[-1].inputs) == 2

    t = tree_multiply(*tensors, combine_size=4)
    assert isinstance(t.op, TensorTreeMultiply)
    assert t.issparse() is True
    assert len(t.inputs) == 3
    assert len(t.inputs[0].inputs) == 4
    assert len(t.inputs[-1].inputs) == 2
