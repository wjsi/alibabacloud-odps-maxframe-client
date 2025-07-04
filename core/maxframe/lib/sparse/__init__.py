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

import builtins
import operator
from collections.abc import Iterable
from functools import partial, reduce

from . import linalg
from .array import SparseNDArray, call_sparse
from .core import get_sparse_module, issparse
from .matrix import SparseMatrix
from .vector import SparseVector


def asarray(x, shape=None):
    from .core import issparse

    if issparse(x):
        return SparseNDArray(x, shape=shape)

    return x


def add(a, b, **_):
    try:
        return a + b
    except TypeError:
        if hasattr(b, "__radd__"):
            return b.__radd__(a)
        raise


def subtract(a, b, **_):
    try:
        return a - b
    except TypeError:
        if hasattr(b, "__rsub__"):
            return b.__rsub__(a)
        raise


def multiply(a, b, **_):
    try:
        return a * b
    except TypeError:
        if hasattr(b, "__rmul__"):
            return b.__rmul__(a)
        raise


def divide(a, b, **_):
    try:
        return a / b
    except TypeError:
        if hasattr(b, "__rdiv__"):
            return b.__rdiv__(a)
        raise


def true_divide(a, b, **_):
    try:
        return a / b
    except TypeError:
        if hasattr(b, "__rtruediv__"):
            return b.__rtruediv__(a)
        raise


def floor_divide(a, b, **_):
    try:
        return a // b
    except TypeError:
        if hasattr(b, "__rfloordiv__"):
            return b.__rfloordiv__(a)
        raise


def power(a, b, **_):
    try:
        return a**b
    except TypeError:
        if hasattr(b, "__rpow__"):
            return b.__rpow__(a)
        raise


def mod(a, b, **_):
    try:
        return a % b
    except TypeError:
        if hasattr(b, "__rmod__"):
            return b.__rmod__(a)
        raise


def _call_bin(method, a, b, **kwargs):
    from .core import cp, get_array_module, issparse

    # order does not take effect for sparse
    kwargs.pop("order", None)
    if hasattr(a, method):
        res = getattr(a, method)(b, **kwargs)
    elif get_array_module(a).isscalar(a):
        res = call_sparse(method, a, b, **kwargs)
    else:
        assert get_array_module(a) == get_array_module(b)
        xp = get_array_module(a)
        try:
            res = getattr(xp, method)(a, b, **kwargs)
        except TypeError:
            if xp is cp and issparse(b):
                res = getattr(xp, method)(a, b.toarray(), **kwargs)
            else:
                raise

    if res is NotImplemented:
        raise NotImplementedError

    return res


def _call_unary(method, x, *args, **kwargs):
    from .core import get_array_module

    # order does not take effect for sparse
    kwargs.pop("order", None)
    if hasattr(x, method):
        res = getattr(x, method)(*args, **kwargs)
    else:
        xp = get_array_module(x)
        res = getattr(xp, method)(x, *args, **kwargs)

    if res is NotImplemented:
        raise NotImplementedError

    return res


def float_power(a, b, **kw):
    return _call_bin("float_power", a, b, **kw)


def fmod(a, b, **kw):
    return _call_bin("fmod", a, b, **kw)


def logaddexp(a, b, **kw):
    return _call_bin("logaddexp", a, b, **kw)


def logaddexp2(a, b, **kw):
    return _call_bin("logaddexp2", a, b, **kw)


def negative(x, **_):
    return -x


def positive(x, **_):
    return operator.pos(x)


def absolute(x, **_):
    return builtins.abs(x)


abs = absolute


fabs = partial(_call_unary, "fabs")


def rint(x, **kw):
    return _call_unary("rint", x, **kw)


def sign(x, **kw):
    return _call_unary("sign", x, **kw)


def conj(x, **kw):
    return _call_unary("conj", x, **kw)


def exp(x, **kw):
    return _call_unary("exp", x, **kw)


def exp2(x, **kw):
    return _call_unary("exp2", x, **kw)


def log(x, **kw):
    return _call_unary("log", x, **kw)


def log2(x, **kw):
    return _call_unary("log2", x, **kw)


def log10(x, **kw):
    return _call_unary("log10", x, **kw)


def expm1(x, **kw):
    return _call_unary("expm1", x, **kw)


def log1p(x, **kw):
    return _call_unary("log1p", x, **kw)


def sqrt(x, **kw):
    return _call_unary("sqrt", x, **kw)


def square(x, **kw):
    return _call_unary("square", x, **kw)


def cbrt(x, **kw):
    return _call_unary("cbrt", x, **kw)


def reciprocal(x, **kw):
    return _call_unary("reciprocal", x, **kw)


gamma = partial(_call_unary, "gamma")
gammaln = partial(_call_unary, "gammaln")
loggamma = partial(_call_unary, "loggamma")
gammasgn = partial(_call_unary, "gammasgn")
gammainc = partial(_call_bin, "gammainc")
gammaincinv = partial(_call_bin, "gammaincinv")
gammaincc = partial(_call_bin, "gammaincc")
gammainccinv = partial(_call_bin, "gammainccinv")
beta = partial(_call_bin, "beta")
betaln = partial(_call_bin, "betaln")
betainc = partial(call_sparse, "betainc")
betaincinv = partial(call_sparse, "betaincinv")
psi = partial(_call_unary, "psi")
rgamma = partial(_call_unary, "rgamma")
polygamma = partial(_call_bin, "polygamma")
multigammaln = partial(_call_bin, "multigammaln")
digamma = partial(_call_unary, "digamma")
poch = partial(_call_bin, "poch")

entr = partial(_call_unary, "entr")
rel_entr = partial(_call_bin, "rel_entr")
kl_div = partial(_call_bin, "kl_div")

xlogy = partial(_call_bin, "xlogy")

erf = partial(_call_unary, "erf")
erfc = partial(_call_unary, "erfc")
erfcx = partial(_call_unary, "erfcx")
erfi = partial(_call_unary, "erfi")
erfinv = partial(_call_unary, "erfinv")
erfcinv = partial(_call_unary, "erfcinv")
wofz = partial(_call_unary, "wofz")
dawsn = partial(_call_unary, "dawsn")
voigt_profile = partial(call_sparse, "voigt_profile")

jv = partial(_call_bin, "jv")
jve = partial(_call_bin, "jve")
yn = partial(_call_bin, "yn")
yv = partial(_call_bin, "yv")
yve = partial(_call_bin, "yve")
kn = partial(_call_bin, "kn")
kv = partial(_call_bin, "kv")
kve = partial(_call_bin, "kve")
iv = partial(_call_bin, "iv")
ive = partial(_call_bin, "ive")
hankel1 = partial(_call_bin, "hankel1")
hankel1e = partial(_call_bin, "hankel1e")
hankel2 = partial(_call_bin, "hankel2")
hankel2e = partial(_call_bin, "hankel2e")

hyp2f1 = partial(call_sparse, "hyp2f1")
hyp1f1 = partial(call_sparse, "hyp1f1")
hyperu = partial(call_sparse, "hyperu")
hyp0f1 = partial(_call_bin, "hyp0f1")

ellip_harm = partial(call_sparse, "ellip_harm")
ellip_harm_2 = partial(call_sparse, "ellip_harm_2")
ellip_normal = partial(call_sparse, "ellip_normal")

ellipk = partial(_call_unary, "ellipk")
ellipkm1 = partial(_call_unary, "ellipkm1")
ellipkinc = partial(_call_bin, "ellipkinc")
ellipe = partial(_call_unary, "ellipe")
ellipeinc = partial(_call_bin, "ellipeinc")
elliprc = partial(_call_bin, "elliprc")
elliprd = partial(call_sparse, "elliprd")
elliprf = partial(call_sparse, "elliprf")
elliprg = partial(call_sparse, "elliprg")
elliprj = partial(call_sparse, "elliprj")

airy = partial(_call_unary, "airy")
airye = partial(_call_unary, "airye")
itairy = partial(_call_unary, "itairy")

expit = partial(call_sparse, "expit")
logit = partial(call_sparse, "logit")
log_expit = partial(call_sparse, "log_expit")

softplus = partial(call_sparse, "softplus")


def equal(a, b, **_):
    try:
        return a == b
    except TypeError:
        return b == a


def not_equal(a, b, **_):
    try:
        return a != b
    except TypeError:
        return b != a


def less(a, b, **_):
    try:
        return a < b
    except TypeError:
        return b > a


def less_equal(a, b, **_):
    try:
        return a <= b
    except TypeError:
        return b >= a


def greater(a, b, **_):
    try:
        return a > b
    except TypeError:
        return b < a


def greater_equal(a, b, **_):
    try:
        return a >= b
    except TypeError:
        return b <= a


def logical_and(a, b, **kw):
    return _call_bin("logical_and", a, b, **kw)


def logical_or(a, b, **kw):
    return _call_bin("logical_or", a, b, **kw)


def logical_xor(a, b, **kw):
    return _call_bin("logical_xor", a, b, **kw)


def logical_not(x, **kw):
    return _call_unary("logical_not", x, **kw)


def isclose(a, b, **kw):
    return _call_bin("isclose", a, b, **kw)


def bitwise_and(a, b, **_):
    try:
        return a & b
    except TypeError:
        return b & a


def bitwise_or(a, b, **_):
    try:
        return a | b
    except TypeError:
        return b | a


def bitwise_xor(a, b, **_):
    try:
        return operator.xor(a, b)
    except TypeError:
        return operator.xor(b, a)


def invert(x, **_):
    return ~x


def left_shift(a, b, **_):
    return a << b


def right_shift(a, b, **_):
    return a >> b


def sin(x, **kw):
    return _call_unary("sin", x, **kw)


def cos(x, **kw):
    return _call_unary("cos", x, **kw)


def tan(x, **kw):
    return _call_unary("tan", x, **kw)


def arcsin(x, **kw):
    return _call_unary("arcsin", x, **kw)


def arccos(x, **kw):
    return _call_unary("arccos", x, **kw)


def arctan(x, **kw):
    return _call_unary("arctan", x, **kw)


def arctan2(a, b, **kw):
    return _call_bin("arctan2", a, b, **kw)


def hypot(a, b, **kw):
    return _call_bin("hypot", a, b, **kw)


def sinh(x, **kw):
    return _call_unary("sinh", x, **kw)


def cosh(x, **kw):
    return _call_unary("cosh", x, **kw)


def tanh(x, **kw):
    return _call_unary("tanh", x, **kw)


def arcsinh(x, **kw):
    return _call_unary("arcsinh", x, **kw)


def arccosh(x, **kw):
    return _call_unary("arccosh", x, **kw)


def around(x, **kw):
    return _call_unary("around", x, **kw)


def arctanh(x, **kw):
    return _call_unary("arctanh", x, **kw)


def deg2rad(x, **kw):
    return _call_unary("deg2rad", x, **kw)


def rad2deg(x, **kw):
    return _call_unary("rad2deg", x, **kw)


def angle(x, **kw):
    return _call_unary("angle", x, **kw)


def isinf(x, **kw):
    return _call_unary("isinf", x, **kw)


def isnan(x, **kw):
    return _call_unary("isnan", x, **kw)


def signbit(x, **kw):
    return _call_unary("signbit", x, **kw)


def dot(a, b, sparse=True, **_):
    from .core import issparse

    if not issparse(a):
        ret = a.dot(b)
        if not sparse:
            return ret
        else:
            xps = get_sparse_module(ret)
            return SparseNDArray(xps.csr_matrix(ret), shape=ret.shape)

    return a.dot(b, sparse=sparse)


def tensordot(a, b, axes=2, sparse=True):
    if isinstance(axes, Iterable):
        a_axes, b_axes = axes
    else:
        a_axes = tuple(range(a.ndim - 1, a.ndim - axes - 1, -1))
        b_axes = tuple(range(0, axes))

    if isinstance(a_axes, Iterable):
        a_axes = tuple(a_axes)
    else:
        a_axes = (a_axes,)
    if isinstance(b_axes, Iterable):
        b_axes = tuple(b_axes)
    else:
        b_axes = (b_axes,)

    if a_axes == (a.ndim - 1,) and b_axes == (b.ndim - 2,):
        return dot(a, b, sparse=sparse)

    if a.ndim == b.ndim == 2:
        if a_axes == (a.ndim - 1,) and b_axes == (b.ndim - 1,):
            # inner product of multiple dims
            return dot(a, b.T, sparse=sparse)

    if a.ndim == 1 or b.ndim == 1:
        return dot(a, b, sparse=sparse)

    raise NotImplementedError


def matmul(a, b, sparse=True, **_):
    return dot(a, b, sparse=sparse)


def concatenate(tensors, axis=0):
    return reduce(lambda a, b: _call_bin("concatenate", a, b, axis=axis), tensors)


def transpose(tensor, axes=None):
    return _call_unary("transpose", tensor, axes=axes)


def swapaxes(tensor, axis1, axis2):
    return _call_unary("swapaxes", tensor, axis1, axis2)


def sum(tensor, axis=None, **kw):
    return _call_unary("sum", tensor, axis=axis, **kw)


def prod(tensor, axis=None, **kw):
    return _call_unary("prod", tensor, axis=axis, **kw)


def amax(tensor, axis=None, **kw):
    return _call_unary("amax", tensor, axis=axis, **kw)


max = amax


def amin(tensor, axis=None, **kw):
    return _call_unary("amin", tensor, axis=axis, **kw)


min = amin


def all(tensor, axis=None, **kw):
    return _call_unary("all", tensor, axis=axis, **kw)


def any(tensor, axis=None, **kw):
    return _call_unary("any", tensor, axis=axis, **kw)


def mean(tensor, axis=None, **kw):
    return _call_unary("mean", tensor, axis=axis, **kw)


def nansum(tensor, axis=None, **kw):
    return _call_unary("nansum", tensor, axis=axis, **kw)


def nanprod(tensor, axis=None, **kw):
    return _call_unary("nanprod", tensor, axis=axis, **kw)


def nanmax(tensor, axis=None, **kw):
    return _call_unary("nanmax", tensor, axis=axis, **kw)


def nanmin(tensor, axis=None, **kw):
    return _call_unary("nanmin", tensor, axis=axis, **kw)


def argmax(tensor, axis=None, **kw):
    return _call_unary("argmax", tensor, axis=axis, **kw)


def nanargmax(tensor, axis=None, **kw):
    return _call_unary("nanargmax", tensor, axis=axis, **kw)


def argmin(tensor, axis=None, **kw):
    return _call_unary("argmin", tensor, axis=axis, **kw)


def nanargmin(tensor, axis=None, **kw):
    return _call_unary("nanargmin", tensor, axis=axis, **kw)


def var(tensor, axis=None, **kw):
    return _call_unary("var", tensor, axis=axis, **kw)


def cumsum(tensor, axis=None, **kw):
    return _call_unary("cumsum", tensor, axis=axis, **kw)


def cumprod(tensor, axis=None, **kw):
    return _call_unary("cumprod", tensor, axis=axis, **kw)


def nancumsum(tensor, axis=None, **kw):
    return _call_unary("nancumsum", tensor, axis=axis, **kw)


def nancumprod(tensor, axis=None, **kw):
    return _call_unary("nancumprod", tensor, axis=axis, **kw)


def count_nonzero(tensor, axis=None, **kw):
    return _call_unary("count_nonzero", tensor, axis=axis, **kw)


def maximum(a, b, **kw):
    return _call_bin("maximum", a, b, **kw)


def minimum(a, b, **kw):
    return _call_bin("minimum", a, b, **kw)


def fmax(a, b, **kw):
    return _call_bin("fmax", a, b, **kw)


def fmin(a, b, **kw):
    return _call_bin("fmin", a, b, **kw)


def floor(x, **kw):
    return _call_unary("floor", x, **kw)


def ceil(x, **kw):
    return _call_unary("ceil", x, **kw)


def trunc(x, **kw):
    return _call_unary("trunc", x, **kw)


def degrees(x, **kw):
    return _call_unary("degrees", x, **kw)


def radians(x, **kw):
    return _call_unary("radians", x, **kw)


def clip(a, a_max, a_min, **kw):
    from .core import get_array_module

    if hasattr(a, "clip"):
        res = getattr(a, "clip")(a_max, a_min, **kw)
    else:
        xp = get_array_module(a)
        res = getattr(xp, "clip")(a, a_max, a_min, **kw)

    if res is NotImplemented:
        raise NotImplementedError

    return res


def iscomplex(x, **kw):
    return _call_unary("iscomplex", x, **kw)


def real(x, **_):
    return x.real


def imag(x, **_):
    return x.imag


def fix(x, **kw):
    return _call_unary("fix", x, **kw)


def i0(x, **kw):
    return _call_unary("i0", x, **kw)


def nan_to_num(x, **kw):
    return _call_unary("nan_to_num", x, **kw)


def copysign(a, b, **kw):
    return _call_bin("copysign", a, b, **kw)


def nextafter(a, b, **kw):
    return _call_bin("nextafter", a, b, **kw)


def spacing(x, **kw):
    return _call_unary("spacing", x, **kw)


def ldexp(a, b, **kw):
    return _call_bin("ldexp", a, b, **kw)


def frexp(x, **kw):
    return _call_unary("frexp", x, **kw)


def modf(x, **kw):
    return _call_unary("modf", x, **kw)


def sinc(x, **kw):
    return _call_unary("sinc", x, **kw)


def isfinite(x, **kw):
    return _call_unary("isfinite", x, **kw)


def isreal(x, **kw):
    return _call_unary("isreal", x, **kw)


def isfortran(x, **kw):
    return call_sparse("isfortran", x, **kw)


def where(cond, x, y):
    if any([i.ndim not in (0, 2) for i in (cond, x, y)]):
        raise NotImplementedError

    from .matrix import where as matrix_where

    return matrix_where(cond, x, y)


def digitize(x, bins, right=False):
    return _call_unary("digitize", x, bins, right)


def repeat(a, repeats, axis=None):
    return _call_unary("repeat", a, repeats, axis=axis)


def fill_diagonal(a, val, wrap=False):
    return _call_unary("fill_diagonal", a, val, wrap=wrap)


def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None):
    return _call_unary(
        "unique",
        a,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    )


def zeros(shape, dtype=float, gpu=False):
    if len(shape) == 2:
        from .matrix import zeros_sparse_matrix

        return zeros_sparse_matrix(shape, dtype=dtype, gpu=gpu)

    raise NotImplementedError


def ones_like(x):
    from .core import get_array_module

    return get_array_module(x).ones(x.shape)


def diag(v, k=0, gpu=False):
    assert v.ndim in {1, 2}

    from .matrix import diag_sparse_matrix

    return diag_sparse_matrix(v, k=k, gpu=gpu)


def eye(N, M=None, k=0, dtype=float, gpu=False):
    from .matrix import eye_sparse_matrix

    return eye_sparse_matrix(N, M=M, k=k, dtype=dtype, gpu=gpu)


def triu(m, k=0, gpu=False):
    if m.ndim == 2:
        from .matrix import triu_sparse_matrix

        return triu_sparse_matrix(m, k=k, gpu=gpu)

    raise NotImplementedError


def tril(m, k=0, gpu=False):
    if m.ndim == 2:
        from .matrix import tril_sparse_matrix

        return tril_sparse_matrix(m, k=k, gpu=gpu)

    raise NotImplementedError


def block(arrs):
    arr = arrs[0]
    while isinstance(arr, list):
        arr = arr[0]
    if arr.ndim == 1:
        return concatenate(arrs)
    elif arr.ndim != 2:  # pragma: no cover
        raise NotImplementedError

    from .matrix import block

    return block(arrs)
