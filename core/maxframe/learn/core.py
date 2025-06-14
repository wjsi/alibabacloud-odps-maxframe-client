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

import functools
import importlib
import inspect
from typing import Type

from ..core import ENTITY_TYPE
from ..core.entity.objects import Object, ObjectData
from ..core.operator import TileableOperatorMixin
from ..session import execute as execute_tileables
from ..session import fetch as fetch_tileables

try:
    from sklearn.base import BaseEstimator as SkBaseEstimator
except ImportError:
    SkBaseEstimator = object


class ModelData(ObjectData):
    pass


class Model(Object):
    pass


MODEL_TYPE = (Model, ModelData)


class LearnOperatorMixin(TileableOperatorMixin):
    _op_module_ = "learn"


@functools.lru_cache(100)
def _get_sklearn_estimator_cls(estimator_cls: Type["BaseEstimator"]):
    mod_path = estimator_cls.__module__.replace("maxframe.learn", "sklearn").split(".")
    mod_path = ".".join(p for p in mod_path if not p.startswith("_"))

    exc = ValueError
    while mod_path.startswith("sklearn."):
        try:
            mod = importlib.import_module(mod_path)
            return getattr(mod, estimator_cls.__name__)
        except (AttributeError, ImportError) as ex:
            exc = ex
            mod_path = mod_path.rsplit(".", 1)[0]
    raise exc


class BaseEstimator(SkBaseEstimator):
    _data_attributes = []

    def _get_data_attributes(self):
        return self._data_attributes or [
            attr
            for attr in dir(self)
            if not attr.startswith("_") and attr.endswith("_")
        ]

    def _get_sklearn_cls(self):
        return _get_sklearn_estimator_cls(type(self))

    def _validate_data(
        self, X, y=None, reset=True, validate_separately=False, **check_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default=None
            The targets. If None, `check_array` is called on `X` and
            `check_X_y` is called otherwise.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

        Returns
        -------
        out : tensor or tuple of these
            The validated input. A tuple is returned if `y` is not None.
        """
        from .utils.validation import check_array, check_X_y

        if y is None:
            if hasattr(self, "_get_tags") and self._get_tags().get(
                "requires_y", False
            ):  # pragma: no cover
                raise ValueError(
                    f"This {type(self).__name__} estimator requires y to be passed, "
                    "but the target y is None."
                )
            X = check_array(X, **check_params)
            out = X
        elif isinstance(y, str) and y == "no_validation":
            X = check_array(X, **check_params)
            out = X
        else:  # pragma: no cover
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = check_array(X, **check_X_params)
                y = check_array(y, **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get("ensure_2d", True) and hasattr(self, "_check_n_features"):
            self._check_n_features(X, reset=reset)

        return out

    def execute(self, session=None, run_kwargs=None, extra_tileables=None):
        from .utils.validation import check_is_fitted

        check_is_fitted(self)

        if extra_tileables is not None and not isinstance(
            extra_tileables, (list, tuple)
        ):
            extra_tileables = [extra_tileables]
        extra_tileables = list(extra_tileables or [])

        attrs = [getattr(self, attr, None) for attr in self._get_data_attributes()]
        attrs = [a for a in attrs + extra_tileables if isinstance(a, ENTITY_TYPE)]
        execute_tileables(*attrs, session=session, run_kwargs=run_kwargs)
        return self

    def fetch(self, session=None, run_kwargs=None):
        from .utils.validation import check_is_fitted

        check_is_fitted(self)

        regressor_cls = self._get_sklearn_cls()
        cls_init_args = inspect.getfullargspec(regressor_cls.__init__)
        cls_args = cls_init_args.args[1:] + cls_init_args.kwonlyargs
        init_kw = {k: getattr(self, k, None) for k in cls_args}
        init_kw = {k: v for k, v in init_kw.items() if v is not None}
        regressor = regressor_cls(**init_kw)

        attrs = [
            (attr, getattr(self, attr, None)) for attr in self._get_data_attributes()
        ]
        attrs = [tp for tp in attrs if tp[-1] is not None]
        ent_attrs = [tp for tp in attrs if isinstance(tp[-1], ENTITY_TYPE)]
        ent_attr_keys, ent_attr_vals = [list(x) for x in zip(*ent_attrs)]

        ent_attr_vals = fetch_tileables(
            *ent_attr_vals, session=session, run_kwargs=run_kwargs
        )
        if len(ent_attr_keys) == 1:
            ent_attr_vals = (ent_attr_vals,)

        attr_dict = dict(attrs)
        attr_dict.update(zip(ent_attr_keys, ent_attr_vals))
        for k, v in attr_dict.items():
            setattr(regressor, k, v)
        return regressor


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class RegressorMixin:
    """Mixin class for all regression estimators in scikit-learn."""

    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination :math:`R^2` of the
        prediction.

        The coefficient :math:`R^2` is defined as :math:`(1 - \\frac{u}{v})`,
        where :math:`u` is the residual sum of squares ``((y_true - y_pred)
        ** 2).sum()`` and :math:`v` is the total sum of squares ``((y_true -
        y_true.mean()) ** 2).sum()``. The best possible score is 1.0 and it
        can be negative (because the model can be arbitrarily worse). A
        constant model that always predicts the expected value of `y`,
        disregarding the input features, would get a :math:`R^2` score of
        0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : Tensor
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.

        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """

        from .metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):  # noqa: R0201  # pylint: disable=no-self-use
        return {"requires_y": True}
