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

import math
import numbers

import pandas as pd

from ...dataframe import DataFrame, Series
from ...dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ...tensor import tensor as astensor


def convert_to_tensor_or_dataframe(item):
    if isinstance(item, (DATAFRAME_TYPE, pd.DataFrame)):
        item = DataFrame(item)
    elif isinstance(item, (SERIES_TYPE, pd.Series)):
        item = Series(item)
    else:
        item = astensor(item)
    return item


def is_scalar_nan(x):
    """Tests if x is NaN.

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not float('nan').

    Parameters
    ----------
    x : any type

    Returns
    -------
    boolean

    Examples
    --------
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    return isinstance(x, numbers.Real) and math.isnan(x)
