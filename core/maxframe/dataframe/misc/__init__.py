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

from .apply import df_apply, series_apply
from .astype import astype, index_astype
from .case_when import case_when
from .check_monotonic import (
    check_monotonic,
    is_monotonic,
    is_monotonic_decreasing,
    is_monotonic_increasing,
)
from .cut import cut
from .describe import describe
from .diff import df_diff, series_diff
from .drop import df_drop, df_pop, index_drop, series_drop
from .drop_duplicates import (
    df_drop_duplicates,
    index_drop_duplicates,
    series_drop_duplicates,
)
from .duplicated import df_duplicated, index_duplicated, series_duplicated
from .eval import df_eval, df_query
from .explode import df_explode, series_explode
from .isin import df_isin, series_isin
from .map import index_map, series_map
from .melt import melt
from .memory_usage import df_memory_usage, index_memory_usage, series_memory_usage
from .pct_change import pct_change
from .pivot import pivot
from .pivot_table import pivot_table
from .qcut import qcut
from .rechunk import rechunk
from .select_dtypes import select_dtypes
from .shift import shift, tshift
from .stack import stack
from .transform import df_transform, series_transform
from .transpose import transpose
from .value_counts import value_counts


def _install():
    from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE

    for t in DATAFRAME_TYPE:
        setattr(t, "apply", df_apply)
        setattr(t, "astype", astype)
        setattr(t, "describe", describe)
        setattr(
            t, "__delitem__", lambda df, items: df_drop(df, items, axis=1, inplace=True)
        )
        setattr(t, "diff", df_diff)
        setattr(t, "drop_duplicates", df_drop_duplicates)
        setattr(t, "duplicated", df_duplicated)
        setattr(t, "drop", df_drop)
        setattr(t, "eval", df_eval)
        setattr(t, "explode", df_explode)
        setattr(t, "isin", df_isin)
        setattr(t, "melt", melt)
        setattr(t, "memory_usage", df_memory_usage)
        setattr(t, "pct_change", pct_change)
        setattr(t, "pivot", pivot)
        setattr(t, "pivot_table", pivot_table)
        setattr(t, "pop", df_pop)
        setattr(t, "query", df_query)
        setattr(t, "rechunk", rechunk)
        setattr(t, "select_dtypes", select_dtypes)
        setattr(t, "shift", shift)
        setattr(t, "stack", stack)
        setattr(t, "transform", df_transform)
        setattr(t, "transpose", transpose)
        setattr(t, "tshift", tshift)

    for t in SERIES_TYPE:
        setattr(t, "apply", series_apply)
        setattr(t, "astype", astype)
        setattr(t, "case_when", case_when)
        setattr(t, "check_monotonic", check_monotonic)
        setattr(t, "describe", describe)
        setattr(t, "diff", series_diff)
        setattr(t, "drop", series_drop)
        setattr(t, "drop_duplicates", series_drop_duplicates)
        setattr(t, "duplicated", series_duplicated)
        setattr(t, "explode", series_explode)
        setattr(t, "is_monotonic", property(fget=is_monotonic))
        setattr(t, "is_monotonic_decreasing", property(fget=is_monotonic_decreasing))
        setattr(t, "is_monotonic_increasing", property(fget=is_monotonic_increasing))
        setattr(t, "isin", series_isin)
        setattr(t, "map", series_map)
        setattr(t, "memory_usage", series_memory_usage)
        setattr(t, "pct_change", pct_change)
        setattr(t, "rechunk", rechunk)
        setattr(t, "shift", shift)
        setattr(t, "transform", series_transform)
        setattr(t, "tshift", tshift)
        setattr(t, "value_counts", value_counts)

    for t in INDEX_TYPE:
        setattr(t, "astype", index_astype)
        setattr(t, "check_monotonic", check_monotonic)
        setattr(t, "drop", index_drop)
        setattr(t, "drop_duplicates", index_drop_duplicates)
        setattr(t, "duplicated", index_duplicated)
        setattr(t, "is_monotonic", property(fget=is_monotonic))
        setattr(t, "is_monotonic_increasing", property(fget=is_monotonic_increasing))
        setattr(t, "is_monotonic_decreasing", property(fget=is_monotonic_decreasing))
        setattr(t, "map", index_map)
        setattr(t, "memory_usage", index_memory_usage)
        setattr(t, "rechunk", rechunk)
        setattr(t, "value_counts", value_counts)


_install()
del _install
