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
import pandas as pd
import pyarrow as pa
import pytest
from odps import types as odps_types

from .... import dataframe as md
from .... import tensor as mt
from ....core import OutputType
from ....lib.dtypes_extension import ArrowDtype, dict_, list_
from ....utils import pd_release_version
from ..schema import (
    arrow_schema_to_odps_schema,
    build_dataframe_table_meta,
    build_table_column_name,
    odps_schema_to_arrow_schema,
    odps_schema_to_pandas_dtypes,
    pandas_dtypes_to_arrow_schema,
    pandas_to_odps_schema,
    pandas_types_to_arrow_schema,
)


def _wrap_maxframe_obj(obj, wrap="no"):
    if wrap == "no":
        return obj
    if isinstance(obj, pd.DataFrame):
        obj = md.DataFrame(obj)
    elif isinstance(obj, pd.Series):
        obj = md.Series(obj)
    elif isinstance(obj, pd.Index):
        obj = md.Index(obj)
    elif isinstance(obj, np.ndarray):
        obj = mt.tensor(obj)
    else:
        obj = mt.scalar(obj)
    if wrap == "data":
        return obj.data
    return obj


@pytest.mark.parametrize("wrap_obj", ["no", "yes", "data"])
def test_pandas_to_odps_schema_dataframe(wrap_obj):
    data = pd.DataFrame(np.random.rand(100, 5), columns=list("ABCDE"))

    test_df = _wrap_maxframe_obj(data, wrap=wrap_obj)
    schema, meta = pandas_to_odps_schema(test_df)
    assert [c.name for c in schema.columns] == ["_idx_0"] + list(
        test_df.dtypes.index.str.lower()
    )
    assert [c.type.name for c in schema.columns] == ["bigint"] + ["double"] * len(
        test_df.columns
    )
    assert meta.type == OutputType.dataframe
    assert meta.table_column_names == list(test_df.dtypes.index.str.lower())
    assert meta.table_index_column_names == ["_idx_0"]
    assert meta.pd_column_level_names == [None]
    assert meta.pd_index_level_names == [None]

    test_df = _wrap_maxframe_obj(data, wrap=wrap_obj)
    schema, meta = pandas_to_odps_schema(test_df, ignore_index=True)
    assert [c.name for c in schema.columns] == list(test_df.dtypes.index.str.lower())
    assert [c.type.name for c in schema.columns] == ["double"] * len(test_df.columns)
    assert meta.type == OutputType.dataframe
    assert meta.table_column_names == list(test_df.dtypes.index.str.lower())
    assert meta.table_index_column_names == []
    assert meta.pd_column_level_names == [None]
    assert meta.pd_index_level_names == []

    data.columns = pd.MultiIndex.from_tuples(
        [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B")], names=["c1", "c2"]
    )
    data.index = pd.MultiIndex.from_arrays(
        [np.random.choice(list("ABC"), 100), np.random.randint(0, 10, 100)]
    )
    test_df = _wrap_maxframe_obj(data, wrap=wrap_obj)
    schema, meta = pandas_to_odps_schema(test_df, unknown_as_string=True)
    assert [c.name for c in schema.columns] == ["_idx_0", "_idx_1"] + list(
        test_df.dtypes.index.map("_".join).str.lower()
    )
    assert [c.type.name for c in schema.columns] == ["string", "bigint"] + [
        "double"
    ] * len(test_df.columns)
    assert meta.type == OutputType.dataframe
    assert meta.table_column_names == list(
        test_df.dtypes.index.map("_".join).str.lower()
    )
    assert meta.table_index_column_names == ["_idx_0", "_idx_1"]
    assert meta.pd_column_level_names == ["c1", "c2"]
    assert meta.pd_index_level_names == [None, None]


@pytest.mark.parametrize("wrap_obj", ["no", "yes", "data"])
def test_pandas_to_odps_schema_series(wrap_obj):
    data = pd.Series(np.random.rand(100))

    test_s = _wrap_maxframe_obj(data, wrap=wrap_obj)
    schema, meta = pandas_to_odps_schema(test_s)
    assert [c.name for c in schema.columns] == ["_idx_0", "_data"]
    assert [c.type.name for c in schema.columns] == ["bigint", "double"]
    assert meta.type == OutputType.series
    assert meta.table_column_names == ["_data"]
    assert meta.table_index_column_names == ["_idx_0"]
    assert meta.pd_column_names == [None]
    assert meta.pd_column_level_names == [None]
    assert meta.pd_index_level_names == [None]

    schema, meta = pandas_to_odps_schema(test_s, ignore_index=True)
    assert [c.name for c in schema.columns] == ["_data"]
    assert [c.type.name for c in schema.columns] == ["double"]
    assert meta.type == OutputType.series
    assert meta.table_column_names == ["_data"]
    assert meta.table_index_column_names == []
    assert meta.pd_column_level_names == [None]
    assert meta.pd_index_level_names == []

    data.index = pd.MultiIndex.from_arrays(
        [np.random.choice(list("ABC"), 100), np.random.randint(0, 10, 100)],
        names=["c1", "c2"],
    )
    data.name = "col_name"
    test_s = _wrap_maxframe_obj(data, wrap=wrap_obj)
    schema, meta = pandas_to_odps_schema(test_s, unknown_as_string=True)
    assert [c.name for c in schema.columns] == ["_idx_0", "_idx_1", "col_name"]
    assert [c.type.name for c in schema.columns] == ["string", "bigint", "double"]
    assert meta.type == OutputType.series
    assert meta.table_column_names == ["col_name"]
    assert meta.table_index_column_names == ["_idx_0", "_idx_1"]
    assert meta.pd_column_names == ["col_name"]
    assert meta.pd_column_level_names == [None]
    assert meta.pd_index_level_names == ["c1", "c2"]


@pytest.mark.parametrize("wrap_obj", ["no", "yes", "data"])
def test_pandas_to_odps_schema_index(wrap_obj):
    data = pd.Index(np.random.randint(0, 100, 100))

    test_idx = _wrap_maxframe_obj(data, wrap=wrap_obj)
    for ignore_idx in (False, True):
        schema, meta = pandas_to_odps_schema(
            test_idx, unknown_as_string=True, ignore_index=ignore_idx
        )
        assert [c.name for c in schema.columns] == ["_idx_0"]
        assert [c.type.name for c in schema.columns] == ["bigint"]
        assert meta.type == OutputType.index
        assert meta.table_column_names == []
        assert meta.table_index_column_names == ["_idx_0"]
        assert meta.pd_column_level_names == []
        assert meta.pd_index_level_names == [None]

    data = pd.MultiIndex.from_arrays(
        [np.random.choice(list("ABC"), 100), np.random.randint(0, 10, 100)],
        names=["c1", "c2"],
    )
    test_idx = _wrap_maxframe_obj(data, wrap=wrap_obj)
    schema, meta = pandas_to_odps_schema(test_idx, unknown_as_string=True)
    assert [c.name for c in schema.columns] == ["_idx_0", "_idx_1"]
    assert [c.type.name for c in schema.columns] == ["string", "bigint"]
    assert meta.type == OutputType.index
    assert meta.table_column_names == []
    assert meta.table_index_column_names == ["_idx_0", "_idx_1"]
    assert meta.pd_column_level_names == []
    assert meta.pd_index_level_names == ["c1", "c2"]


@pytest.mark.parametrize("wrap_obj", ["no", "yes", "data"])
def test_pandas_to_odps_schema_scalar(wrap_obj):
    data = 1234.56

    test_scalar = _wrap_maxframe_obj(data, wrap=wrap_obj)
    if wrap_obj != "no":
        test_scalar.op.data = None

    schema, meta = pandas_to_odps_schema(test_scalar, unknown_as_string=True)
    assert schema.columns[0].name == "_idx_0"
    assert schema.columns[0].type.name == "double"
    assert meta.type == OutputType.scalar
    assert meta.table_column_names == []
    assert meta.table_index_column_names == ["_idx_0"]
    assert meta.pd_column_level_names == []
    assert meta.pd_index_level_names == [None]


@pytest.mark.parametrize("wrap_obj", ["no", "yes", "data"])
def test_pandas_to_odps_schema_tensor(wrap_obj):
    data = np.array([1, 2, 3])

    test_tensor = _wrap_maxframe_obj(data, wrap=wrap_obj)
    if wrap_obj != "no":
        test_tensor.op.data = None

    schema, meta = pandas_to_odps_schema(test_tensor, unknown_as_string=True)
    assert schema.columns[0].name == "_idx_0"
    assert schema.columns[0].type.name == "bigint"
    assert meta.type == OutputType.tensor
    assert meta.table_column_names == []
    assert meta.table_index_column_names == ["_idx_0"]
    assert meta.pd_column_level_names == []
    assert meta.pd_index_level_names == [None]


def test_odps_arrow_schema_conversion():
    odps_schema = odps_types.OdpsSchema(
        [
            odps_types.Column("col1", "string"),
            odps_types.Column("col2", "binary"),
            odps_types.Column("col3", "tinyint"),
            odps_types.Column("col4", "smallint"),
            odps_types.Column("col5", "int"),
            odps_types.Column("col6", "bigint"),
            odps_types.Column("col7", "boolean"),
            odps_types.Column("col8", "float"),
            odps_types.Column("col9", "double"),
            odps_types.Column("col10", "date"),
            odps_types.Column("col11", "datetime"),
            odps_types.Column("col12", "timestamp"),
            odps_types.Column("col13", "decimal(10, 2)"),
            odps_types.Column("col14", "array<string>"),
            odps_types.Column("col15", "map<string, bigint>"),
            odps_types.Column("col16", "struct<a1: string, a2: map<string, bigint>>"),
            odps_types.Column("col17", "CHAR(15)"),
            odps_types.Column("col18", "VARCHAR(15)"),
            odps_types.Column("col19", "decimal"),
        ]
    )
    arrow_schema = odps_schema_to_arrow_schema(odps_schema)
    assert arrow_schema.names == [f"col{i}" for i in range(1, 20)]
    assert arrow_schema.types == [
        pa.string(),
        pa.binary(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.bool_(),
        pa.float32(),
        pa.float64(),
        pa.date32(),
        pa.timestamp("ms"),
        pa.timestamp("ns"),
        pa.decimal128(10, 2),
        pa.list_(pa.string()),
        pa.map_(pa.string(), pa.int64()),
        pa.struct([("a1", pa.string()), ("a2", pa.map_(pa.string(), pa.int64()))]),
        pa.string(),
        pa.string(),
        pa.decimal128(38, 18),
    ]

    expected_odps_schema = odps_types.OdpsSchema(
        [
            odps_types.Column("col1", "string"),
            odps_types.Column("col2", "binary"),
            odps_types.Column("col3", "tinyint"),
            odps_types.Column("col4", "smallint"),
            odps_types.Column("col5", "int"),
            odps_types.Column("col6", "bigint"),
            odps_types.Column("col7", "boolean"),
            odps_types.Column("col8", "float"),
            odps_types.Column("col9", "double"),
            odps_types.Column("col10", "date"),
            odps_types.Column("col11", "datetime"),
            odps_types.Column("col12", "timestamp"),
            odps_types.Column("col13", "decimal(10, 2)"),
            odps_types.Column("col14", "array<string>"),
            odps_types.Column("col15", "map<string, bigint>"),
            odps_types.Column("col16", "struct<a1: string, a2: map<string, bigint>>"),
            odps_types.Column("col17", "string"),
            odps_types.Column("col18", "string"),
            odps_types.Column("col19", "decimal(38, 18)"),
        ]
    )

    odps_schema2 = arrow_schema_to_odps_schema(arrow_schema)
    assert [c.name for c in expected_odps_schema.columns] == [
        c.name for c in odps_schema2.columns
    ]
    assert [c.type for c in expected_odps_schema.columns] == [
        c.type for c in odps_schema2.columns
    ]

    with pytest.raises(TypeError):
        arrow_schema_to_odps_schema(pa.schema([("col1", pa.float16())]))


def test_odps_pandas_schema_conversion():
    odps_schema = odps_types.OdpsSchema(
        [
            odps_types.Column("col1", "string"),
            odps_types.Column("col2", "binary"),
            odps_types.Column("col3", "tinyint"),
            odps_types.Column("col4", "smallint"),
            odps_types.Column("col5", "int"),
            odps_types.Column("col6", "bigint"),
            odps_types.Column("col7", "boolean"),
            odps_types.Column("col8", "float"),
            odps_types.Column("col9", "double"),
            # odps_types.Column("col10", "date"),
            odps_types.Column("col11", "datetime"),
            odps_types.Column("col12", "timestamp"),
            # odps_types.Column("col13", "decimal(10, 2)"),
            odps_types.Column("col14", "array<string>"),
            odps_types.Column("col15", "map<string, bigint>"),
            # odps_types.Column("col16", "struct<a1: string, a2: map<string, bigint>>"),
            # odps_types.Column("col17", "CHAR(15)"),
            # odps_types.Column("col18", "VARCHAR(15)"),
            # odps_types.Column("col19", "decimal"),
        ]
    )
    pd_dtypes = odps_schema_to_pandas_dtypes(odps_schema)
    pd.testing.assert_series_equal(
        pd_dtypes,
        pd.Series(
            [
                np.dtype("O"),  # string
                np.dtype("O"),  # binary
                np.dtype(np.int8),
                np.dtype(np.int16),
                np.dtype(np.int32),
                np.dtype(np.int64),
                np.dtype(np.bool_),
                np.dtype(np.float32),
                np.dtype(np.float64),
                np.dtype(
                    "datetime64[ms]" if pd_release_version[0] >= 2 else "datetime64[ns]"
                ),
                np.dtype("datetime64[ns]"),
                ArrowDtype(pa.list_(pa.string())),
                ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ],
            index=[c.name for c in odps_schema.columns],
        ),
    )

    expected_odps_schema = odps_types.OdpsSchema(
        [
            odps_types.Column("col1", "string"),
            odps_types.Column("col2", "string"),  # binary
            odps_types.Column("col3", "tinyint"),
            odps_types.Column("col4", "smallint"),
            odps_types.Column("col5", "int"),
            odps_types.Column("col6", "bigint"),
            odps_types.Column("col7", "boolean"),
            odps_types.Column("col8", "float"),
            odps_types.Column("col9", "double"),
            # odps_types.Column("col10", "date"),
            odps_types.Column(
                "col11", "datetime" if pd_release_version[0] >= 2 else "timestamp"
            ),
            odps_types.Column("col12", "timestamp"),
            # odps_types.Column("col13", "decimal(10, 2)"),
            odps_types.Column("col14", "array<string>"),
            odps_types.Column("col15", "map<string, bigint>"),
            # odps_types.Column("col16", "struct<a1: string, a2: map<string, bigint>>"),
            # odps_types.Column("col17", "string"),
            # odps_types.Column("col18", "string"),
            # odps_types.Column("col19", "decimal(38, 18)"),
        ]
    )

    odps_schema2 = arrow_schema_to_odps_schema(
        pandas_dtypes_to_arrow_schema(pd_dtypes, unknown_as_string=True)
    )
    assert [c.name for c in expected_odps_schema.columns] == [
        c.name for c in odps_schema2.columns
    ]
    assert [c.type for c in expected_odps_schema.columns] == [
        c.type for c in odps_schema2.columns
    ]

    with pytest.raises(TypeError):
        arrow_schema_to_odps_schema(pa.schema([("col1", pa.float16())]))


def test_build_column_name():
    records = dict()
    assert build_table_column_name(0, "a" * 127, records) == "a" * 127
    assert build_table_column_name(1, "_abc123", records) == "_abc123"
    assert build_table_column_name(2, "_abc'123", records) == "_column_2"
    assert build_table_column_name(3, "a" * 256, records) == "_column_3"
    assert build_table_column_name(4, ("A", 1), records) == "a_1"


@pytest.mark.parametrize("wrap_obj", ["no", "yes", "data"])
def test_build_table_meta(wrap_obj):
    data = pd.DataFrame(
        np.random.rand(100, 7),
        columns=["A", "A", "A_0", "A_1", "a_1", "B", "C"],
    )

    test_df = _wrap_maxframe_obj(data, wrap=wrap_obj)
    table_meta = build_dataframe_table_meta(test_df)
    expected_cols = ["a_2", "a_3", "a_0", "a_1_0", "a_1_1", "b", "c"]
    assert table_meta.table_column_names == expected_cols


@pytest.mark.skipif(
    pd_release_version[0] < 2, reason="only run under pandas 2.0 or greater"
)
def test_table_meta_with_datetime():
    raw_df = pd.DataFrame(
        [
            [1, "abc", "2024-10-01 11:23:12"],
            [3, "uvw", "2024-10-02 22:55:13"],
        ],
        columns=["col1", "col2", "col3"],
    )
    df = md.DataFrame(raw_df).astype({"col3": "datetime64[ms]"})
    schema, _ = pandas_to_odps_schema(df, unknown_as_string=True)
    assert schema.columns[3].type == odps_types.datetime

    raw_series = pd.Series(
        ["2024-10-01 11:23:12", "2024-10-02 22:55:13"], dtype="datetime64[ms]"
    )
    s = md.Series(raw_series)
    schema, _ = pandas_to_odps_schema(s, unknown_as_string=True)
    assert schema.columns[1].type == odps_types.datetime

    raw_index = pd.Index(
        ["2024-10-01 11:23:12", "2024-10-02 22:55:13"], dtype="datetime64[ms]"
    )
    idx = md.Index(raw_index)
    schema, _ = pandas_to_odps_schema(idx, unknown_as_string=True)
    assert schema.columns[0].type == odps_types.datetime

    src_df = pd.DataFrame(
        [[1, "2024-10-01 11:23:12"], [3, "2024-10-02 22:55:13"]],
        columns=["A", "B"],
    ).astype({"B": "datetime64[ms]"})
    raw_multiindex = pd.MultiIndex.from_frame(src_df)
    multiidx = md.Index(raw_multiindex)
    schema, _ = pandas_to_odps_schema(multiidx, unknown_as_string=True)
    assert schema.columns[1].type == odps_types.datetime


@pytest.mark.skipif(
    pa is None or not hasattr(pd, "ArrowDtype"),
    reason="pandas doesn't support ArrowDtype",
)
def test_pandas_types_to_arrow_schema():
    pd_data = pd.DataFrame(
        {
            "int8": pd.Series([], dtype=np.int8),
            "map": pd.Series([], dtype=dict_(pa.string(), pa.string())),
            "list": pd.Series([], dtype=list_(pa.string())),
        },
    )
    schema = pandas_types_to_arrow_schema(pd_data)
    assert schema.field("int8").type == pa.int8()
    assert schema.field("map").type == pa.map_(pa.string(), pa.string())
    assert schema.field("list").type == pa.list_(pa.string())
