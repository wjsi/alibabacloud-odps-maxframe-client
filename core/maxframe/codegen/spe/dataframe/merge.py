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

from typing import List

from ....dataframe.merge import DataFrameConcat, DataFrameMerge
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(DataFrameMerge)
class DataFrameMergeAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameMerge, context: SPECodeContext) -> List[str]:
        left_var = context.get_input_tileable_variable(op.inputs[0])
        right_var = context.get_input_tileable_variable(op.inputs[1])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kwargs = {"how": op.how}
        if op.how != "cross":
            if op.on is not None:
                kwargs["on"] = op.on
            else:
                if op.left_index:
                    kwargs["left_index"] = True
                else:
                    kwargs["left_on"] = op.left_on

                if op.right_index:
                    kwargs["right_index"] = True
                else:
                    kwargs["right_on"] = op.right_on
        kwargs["sort"] = op.sort
        kwargs["suffixes"] = op.suffixes
        kwargs["indicator"] = op.indicator
        kwargs["validate"] = op.validate
        kwargs["copy"] = op.copy_
        args = ", ".join(self._translate_call_args(context, **kwargs))
        context.register_import("pandas", "pd")
        return [f"{res_var_name} = pd.merge({left_var}, {right_var}, {args})"]


@register_op_adapter(DataFrameConcat)
class DataFrameConcatAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameConcat, context: SPECodeContext) -> List[str]:
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kw_keys = [
            "axis",
            "join",
            "ignore_index",
            "keys",
            "levels",
            "names",
            "verify_integrity",
            "sort",
        ]
        args = self.generate_call_args_with_attributes(
            op,
            context,
            kw_keys=kw_keys,
            inputs="objs",
            copy_="copy",
        )
        context.register_import("pandas", "pd")
        return [f"{res_var_name} = pd.concat({args})"]
