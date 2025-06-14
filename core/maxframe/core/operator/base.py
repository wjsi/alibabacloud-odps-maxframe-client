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

import weakref
from copy import deepcopy
from enum import Enum
from functools import lru_cache, partial
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Type, Union

from ...serialization.core import Placeholder
from ...serialization.serializables import (
    BoolField,
    DictField,
    FieldTypes,
    Float32Field,
    Int32Field,
    ListField,
    ReferenceField,
    Serializable,
    SerializableMeta,
    StringField,
    TupleField,
)
from ...serialization.serializables.core import SerializableSerializer
from ...typing_ import OperatorType
from ...utils import AttributeDict, classproperty, get_user_call_point, tokenize
from ..base import Base
from ..entity.core import ENTITY_TYPE, Entity, EntityData
from ..entity.output_types import OutputType
from ..entity.tileables import Tileable
from ..mode import enter_mode


class OperatorMetaclass(SerializableMeta):
    def __new__(mcs, name: str, bases: Tuple[Type], properties: Dict):
        if "__call__" in properties:
            # if __call__ is specified for an operator,
            # make sure that entering user space
            properties["__call__"] = enter_mode(kernel=False)(properties["__call__"])

        return super().__new__(mcs, name, bases, properties)


class OperatorStage(Enum):
    map = 0
    reduce = 1
    combine = 2
    agg = 3


class SchedulingHint(Serializable):
    # worker to execute, only work for chunk op,
    # if specified, the op should be executed on the specified worker
    # only work for those operator that has no input
    expect_worker = StringField("expect_worker", default=None)
    # band to execute, only work for chunk op,
    # if specified, the op should be executed on the specified band
    # only work for those operator that has no input
    expect_band = TupleField(
        "expect_band",
        FieldTypes.tuple(FieldTypes.string, FieldTypes.string),
        default=None,
    )
    # will this operator be assigned a worker or not
    reassign_worker = BoolField("reassign_worker", default=False)
    # mark a op as fuseable
    fuseable = BoolField("fuseable", default=True)
    # True means control dependency, False means data dependency
    _pure_depends = ListField("pure_depends", FieldTypes.bool, default=None)
    # useful when setting chunk index as priority,
    # useful for those op like read_csv, the first chunk
    # need to be executed not later than the later ones,
    # because the range index of later chunk should be accumulated from
    # indexes of previous ones
    # `gpu` indicates that if the operator should be executed on the GPU.
    gpu = BoolField("gpu", default=None)
    priority = Int32Field("priority", default=None)
    expect_engine = StringField("expect_engine", default=None)
    expect_resources = DictField("expect_resources", FieldTypes.string, default=None)
    # id of gang scheduling for machine learning trainings
    gang_scheduling_id = StringField("gang_scheduling_id", default=None)

    @classproperty
    @lru_cache(1)
    def all_hint_names(cls):
        return list(cls._FIELDS)

    def can_be_fused(self) -> bool:
        if not self.fuseable:
            return False
        if self.reassign_worker:
            return False
        if self._pure_depends and any(depend for depend in self._pure_depends):
            # control dependency exists
            return False
        return True


def _install_scheduling_hint_properties(cls: Type["Operator"]):
    def get_hint(name):
        def _get_val(operator: "Operator"):
            if operator.scheduling_hint:
                return getattr(operator.scheduling_hint, name)

        def _set_val(operator: "Operator", val: Any):
            if not operator.scheduling_hint:
                operator.scheduling_hint = SchedulingHint(**{name: val})
            else:
                setattr(operator.scheduling_hint, name, val)

        return property(_get_val, _set_val)

    for hint_name in SchedulingHint.all_hint_names:
        setattr(cls, hint_name, get_hint(hint_name))
    return cls


class OperatorLogicKeyGeneratorMixin:
    """
    This generator will generate an unique and deterministic key for operator compute logic. It should be same
    for different run if the compute logic doesn't change. This id will be used in substep speculative
    execution and hbo scheduling and so on.
    """

    def get_logic_key(self):
        """The subclass may need to override this method to ensure unique and deterministic."""
        fields = self._get_logic_key_token_values()
        try:
            return tokenize(*fields)
        except Exception as e:  # pragma: no cover
            raise ValueError(
                f"Cannot generate logic key for operator {self} with fields {fields}"
            ) from e

    def _get_logic_key_token_values(self):
        token_values = [type(self).__module__, type(self).__qualname__]
        if self.stage is not None:
            token_values.append(self.stage.name)
        return token_values


class LogicKeyGenerator:
    def __init__(self):
        self.operator_id_to_logic_key = {}

    def get_logic_key(self, op: "Operator"):
        assert isinstance(op, Operator)
        logic_key = self.operator_id_to_logic_key.get(op.id)
        if logic_key is None:
            logic_key = self.operator_id_to_logic_key[op.id] = op.get_logic_key()
        return logic_key


class CallPoint(Serializable):
    """
    Representing the call stack information of the last frame of the user codes.
    """

    filename = StringField("filename", default=None)
    lineno = StringField("lineno", default=None)
    name = StringField("name", default=None)

    @staticmethod
    def from_current_user_call() -> Optional["CallPoint"]:
        """
        Get the call information of user code stack.

        Returns
        -------
        CallPoint: optional
            The call stack information of the last frame in user codes (the parent of
            the first maxframe frame). If it returns None, the whole call frames are all
            in maxframe codes.
        """
        frame = get_user_call_point()
        if not frame:
            return None
        return CallPoint(
            filename=frame.f_code.co_filename,
            lineno=frame.f_lineno,
            name=frame.f_code.co_name,
        )

    def format_output(self) -> List[str]:
        return [f'  File "{self.filename}", line {self.lineno}, in {self.name}']


@_install_scheduling_hint_properties
class Operator(Base, OperatorLogicKeyGeneratorMixin, metaclass=OperatorMetaclass):
    """
    Operator misc class. All operators should have a type, which can be Add, Subtract etc.
    `sparse` indicates that if the operator is applied on a sparse tensor/chunk.
    `device`, 0 means the CPU, otherwise means the GPU device.
    Operator can have inputs and outputs
    which should be the :class:`maxframe.tensor.core.TensorData`, :class:`maxframe.tensor.core.ChunkData` etc.
    """

    attr_tag = "attr"
    _init_update_key_ = False
    _output_type_ = None
    _no_copy_attrs_ = Base._no_copy_attrs_ | {"scheduling_hint", "call_points"}
    _cache_primitive_serial = True

    sparse = BoolField("sparse", default=False)
    device = Int32Field("device", default=None)
    # will this operator create a view of input data or not
    create_view = BoolField("create_view", default=False)
    stage = ReferenceField("stage", OperatorStage, default=None)
    memory_scale = Float32Field("memory_scale", default=None)
    tileable_op_key = StringField("tileable_op_key", default=None)
    extra_params = DictField("extra_params", key_type=FieldTypes.string)
    # scheduling hint
    scheduling_hint = ReferenceField("scheduling_hint", SchedulingHint, default=None)

    # User call points. As one operator may be a merged one from many standalone
    # operators, we should use a list to keep the original call points.
    call_points = ListField(
        "call_points", FieldTypes.reference(CallPoint), default_factory=list
    )

    _inputs = ListField(
        "inputs", FieldTypes.reference(EntityData), default_factory=list
    )
    # outputs are weak-refs which are not pickle-able
    _outputs = ListField(
        "outputs", default=None, on_serialize=lambda outputs: [o() for o in outputs]
    )
    _output_types = ListField(
        "output_type",
        FieldTypes.int32,
        default=None,
        on_serialize=OutputType.serialize_list,
        on_deserialize=OutputType.deserialize_list,
    )

    def __init__(self: OperatorType, *args, **kwargs):
        self._parse_kwargs(kwargs)
        call_points = kwargs.pop("call_points", None) or [
            CallPoint.from_current_user_call()
        ]
        super().__init__(call_points=call_points, *args, **kwargs)

    @classmethod
    def _parse_kwargs(cls, kwargs: Dict[str, Any]):
        extra_params = kwargs.pop("extra_params", {})
        kwargs["extra_params"] = extras = AttributeDict(extra_params)
        kwargs["scheduling_hint"] = scheduling_hint = kwargs.get(
            "scheduling_hint", SchedulingHint()
        )
        for k in set(kwargs):
            if k in cls._FIELDS:
                continue
            elif k in SchedulingHint.all_hint_names:
                setattr(scheduling_hint, k, kwargs.pop(k))
            else:
                extras[k] = kwargs.pop(k)

    @property
    def op_name(self) -> str:
        return type(self).__name__

    def __repr__(self):
        if self.stage is None:
            return f"{type(self).__name__} <key={self.key}>"
        else:
            return f"{type(self).__name__} <key={self.key}, stage={self.stage.name}>"

    @classmethod
    def _get_entity_data(cls, entity):
        if isinstance(entity, Entity):
            return entity.data
        return entity

    @classmethod
    def _get_inputs_data(cls, inputs):
        return [cls._get_entity_data(inp) for inp in inputs]

    @classmethod
    def _set_inputs(cls, op: "Operator", inputs: List[EntityData]):
        if inputs is not None:
            inputs = cls._get_inputs_data(inputs)
        if hasattr(op, "check_inputs"):
            op.check_inputs(inputs)
        setattr(op, "_inputs", inputs)

    def replace_input(self, index: int, replaced_input: ENTITY_TYPE):
        """
        Replace the input[index] with replaced_input.

        Parameters
        ----------
        index : int
            The input to be replaced index.
        replaced_input : ENTITY_TYPE
            The replaced input object.
        """
        self.inputs[index] = replaced_input
        self._set_inputs(self, self.inputs)

    @property
    def inputs(self) -> List[Union[ENTITY_TYPE]]:
        inputs = self._inputs
        if inputs is None:
            inputs = self._inputs = []
        return inputs

    @inputs.setter
    def inputs(self, vals):
        self._set_inputs(self, vals)

    @property
    def output_limit(self) -> int:
        return 1

    @property
    def pure_depends(self):
        val = self._pure_depends  # pylint: disable=access-member-before-definition
        if not val:
            val = self._pure_depends = [False] * len(self.inputs or ())
        return val

    @property
    def output_types(self):
        return self._output_types

    @output_types.setter
    def output_types(self, value):
        self._output_types = value

    def _attach_outputs(self, *outputs):
        self._outputs = [
            weakref.ref(self._get_entity_data(o)) if o is not None else o
            for o in outputs
        ]

        if len(self._outputs) > self.output_limit:
            raise ValueError("Outputs' size exceeds limitation")

    @property
    def outputs(self) -> List[Tileable]:
        outputs = self._outputs
        if outputs:
            return [ref() for ref in outputs]

    @outputs.setter
    def outputs(self, outputs):
        self._attach_outputs(*outputs)

    def is_sparse(self) -> bool:
        return self.sparse

    issparse = is_sparse

    def is_gpu(self) -> bool:
        return self.gpu

    def has_custom_code(self) -> bool:
        return False

    @property
    def retryable(self) -> bool:
        return True

    def get_dependent_data_keys(self):
        return [dep.key for dep in self.inputs or ()]

    def _get_output_type(self, output_idx):
        if self.output_types:
            try:
                return self.output_types[output_idx]
            except IndexError:
                return self.output_types[0]
        else:
            return self._output_type_

    def copy(self: OperatorType) -> OperatorType:
        new_op = super().copy()
        new_op.outputs = []
        # copy scheduling_hint
        new_op.scheduling_hint = self.scheduling_hint.copy()
        # copy call_points
        new_op.call_points = self.call_points.copy()
        extra_params = self.extra_params
        if extra_params:
            new_op.extra_params = deepcopy(extra_params)
        return new_op

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "Operator"
    ) -> None:
        try:
            total_input_size = sum(ctx[inp.key] for inp in (op.inputs or ()))
            for out in op.outputs:
                ctx[out.key] = total_input_size // len(op.outputs)
        except KeyError:
            for out in op.outputs:
                ctx[out.key] = float("inf")

    def on_output_modify(self, new_output):
        # when `create_view` is True, if the output is modified,
        # the modification should be set back to the input.
        # This function is for this sort of usage.
        # Remember, if `create_view` is False, this function should take no effect.
        raise NotImplementedError

    def on_input_modify(self, new_input):
        # when `create_view` is True, if the input is modified,
        # this function could be used to respond the modification.
        # Remember, if `create_view` is False, this function should take no effect.
        raise NotImplementedError


class OperatorSerializer(SerializableSerializer):
    def serial(self, obj: Serializable, context: Dict):
        res = super().serial(obj, context)
        return res

    def deserial(self, serialized: List, context: Dict, subs: List) -> Operator:
        # convert outputs back to weak-refs
        operator: Operator = super().deserial(serialized, context, subs)
        for i, out in enumerate(operator._outputs):

            def cb(o, index):
                outputs = operator._outputs
                outputs[index] = weakref.ref(o)

                if len(outputs) > 1 and all(
                    not isinstance(o, Placeholder) for o in outputs
                ):
                    # all replaced
                    # add siblings for multiple outputs
                    outputs = operator.outputs
                    for j in range(len(outputs)):
                        outputs[j]._siblings = outputs[:j] + outputs[j + 1 :]

            if isinstance(out, Placeholder):
                out.callbacks.append(partial(cb, index=i))
            else:
                cb(out, i)
        return operator


OperatorSerializer.register(Operator)


class VirtualOperator(Operator):
    def get_dependent_data_keys(self):
        return []


class HasInput(Operator):
    __slots__ = ()

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op: "Operator", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op._input = op._inputs[0]
