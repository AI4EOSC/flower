"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import flwr.proto.transport_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Task(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TASK_ID_FIELD_NUMBER: builtins.int
    LEGACY_SERVER_MESSAGE_FIELD_NUMBER: builtins.int
    task_id: builtins.int
    @property
    def legacy_server_message(self) -> flwr.proto.transport_pb2.ServerMessage: ...
    def __init__(self,
        *,
        task_id: builtins.int = ...,
        legacy_server_message: typing.Optional[flwr.proto.transport_pb2.ServerMessage] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["legacy_server_message",b"legacy_server_message"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["legacy_server_message",b"legacy_server_message","task_id",b"task_id"]) -> None: ...
global___Task = Task

class TaskAssignment(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TASK_FIELD_NUMBER: builtins.int
    NODE_IDS_FIELD_NUMBER: builtins.int
    @property
    def task(self) -> global___Task: ...
    @property
    def node_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(self,
        *,
        task: typing.Optional[global___Task] = ...,
        node_ids: typing.Optional[typing.Iterable[builtins.int]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["task",b"task"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["node_ids",b"node_ids","task",b"task"]) -> None: ...
global___TaskAssignment = TaskAssignment

class Result(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TASK_ID_FIELD_NUMBER: builtins.int
    LEGACY_CLIENT_MESSAGE_FIELD_NUMBER: builtins.int
    task_id: builtins.int
    @property
    def legacy_client_message(self) -> flwr.proto.transport_pb2.ClientMessage: ...
    def __init__(self,
        *,
        task_id: builtins.int = ...,
        legacy_client_message: typing.Optional[flwr.proto.transport_pb2.ClientMessage] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["legacy_client_message",b"legacy_client_message"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["legacy_client_message",b"legacy_client_message","task_id",b"task_id"]) -> None: ...
global___Result = Result