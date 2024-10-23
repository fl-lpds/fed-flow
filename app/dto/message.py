import io
import pickle
from abc import ABC
from typing import NewType, Final

import torch

MessageType = NewType("MessageType", str)


class BaseMessage(ABC):
    @staticmethod
    def create_message_by_type(msg_type: MessageType, data: any = None) -> 'BaseMessage':
        if msg_type == GlobalWeightMessage.MESSAGE_TYPE:
            return GlobalWeightMessage(data)
        elif msg_type == JsonMessage.MESSAGE_TYPE:
            return JsonMessage(data)
        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'BaseMessage':
        cls = globals()[msg_type]
        deserialize = getattr(cls, 'deserialize')
        return deserialize(data, msg_type)

    def serialize(self) -> bytes:
        raise NotImplementedError

    def get_message_type(self) -> MessageType:
        raise NotImplementedError

    # def get_message_size(self) -> int:
    #     raise NotImplementedError


class GlobalWeightMessage(BaseMessage):
    weights: list[dict]
    dataset_len: int
    MESSAGE_TYPE = MessageType('GlobalWeightMessage')

    def __init__(self, weights: list[dict], dataset_len: int = None):
        self.weights = weights
        self.dataset_len = dataset_len

    def serialize(self) -> bytes:
        ll = []
        for o in self.weights:
            to_send = io.BytesIO()
            torch.save(o, to_send, _use_new_zipfile_serialization=False)
            to_send.seek(0)
            ll.append(bytes(to_send.read()))

        if self.dataset_len is None:
            return pickle.dumps((ll,))
        else:
            return pickle.dumps((ll, self.dataset_len))

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'GlobalWeightMessage':
        deserialized_data = pickle.loads(data)
        if len(deserialized_data) == 1:
            ll = deserialized_data[0]
            dataset_len = None
        else:
            ll, dataset_len = deserialized_data
        fl: list[dict] = []
        for o in ll:
            fl.append(torch.load(io.BytesIO(o)))

        return GlobalWeightMessage(fl, dataset_len)

    def get_message_type(self) -> MessageType:
        return self.MESSAGE_TYPE


class NetworkTestMessage(GlobalWeightMessage):
    MESSAGE_TYPE: Final = MessageType('NetworkTestMessage')

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'NetworkTestMessage':
        msg = GlobalWeightMessage.deserialize(data, msg_type)
        return NetworkTestMessage(msg.weights)


class JsonMessage(BaseMessage):
    MESSAGE_TYPE = MessageType('JsonMessage')
    data: any

    def __init__(self, data: any):
        self.data = data

    def serialize(self) -> bytes:
        return pickle.dumps(self.data)

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'JsonMessage':
        return JsonMessage(pickle.loads(data))

    def get_message_type(self) -> MessageType:
        return self.MESSAGE_TYPE


class IterationFlagMessage(JsonMessage):
    MESSAGE_TYPE = MessageType('IterationFlagMessage')
    flag: bool

    def __init__(self, flag: bool):
        super().__init__(flag)
        self.flag = flag

    def serialize(self) -> bytes:
        return super().serialize()

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'IterationFlagMessage':
        msg = JsonMessage.deserialize(data, JsonMessage.MESSAGE_TYPE)
        return IterationFlagMessage(msg.data)


class EnergyReportMessage(JsonMessage):
    MESSAGE_TYPE = MessageType('EnergyReportMessage')

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'JsonMessage':
        msg = JsonMessage.deserialize(data, msg_type)
        return EnergyReportMessage(msg.data)


class SplitLayerConfigMessage(JsonMessage):
    MESSAGE_TYPE = MessageType('SplitLayerConfigMessage')

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'JsonMessage':
        msg = JsonMessage.deserialize(data, msg_type)
        return SplitLayerConfigMessage(msg.data)


class DataSizeMessage(JsonMessage):
    MESSAGE_TYPE = MessageType('DataSizeMessage')

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'JsonMessage':
        msg = JsonMessage.deserialize(data, msg_type)
        return DataSizeMessage(msg.data)