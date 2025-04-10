from dataclasses import dataclass

from app.dto.message import BaseMessage, MessageType
from app.entity.node_identifier import NodeIdentifier


@dataclass
class ReceivedMessage:
    message: BaseMessage
    sender: NodeIdentifier
