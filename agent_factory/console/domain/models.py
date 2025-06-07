from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from textual.message import Message


class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"
    FUNCTION_CALL = "function-call"
    FUNCTION_RESULT = "function-result"
    AGENT_INSTRUCTIONS = "agent-instructions"


@dataclass
class ChatMessage:
    type: MessageType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class AgentMessage(Message):
    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name


class MessageSubmitted(Message):
    def __init__(self, content: str):
        super().__init__()
        self.content = content


class UserMessageSent(AgentMessage):
    def __init__(self, content: str, agent_name: str):
        super().__init__(agent_name)
        self.content = content


class AgentSelected(AgentMessage):
    pass


class TabCreated(AgentMessage):
    pass


class TabActivated(AgentMessage):
    pass


class TabRemoved(AgentMessage):
    pass


class StreamingStarted(AgentMessage):
    def __init__(self, agent_name: str, message_type: MessageType = MessageType.ASSISTANT):
        super().__init__(agent_name)
        self.message_type = message_type


class StreamingChunk(AgentMessage):
    def __init__(self, agent_name: str, chunk: str):
        super().__init__(agent_name)
        self.chunk = chunk


class StreamingEnded(AgentMessage):
    pass


class IntermediateMessage(AgentMessage):
    def __init__(self, agent_name: str, message_type: MessageType, content: str):
        super().__init__(agent_name)
        self.message_type = message_type
        self.content = content


class ErrorMessage(AgentMessage):
    def __init__(self, agent_name: str, error: str):
        super().__init__(agent_name)
        self.error = error
