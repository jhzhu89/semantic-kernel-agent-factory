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


class MessageSubmitted(Message):

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content


class InputCleared(Message):

    def __init__(self) -> None:
        super().__init__()


class UserMessageSent(Message):

    def __init__(self, content: str, agent_name: str) -> None:
        super().__init__()
        self.content = content
        self.agent_name = agent_name


class AgentSelected(Message):

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name


class TabCreated(Message):

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name


class TabActivated(Message):

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name


class TabRemoved(Message):

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name


class StreamingStarted(Message):

    def __init__(self, agent_name: str, message_type: MessageType = MessageType.ASSISTANT) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.message_type = message_type


class StreamingChunk(Message):

    def __init__(self, agent_name: str, chunk: str) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.chunk = chunk


class StreamingEnded(Message):

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name


class IntermediateMessage(Message):

    def __init__(self, agent_name: str, message_type: MessageType, content: str) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.message_type = message_type
        self.content = content


class HandoffAgentMessage(Message):

    def __init__(self, source_agent: str, target_agent: str, content: str) -> None:
        super().__init__()
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.content = content


class HandoffFinalOutput(Message):

    def __init__(self, agent_name: str, final_result: str) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.final_result = final_result


class ErrorMessage(Message):

    def __init__(self, agent_name: str, error: str) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.error = error
