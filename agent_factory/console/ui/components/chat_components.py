from __future__ import annotations

from datetime import datetime
from typing import Optional

from textual import on
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Static, TextArea

from ...domain.models import (
    MessageSubmitted,
    MessageType,
)

MESSAGE_TYPE_ICONS = {
    MessageType.USER: ("👤", "You"),
    MessageType.ASSISTANT: ("🤖", "Assistant"),
    MessageType.SYSTEM: ("💡", "System"),
    MessageType.FUNCTION_CALL: ("⚡", "Function Call"),
    MessageType.FUNCTION_RESULT: ("✅", "Function Result"),
    MessageType.ERROR: ("❌", "Error"),
    MessageType.AGENT_INSTRUCTIONS: ("📋", "Agent Instructions"),
}


class MessageInput(TextArea):
    BINDINGS = [
        Binding("ctrl+enter,ctrl+j", "submit_message", "Submit message", show=False),
        Binding("escape", "clear_input", "Clear input", show=False),
    ]

    def action_submit_message(self) -> None:
        content = self.text.strip()
        if content:
            self.post_message(MessageSubmitted(content))
            self.clear()

    def action_clear_input(self) -> None:
        self.clear()


class ChatBubbleContainer(Container):
    def __init__(self, bubble: "ChatBubble", **kwargs):
        super().__init__(classes=f"{bubble.message_type.value.lower()}-container", **kwargs)
        self.bubble = bubble

    def compose(self):
        spacer = Container(classes="spacer")
        if self.bubble.message_type == MessageType.USER:
            yield spacer
            yield self.bubble
        else:
            yield self.bubble
            yield spacer


class ChatBubble(Static):
    def __init__(
        self,
        message_type: MessageType,
        content: str = "",
        timestamp: Optional[datetime] = None,
        **kwargs,
    ):
        self.timestamp = timestamp or datetime.now()
        self.message_type = message_type
        kwargs["markup"] = False

        full_content = self._generate_header() + "\n\n" + content
        super().__init__(full_content, classes=f"bubble {message_type.value.lower()}", **kwargs)

    def _generate_header(self) -> str:
        timestamp_str = self.timestamp.strftime("%b %d, %I:%M:%S %p")
        icon, title = MESSAGE_TYPE_ICONS.get(self.message_type, ("💬", "Message"))
        return f"{icon} {title} [{timestamp_str}]"

    def update_content(self, content: str) -> None:
        full_content = self._generate_header() + "\n\n" + content
        self.update(full_content)


class StreamingBubble(ChatBubble):
    def __init__(self, message_type: MessageType, timestamp: Optional[datetime] = None, **kwargs):
        super().__init__(message_type, "", timestamp, **kwargs)
        self._streaming_content = ""

    def append_content(self, new_content: str) -> None:
        if new_content:
            self._streaming_content += new_content
            self.update_content(self._streaming_content)

    def get_final_content(self) -> str:
        return self._streaming_content
