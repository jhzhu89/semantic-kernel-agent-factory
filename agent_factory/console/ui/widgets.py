from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import RichLog, Static, TabbedContent, TabPane

from ..domain.models import (
    ChatMessage,
    MessageSubmitted,
    MessageType,
    TabActivated,
    TabCreated,
    TabRemoved,
    UserMessageSent,
)
from .components.chat_components import (
    ChatBubble,
    ChatBubbleContainer,
    MessageInput,
    StreamingBubble,
)


class ChatLog(ScrollableContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[ChatMessage] = []
        self.current_streaming_bubble: Optional[StreamingBubble] = None
        self.pending_function_messages: List[ChatMessage] = []

    def add_message(self, message: ChatMessage) -> None:
        if message.type in [MessageType.FUNCTION_CALL, MessageType.FUNCTION_RESULT]:
            if self.current_streaming_bubble is not None:
                logger.debug(f"Pending function message: {message.type.value}")
                self.pending_function_messages.append(message)
                return
            elif self.pending_function_messages:
                logger.debug(f"Flushing {len(self.pending_function_messages)} pending messages")
                for pending_msg in self.pending_function_messages:
                    self.messages.append(pending_msg)
                    bubble = ChatBubble(
                        pending_msg.type, pending_msg.content, pending_msg.timestamp
                    )
                    container = ChatBubbleContainer(bubble)
                    self.mount(container)
                self.pending_function_messages.clear()

        logger.debug(f"Adding message: {message.type.value}")
        self.messages.append(message)
        bubble = ChatBubble(message.type, message.content, message.timestamp)
        container = ChatBubbleContainer(bubble)
        self.mount(container)
        self.scroll_end(animate=False)

    def start_streaming_message(self, message_type: MessageType = MessageType.ASSISTANT) -> None:
        if self.pending_function_messages:
            logger.debug(
                f"Starting stream with {len(self.pending_function_messages)} pending messages"
            )
            for pending_msg in self.pending_function_messages:
                self.messages.append(pending_msg)
                bubble = ChatBubble(pending_msg.type, pending_msg.content, pending_msg.timestamp)
                container = ChatBubbleContainer(bubble)
                self.mount(container)
            self.pending_function_messages.clear()
        else:
            logger.debug("Starting stream with no pending messages")

        bubble = StreamingBubble(message_type, datetime.now())
        self.current_streaming_bubble = bubble
        container = ChatBubbleContainer(bubble)
        self.mount(container)
        self.scroll_end(animate=False)

    def append_to_streaming(self, chunk: str) -> None:
        if self.current_streaming_bubble and chunk:
            self.current_streaming_bubble.append_content(chunk)
            self.scroll_end(animate=False)

    def finalize_streaming_message(self) -> None:
        if self.current_streaming_bubble:
            logger.debug("Finalizing streaming message")
            final_message = ChatMessage(
                type=self.current_streaming_bubble.message_type,
                content=(
                    self.current_streaming_bubble.get_final_content()
                    if hasattr(self.current_streaming_bubble, "get_final_content")
                    else ""
                ),
                timestamp=self.current_streaming_bubble.timestamp,
            )
            self.messages.append(final_message)
            self.current_streaming_bubble = None

            if self.pending_function_messages:
                logger.debug(
                    f"Processing {len(self.pending_function_messages)} pending messages after stream end"
                )
                for pending_msg in self.pending_function_messages:
                    self.messages.append(pending_msg)
                    bubble = ChatBubble(
                        pending_msg.type, pending_msg.content, pending_msg.timestamp
                    )
                    container = ChatBubbleContainer(bubble)
                    self.mount(container)
                self.pending_function_messages.clear()
                self.scroll_end(animate=False)

    def clear(self) -> None:
        logger.debug(
            f"Clearing chat log with {len(self.pending_function_messages)} pending messages"
        )
        self.messages.clear()
        self.current_streaming_bubble = None
        self.pending_function_messages.clear()
        for child in list(self.children):
            child.remove()


class StatusBar(Static):
    def __init__(self, agent_name: str = "Assistant", message_count: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.message_count = message_count
        self.online_status = True
        self._update_display()

    def update_stats(self, agent_name: str, message_count: int, online: bool = True) -> None:
        self.agent_name = agent_name
        self.message_count = message_count
        self.online_status = online
        self._update_display()

    def _update_display(self) -> None:
        status_indicator = "🟢" if self.online_status else "🔴"
        status_text = (
            f"🤖 {self.agent_name} | 💬 {self.message_count} messages | {status_indicator}"
        )
        self.update(status_text)


class ChatTab(TabPane):
    def __init__(self, agent_name: str, **kwargs):
        super().__init__(agent_name, id=f"tab-{agent_name}", **kwargs)
        self.agent_name = agent_name
        self.message_count = 0

        self.chat_log = ChatLog(classes="chat-log")
        self.status_bar = StatusBar(agent_name=agent_name, message_count=0, classes="status-bar")
        self.message_input = MessageInput(classes="message-input")

    def compose(self) -> ComposeResult:
        yield self.chat_log
        with Container(classes="status-container"):
            yield self.status_bar
        with Container(classes="input-container"):
            yield self.message_input

    def on_mount(self) -> None:
        self.message_input.focus()

    @on(MessageSubmitted)
    async def handle_message_submitted(self, message: MessageSubmitted) -> None:
        self.post_message(UserMessageSent(message.content, self.agent_name))


class MultiChatContainer(Container):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_tabs: Dict[str, ChatTab] = {}
        self._tabbed_content: TabbedContent = TabbedContent(id="chat-tabs")
        self._logs_widget = RichLog(highlight=True, markup=True, id="logs-content")
        self._logs_visible = False

    def compose(self) -> ComposeResult:
        yield self._tabbed_content
        with Container(id="logs-overlay", classes="hidden"):
            with Container(id="logs-header"):
                yield Static("📋 Application Logs", classes="logs-title")
            yield self._logs_widget

    def add_chat_tab(self, agent_name: str) -> None:
        if agent_name in self.chat_tabs:
            self.activate_tab(agent_name)
            return

        chat_tab = ChatTab(agent_name)
        self.chat_tabs[agent_name] = chat_tab

        self._tabbed_content.add_pane(chat_tab)
        self._tabbed_content.active = f"tab-{agent_name}"

        self.post_message(TabCreated(agent_name))
        self.post_message(TabActivated(agent_name))

    def activate_tab(self, agent_name: str) -> None:
        if agent_name in self.chat_tabs:
            self._tabbed_content.active = f"tab-{agent_name}"
            self.post_message(TabActivated(agent_name))

    def remove_tab(self, agent_name: str) -> None:
        if agent_name in self.chat_tabs:
            self._tabbed_content.remove_pane(f"tab-{agent_name}")
            del self.chat_tabs[agent_name]
            self.post_message(TabRemoved(agent_name))

    def get_tab_by_agent_name(self, agent_name: str) -> Optional[ChatTab]:
        return self.chat_tabs.get(agent_name)

    def get_tab(self, agent_name: str) -> Optional[ChatTab]:
        return self.chat_tabs.get(agent_name)

    def get_active_tab(self) -> Optional[ChatTab]:
        active_id = self._tabbed_content.active
        if active_id and active_id.startswith("tab-"):
            agent_name = active_id[4:]
            return self.chat_tabs.get(agent_name)
        return None

    def add_message(self, agent_name: str, message: ChatMessage) -> None:
        tab = self.get_tab(agent_name)
        if tab:
            tab.chat_log.add_message(message)
            MESSAGE_COUNT_TYPES = {MessageType.USER, MessageType.ASSISTANT}
            if message.type in MESSAGE_COUNT_TYPES:
                tab.message_count += 1
                self.update_tab_title(agent_name, tab.message_count)

    def start_streaming(self, agent_name: str) -> None:
        tab = self.get_tab(agent_name)
        if tab:
            tab.chat_log.start_streaming_message(MessageType.ASSISTANT)

    def add_streaming_chunk(self, agent_name: str, content: str) -> None:
        tab = self.get_tab(agent_name)
        if tab:
            tab.chat_log.append_to_streaming(content)

    def end_streaming(self, agent_name: str) -> None:
        tab = self.get_tab(agent_name)
        if tab:
            tab.chat_log.finalize_streaming_message()

    def update_tab_title(self, agent_name: str, message_count: int) -> None:
        tab = self.get_tab(agent_name)
        if tab:
            tab.status_bar.update_stats(agent_name, message_count)

    def toggle_logs_overlay(self) -> None:
        logs_overlay = self.query_one("#logs-overlay")
        if self._logs_visible:
            logs_overlay.remove_class("visible")
            logs_overlay.add_class("hidden")
        else:
            logs_overlay.remove_class("hidden")
            logs_overlay.add_class("visible")
        self._logs_visible = not self._logs_visible

    def get_log_widget(self) -> RichLog:
        return self._logs_widget

    def get_logs_widget(self) -> RichLog:
        return self._logs_widget
