from datetime import datetime
from typing import Dict, List, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.events import Key
from textual.widgets import Label, ListItem, ListView, Static, TabbedContent, TabPane, TextArea

from .constants import MESSAGE_TYPE_ICONS
from .models import (
    AgentSelected,
    ChatMessage,
    MessageSubmitted,
    MessageType,
    TabActivated,
    TabCreated,
    TabRemoved,
    UserMessageSent,
)
from .utils import format_timestamp


class MessageInput(TextArea):
    @on(Key)
    async def handle_message_input_keys(self, event: Key) -> None:
        if not self.has_focus:
            await super()._on_key(event)
            return

        if event.key == "ctrl+enter" or event.key == "ctrl+j":
            content = self.text.strip()
            if content:
                self.post_message(MessageSubmitted(content))
                self.clear()
                event.prevent_default()
                event.stop()
                return
        elif event.key == "escape":
            self.clear()
            event.prevent_default()
            event.stop()
            return

        await super()._on_key(event)


class AgentListItem(ListItem):
    def __init__(self, agent_name: str, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = agent_name

    def compose(self) -> ComposeResult:
        yield Label(f"🤖 {self.agent_name}")


class AgentPanel(Container):
    def __init__(self, agent_names: List[str], **kwargs):
        super().__init__(**kwargs)
        self.agent_names = agent_names
        self.selected_agent: Optional[str] = None
        self._agent_list: ListView = ListView(id="agent-list")

    def compose(self) -> ComposeResult:
        yield Static("🤖 Available Agents", classes="panel-header")
        yield self._agent_list

    def on_mount(self) -> None:
        for agent_name in self.agent_names:
            self._agent_list.append(AgentListItem(agent_name))

    @on(ListView.Selected)
    def on_agent_selected(self, event: ListView.Selected) -> None:
        if hasattr(event.item, "agent_name"):
            self._update_selection(event.item.agent_name)
            self.post_message(AgentSelected(event.item.agent_name))

    def _update_selection(self, agent_name: str) -> None:
        for item in self._agent_list.children:
            if hasattr(item, "agent_name"):
                if item.agent_name == agent_name:
                    item.add_class("selected")
                else:
                    item.remove_class("selected")

        self.selected_agent = agent_name


class ChatBubbleContainer(Container):
    def __init__(self, bubble: "ChatBubble", **kwargs):
        type_class = f"{bubble.message_type.value.lower()}-container"
        super().__init__(classes=type_class, **kwargs)
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
        classes = f"bubble {message_type.value.lower()}"

        full_content = self._generate_header() + "\n\n" + content
        super().__init__(full_content, classes=classes, **kwargs)

    def _generate_header(self) -> str:
        timestamp_str = format_timestamp(self.timestamp)
        icon, title = MESSAGE_TYPE_ICONS.get(self.message_type, ("💬", "Message"))
        return f"{icon} {title} [{timestamp_str}]"

    def update_content(self, content: str) -> None:
        full_content = self._generate_header() + "\n\n" + content
        self.update(full_content)


class StreamingBubble(ChatBubble):
    def __init__(self, message_type: MessageType, timestamp: Optional[datetime] = None, **kwargs):
        super().__init__(message_type, "", timestamp, **kwargs)
        self._streaming_content = ""

    def append_chunk(self, chunk: str) -> None:
        if chunk:
            self._streaming_content += chunk
            self.update_content(self._streaming_content)

    def get_final_content(self) -> str:
        return self._streaming_content


class ChatLog(ScrollableContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[ChatMessage] = []
        self.current_streaming_bubble: Optional[StreamingBubble] = None

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)
        bubble = ChatBubble(message.type, message.content, message.timestamp)
        container = ChatBubbleContainer(bubble)
        self.mount(container)
        self.scroll_end(animate=False)

    def start_streaming_message(self, message_type: MessageType = MessageType.ASSISTANT) -> None:
        bubble = StreamingBubble(message_type, datetime.now())
        self.current_streaming_bubble = bubble
        container = ChatBubbleContainer(bubble)
        self.mount(container)
        self.scroll_end(animate=False)

    def append_to_streaming(self, chunk: str) -> None:
        if self.current_streaming_bubble and chunk:
            self.current_streaming_bubble.append_chunk(chunk)
            self.scroll_end(animate=False)

    def finalize_streaming_message(self) -> None:
        if self.current_streaming_bubble:
            final_message = ChatMessage(
                type=self.current_streaming_bubble.message_type,
                content=self.current_streaming_bubble.get_final_content(),
                timestamp=self.current_streaming_bubble.timestamp,
            )
            self.messages.append(final_message)
            self.current_streaming_bubble = None

    def clear(self) -> None:
        self.messages.clear()
        self.current_streaming_bubble = None
        for child in list(self.children):
            child.remove()


class StatusBar(Static):
    def __init__(self, agent_name: str = "Assistant", message_count: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.message_count = message_count
        self.online_status = True
        self._update_display()

    def update_stats(self, agent_name: str, message_count: int, online: bool = True):
        self.agent_name = agent_name
        self.message_count = message_count
        self.online_status = online
        self._update_display()

    def _update_display(self):
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

    def compose(self) -> ComposeResult:
        yield self._tabbed_content

    def on_mount(self) -> None:
        pass

    def _generate_tab_id(self, agent_name: str) -> str:
        return f"tab-{agent_name}"

    def add_chat_tab(self, agent_name: str) -> None:
        if agent_name in self.chat_tabs:
            self.post_message(TabActivated(agent_name))
            return

        chat_tab = ChatTab(agent_name)
        self.chat_tabs[agent_name] = chat_tab

        self._tabbed_content.add_pane(chat_tab)
        self._tabbed_content.active = self._generate_tab_id(agent_name)

        self.post_message(TabCreated(agent_name))
        self.post_message(TabActivated(agent_name))

    def activate_tab(self, agent_name: str) -> None:
        if agent_name in self.chat_tabs:
            self._tabbed_content.active = self._generate_tab_id(agent_name)
            self.post_message(TabActivated(agent_name))

    def remove_tab(self, agent_name: str) -> None:
        if agent_name in self.chat_tabs:
            self._tabbed_content.remove_pane(self._generate_tab_id(agent_name))
            del self.chat_tabs[agent_name]
            self.post_message(TabRemoved(agent_name))

    def get_tab_by_agent_name(self, agent_name: str) -> Optional[ChatTab]:
        return self.chat_tabs.get(agent_name)

    def get_active_tab(self) -> Optional[ChatTab]:
        if self._tabbed_content.active and self._tabbed_content.active.startswith("tab-"):
            agent_name = self._tabbed_content.active[4:]
            return self.chat_tabs.get(agent_name)
        return None
