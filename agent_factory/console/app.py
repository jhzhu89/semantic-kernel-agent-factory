from typing import Any, Awaitable, Callable, Dict, Type

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from .chat_service import ChatService
from .constants import MESSAGE_COUNT_TYPES
from .history_config import AgentFactoryCliConfig
from .models import (
    AgentSelected,
    ChatMessage,
    ErrorMessage,
    HandoffAgentMessage,
    HandoffFinalOutput,
    IntermediateMessage,
    MessageType,
    StreamingChunk,
    StreamingEnded,
    StreamingStarted,
    TabActivated,
    TabCreated,
    UserMessageSent,
)
from .widgets import AgentPanel, MultiChatContainer


class AgentFactoryConsole(App):
    CSS_PATH = "styles.tcss"

    HELP_TEXT = """KEYBOARD SHORTCUTS

📝 Chat Controls:
   Ctrl+Enter    Send message to agent
   Ctrl+L        Clear chat history

📋 Navigation:
   Page Up/Down  Scroll chat history
   Home/End      Go to top/bottom
   F1            Toggle agent panel
   Ctrl+W        Close current tab

🔧 Application:
   F10           Toggle this help screen
   Ctrl+Q        Exit application
   Ctrl+R        Refresh display

💬 Text Commands:
   'exit'/'quit' Exit application
   'clear'       Clear chat history
   'help'        Show this help"""

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("f1", "toggle_panel", "Panel"),
        Binding("f10", "toggle_help", "Help"),
        Binding("ctrl+w", "close_tab", "Close Tab"),
    ]

    def __init__(self, factory, config: AgentFactoryCliConfig):
        super().__init__()
        self.chat_service = ChatService(factory, config)
        self._chat_container: MultiChatContainer = MultiChatContainer(id="chat-container")
        self._agent_panel: AgentPanel = AgentPanel(
            agent_names=self.chat_service.get_agent_names(), id="agent-panel"
        )
        self._event_handlers: Dict[Type[Any], Callable[[Any], Awaitable[None]]] = {
            StreamingStarted: self.handle_streaming_started,
            StreamingChunk: self.handle_streaming_chunk,
            StreamingEnded: self.handle_streaming_ended,
            IntermediateMessage: self.handle_intermediate_message,
            HandoffAgentMessage: self.handle_handoff_agent_message,
            HandoffFinalOutput: self.handle_handoff_final_output,
            ErrorMessage: self.handle_error_message,
        }

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container") as main_container:
            self._main_container = main_container
            yield self._agent_panel
            yield self._chat_container
        yield Footer()

    def on_mount(self):
        self.theme = "textual-dark"

    @on(AgentSelected)
    async def on_agent_selected(self, message: AgentSelected) -> None:
        agent_name = message.agent_name

        if agent_name in self._chat_container.chat_tabs:
            self._chat_container.activate_tab(agent_name)
        else:
            self.chat_service.create_chat_session(agent_name)
            self._chat_container.add_chat_tab(agent_name)

    @on(TabCreated)
    async def on_tab_created(self, message: TabCreated) -> None:
        self._add_agent_instructions(message.agent_name)

    @on(TabActivated)
    async def on_tab_activated(self, message: TabActivated) -> None:
        active_tab = self._chat_container.get_active_tab()
        if active_tab:
            active_tab.message_input.focus()

    def _add_agent_instructions(self, agent_name: str) -> None:
        try:
            instructions = self.chat_service.get_agent_instructions(agent_name)
            self._add_message_to_tab(agent_name, MessageType.AGENT_INSTRUCTIONS, instructions)
        except ValueError:
            pass

    def action_toggle_panel(self) -> None:
        self._main_container.toggle_class("hide-panel")

    def action_clear_chat(self) -> None:
        active_tab = self._chat_container.get_active_tab()
        if active_tab:
            active_tab.chat_log.clear()
            active_tab.message_count = 0
            self._update_status_for_tab(active_tab.agent_name)
            self._add_message_to_tab(active_tab.agent_name, MessageType.SYSTEM, "Chat cleared")

    def action_close_tab(self) -> None:
        active_tab = self._chat_container.get_active_tab()
        if active_tab and len(self._chat_container.chat_tabs) > 1:
            self._chat_container.remove_tab(active_tab.agent_name)

    @on(UserMessageSent)
    async def on_user_message_sent(self, message: UserMessageSent) -> None:
        self._add_message_to_tab(message.agent_name, MessageType.USER, message.content)

        async for event in self.chat_service.send_message(message.agent_name, message.content):
            handler = self._event_handlers.get(type(event))
            if handler:
                await handler(event)

    def action_toggle_help(self) -> None:
        active_tab = self._chat_container.get_active_tab()
        if active_tab:
            self._add_message_to_tab(active_tab.agent_name, MessageType.SYSTEM, self.HELP_TEXT)

    def _add_message_to_tab(self, agent_name: str, msg_type: MessageType, content: str) -> None:
        tab = self._chat_container.get_tab_by_agent_name(agent_name)
        if tab:
            message = ChatMessage(type=msg_type, content=content)
            tab.chat_log.add_message(message)
            if msg_type in MESSAGE_COUNT_TYPES:
                tab.message_count += 1
                tab.status_bar.update_stats(agent_name, tab.message_count)

    def _update_status_for_tab(self, agent_name: str) -> None:
        tab = self._chat_container.get_tab_by_agent_name(agent_name)
        if tab:
            tab.status_bar.update_stats(agent_name, tab.message_count)

    async def handle_streaming_started(self, event: StreamingStarted) -> None:
        tab = self._chat_container.get_tab_by_agent_name(event.agent_name)
        if tab:
            tab.chat_log.start_streaming_message(MessageType.ASSISTANT)

    async def handle_streaming_chunk(self, event: StreamingChunk) -> None:
        tab = self._chat_container.get_tab_by_agent_name(event.agent_name)
        if tab:
            tab.chat_log.append_to_streaming(event.chunk)

    async def handle_streaming_ended(self, event: StreamingEnded) -> None:
        tab = self._chat_container.get_tab_by_agent_name(event.agent_name)
        if tab:
            tab.chat_log.finalize_streaming_message()
            tab.message_count += 1
            tab.status_bar.update_stats(event.agent_name, tab.message_count)

    async def handle_intermediate_message(self, event: IntermediateMessage) -> None:
        self._add_message_to_tab(event.agent_name, event.message_type, event.content)

    async def handle_handoff_agent_message(self, event: HandoffAgentMessage) -> None:
        content = f"[{event.source_agent}]: {event.content}"
        self._add_message_to_tab(event.target_agent, MessageType.ASSISTANT, content)

    async def handle_handoff_final_output(self, event: HandoffFinalOutput) -> None:
        self._add_message_to_tab(event.agent_name, MessageType.ASSISTANT, event.final_result)

    async def handle_error_message(self, event: ErrorMessage) -> None:
        self._add_message_to_tab(event.agent_name, MessageType.ERROR, event.error)
