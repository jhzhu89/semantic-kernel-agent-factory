import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from ..core.config import DependencyContainer
from ..domain.models import (
    AgentSelected,
    ChatMessage,
    MessageType,
    TabActivated,
    TabCreated,
    UserMessageSent,
)
from ..domain.strategies import MessageProcessor
from ..infrastructure.config.history_config import AgentFactoryCliConfig
from ..infrastructure.health.mcp_status import MCPServerStatus, MCPStatus
from ..infrastructure.logging.manager import LoggingConfig


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
   F2            Toggle log panel
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
        Binding("f1", "toggle_panel", "Panel"),
        Binding("f2", "toggle_logs", "Logs"),
        Binding("f10", "toggle_help", "Help"),
        Binding("ctrl+w", "close_tab", "Close Tab"),
    ]

    def __init__(self, factory, config: AgentFactoryCliConfig):
        super().__init__()
        self.dependencies = DependencyContainer(factory, config)
        self.config = config
        self.mcp_provider = getattr(factory, "_provider", None)
        self._health_check_timer: Optional[asyncio.Task] = None
        self._initialize_ui_components()
        self._message_processor = MessageProcessor(
            self._chat_container, self._add_message_to_tab, self._update_status_for_tab
        )

    def _initialize_ui_components(self) -> None:
        from .components.agent_components import AgentPanel
        from .widgets import MultiChatContainer

        self._chat_container = MultiChatContainer(id="chat-container")
        mcp_statuses = self._create_mcp_statuses()
        self._agent_panel = AgentPanel(
            agent_names=self.dependencies.session_manager.get_agent_names(),
            mcp_statuses=mcp_statuses,
            id="agent-panel",
        )
        self._main_container = Horizontal(id="main-container")

    def compose(self) -> ComposeResult:
        yield Header()
        with self._main_container:
            yield self._agent_panel
            yield self._chat_container
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "textual-dark"
        self._setup_logging()
        self._start_health_check_timer()

    @on(AgentSelected)
    async def on_agent_selected(self, message: AgentSelected) -> None:
        agent_name = message.agent_name

        if agent_name in self._chat_container.chat_tabs:
            self._chat_container.activate_tab(agent_name)
        else:
            self.dependencies.session_manager.create_chat_session(agent_name)
            self._chat_container.add_chat_tab(agent_name)

    @on(TabCreated)
    async def on_tab_created(self, message: TabCreated) -> None:
        self._add_agent_instructions(message.agent_name)

    @on(TabActivated)
    async def on_tab_activated(self, message: TabActivated) -> None:
        active_tab = self._chat_container.get_active_tab()
        if active_tab:
            active_tab.message_input.focus()

    @on(UserMessageSent)
    async def on_user_message_sent(self, message: UserMessageSent) -> None:
        await self._handle_user_message(message.agent_name, message.content)

    def _create_mcp_statuses(self) -> Dict[str, MCPServerStatus]:
        mcp_configs = self.config.agent_factory.mcp.servers if self.config.agent_factory.mcp else {}
        if not mcp_configs:
            return {}
        statuses = {}
        for name, config in mcp_configs.items():
            server_type = config.type or (
                "streamable_http" if hasattr(config, "url") and config.url else "stdio"
            )
            statuses[name] = MCPServerStatus(
                name=name,
                status=MCPStatus.UNKNOWN,
                server_type=server_type,
                last_check=datetime.now(),
                error_message="Status not checked yet",
                connection_time=None,
            )
        return statuses

    def _add_agent_instructions(self, agent_name: str) -> None:
        try:
            instructions = self.dependencies.session_manager.get_agent_instructions(agent_name)
            self._add_message_to_tab(agent_name, MessageType.AGENT_INSTRUCTIONS, instructions)
        except ValueError:
            pass

    def _add_message_to_tab(self, agent_name: str, message_type: MessageType, content: str) -> None:
        message = ChatMessage(type=message_type, content=content, timestamp=datetime.now())
        self._chat_container.add_message(agent_name, message)

    def _update_status_for_tab(self, agent_name: str) -> None:
        active_tab = self._chat_container.get_tab(agent_name)
        if active_tab:
            self._chat_container.update_tab_title(agent_name, active_tab.message_count)

    async def _handle_user_message(self, agent_name: str, content: str) -> None:
        self._add_message_to_tab(agent_name, MessageType.USER, content)
        async for event in self.dependencies.session_manager.send_message(agent_name, content):
            await self._message_processor.process_event(event)

    def _setup_logging(self) -> None:
        if hasattr(self._chat_container, "get_log_widget"):
            log_widget = self._chat_container.get_log_widget()
            logging_manager = LoggingConfig.get_instance()
            logging_manager.add_ui_logging(log_widget)

    def _start_health_check_timer(self) -> None:
        if self._health_check_timer is None:
            self._health_check_timer = asyncio.create_task(self._periodic_health_check())

    async def _periodic_health_check(self) -> None:
        await asyncio.sleep(0.5)

        while True:
            try:
                await self._check_mcp_servers()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.getLogger(__name__).error(f"Health check error: {e}")

    async def _check_mcp_servers(self) -> None:
        mcp_configs = self.config.agent_factory.mcp.servers if self.config.agent_factory.mcp else {}
        if not mcp_configs:
            return

        for name, config in mcp_configs.items():
            try:
                is_healthy = await self.dependencies.health_checker.check_server_health(
                    name, config
                )
                server_type = config.type or (
                    "streamable_http" if hasattr(config, "url") and config.url else "stdio"
                )
                new_status = MCPServerStatus(
                    name=name,
                    status=MCPStatus.CONNECTED if is_healthy else MCPStatus.FAILED,
                    server_type=server_type,
                    last_check=datetime.now(),
                    error_message=None if is_healthy else "Health check failed",
                    connection_time=datetime.now() if is_healthy else None,
                )
                self._agent_panel.update_mcp_server_status(name, new_status)
            except Exception as e:
                server_type = config.type or (
                    "streamable_http" if hasattr(config, "url") and config.url else "stdio"
                )
                error_status = MCPServerStatus(
                    name=name,
                    status=MCPStatus.FAILED,
                    server_type=server_type,
                    last_check=datetime.now(),
                    error_message=str(e),
                    connection_time=None,
                )
                self._agent_panel.update_mcp_server_status(name, error_status)

    def action_toggle_panel(self) -> None:
        self._main_container.toggle_class("hide-panel")

    def action_toggle_logs(self) -> None:
        self._chat_container.toggle_logs_overlay()

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

    def action_toggle_help(self) -> None:
        active_tab = self._chat_container.get_active_tab()
        if active_tab:
            self._add_message_to_tab(active_tab.agent_name, MessageType.SYSTEM, self.HELP_TEXT)
