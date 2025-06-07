from __future__ import annotations

from typing import Dict, List, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Label, ListItem, ListView, Static

from ...domain.models import AgentSelected
from ...infrastructure.health.mcp_status import MCPServerStatus, MCPStatus

MCP_STATUS_ICONS = {
    MCPStatus.CONNECTED: "🟢",
    MCPStatus.FAILED: "🔴",
    MCPStatus.UNKNOWN: "⚫",
}


class AgentListItem(ListItem):
    def __init__(self, agent_name: str, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = agent_name

    def compose(self) -> ComposeResult:
        yield Label(f"🤖 {self.agent_name}")


class MCPServerListItem(ListItem):
    def __init__(self, server_name: str, status: MCPServerStatus, **kwargs):
        super().__init__(**kwargs)
        self.server_name = server_name
        self.status = status

    def compose(self) -> ComposeResult:
        yield Label(self._format_status_label())

    def _format_status_label(self) -> str:
        status_icon = MCP_STATUS_ICONS.get(self.status.status, "❓")
        return f"{status_icon} {self.server_name}: {self.status.status.value}"

    def update_status(self, new_status: MCPServerStatus) -> None:
        if self.status.status != new_status.status:
            self.status = new_status
            self._refresh_label()

    def _refresh_label(self) -> None:
        for child in self.children:
            if isinstance(child, Label):
                child.update(self._format_status_label())
                break


class AgentPanel(Container):
    def __init__(
        self,
        agent_names: List[str],
        mcp_statuses: Optional[Dict[str, MCPServerStatus]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_names = agent_names
        self.selected_agent: Optional[str] = None
        self._agent_list: ListView = ListView(id="agent-list")

        self.mcp_statuses = mcp_statuses or {}
        self._mcp_list: ListView = ListView(id="mcp-list")

    def on_mount(self) -> None:
        self._populate_agent_list()
        self._populate_mcp_list()

    def _populate_agent_list(self) -> None:
        for agent_name in self.agent_names:
            self._agent_list.append(AgentListItem(agent_name))

    def _populate_mcp_list(self) -> None:
        for server_name, status in self.mcp_statuses.items():
            self._mcp_list.append(MCPServerListItem(server_name, status))

    def compose(self) -> ComposeResult:
        with Container(classes="agent-list-container"):
            yield Static("🤖 Available Agents", classes="panel-header")
            yield self._agent_list

        with Container(classes="mcp-status-container"):
            yield Static("🔗 MCP Servers", classes="panel-header")
            yield self._mcp_list

    @on(ListView.Selected)
    def on_agent_selected(self, event: ListView.Selected) -> None:
        if hasattr(event.item, "agent_name"):
            self._update_selection(event.item.agent_name)
            self.post_message(AgentSelected(event.item.agent_name))

    def _update_selection(self, agent_name: str) -> None:
        for item in self._agent_list.children:
            if hasattr(item, "agent_name"):
                item.set_class(item.agent_name == agent_name, "selected")
        self.selected_agent = agent_name

    def update_mcp_server_status(self, server_name: str, status: MCPServerStatus) -> None:
        self.mcp_statuses[server_name] = status

        for item in self._mcp_list.children:
            if hasattr(item, "server_name") and item.server_name == server_name:
                if hasattr(item, "update_status"):
                    item.update_status(status)
                return

    def get_mcp_status_panel(self):
        return self
