from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..infrastructure.config.history_config import AgentFactoryCliConfig

if TYPE_CHECKING:
    from ..infrastructure.health.mcp_status import MCPHealthChecker
    from .session_manager import SessionManager


class DependencyContainer:
    def __init__(self, factory, config: AgentFactoryCliConfig):
        self.factory = factory
        self.config = config
        self._session_manager: Optional[SessionManager] = None
        self._health_checker: Optional[MCPHealthChecker] = None

    @property
    def session_manager(self):
        if self._session_manager is None:
            from .session_manager import SessionManager

            self._session_manager = SessionManager(self.factory, self.config)
        return self._session_manager

    @property
    def health_checker(self):
        if self._health_checker is None:
            from ..infrastructure.health.mcp_status import MCPHealthChecker

            self._health_checker = MCPHealthChecker(
                self.config.agent_factory.mcp.servers if self.config.agent_factory.mcp else {}
            )
        return self._health_checker
