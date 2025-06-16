from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Dict, Optional, Union, overload

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from starlette.applications import Starlette
from starlette.routing import Mount

from ..core.factory import AgentFactory
from .config import A2AAgentConfig, A2AServiceConfig, AgentServiceFactoryConfig
from .executor import SemanticKernelAgentExecutor

logger = logging.getLogger(__name__)


class AgentServiceFactory:
    @overload
    def __init__(
        self, agent_factory_or_config: AgentFactory, a2a_config: A2AServiceConfig
    ) -> None: ...

    @overload
    def __init__(self, agent_factory_or_config: AgentServiceFactoryConfig) -> None: ...

    def __init__(
        self,
        agent_factory_or_config: Union[AgentFactory, AgentServiceFactoryConfig],
        a2a_config: Optional[A2AServiceConfig] = None,
    ) -> None:
        self._stack = AsyncExitStack()
        self._app: Optional[Starlette] = None
        self._a2a_apps: Dict[str, Starlette] = {}
        self._executors: Dict[str, SemanticKernelAgentExecutor] = {}
        self._owns_agent_factory = False
        self._agent_factory: Optional[AgentFactory] = None

        if isinstance(agent_factory_or_config, AgentFactory):
            if a2a_config is None:
                raise ValueError("a2a_config required when first argument is AgentFactory")
            self._agent_factory = agent_factory_or_config
            self._a2a_config = a2a_config
        else:
            self._a2a_config = agent_factory_or_config.service_factory
            self._agent_factory_config = agent_factory_or_config.agent_factory
            self._agent_factory = None
            self._owns_agent_factory = True

    async def __aenter__(self):
        await self._stack.__aenter__()

        if self._owns_agent_factory and self._agent_factory is None:
            self._agent_factory = await self._stack.enter_async_context(
                AgentFactory(self._agent_factory_config)
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            for executor in self._executors.values():
                await executor.cleanup()
            self._executors.clear()
            self._app = None
            self._a2a_apps.clear()
        except Exception as e:
            logger.error(f"Error during service factory cleanup: {e}")
        finally:
            return await self._stack.__aexit__(exc_type, exc_val, exc_tb)

    async def create_application(self) -> Starlette:
        if self._app:
            return self._app

        routes = []
        for agent_id, agent_config in self._a2a_config.services.items():
            try:
                a2a_app = await self._create_a2a_app(agent_id, agent_config)
                if a2a_app:
                    path_prefix = agent_config.path_prefix or f"/{agent_id}"
                    routes.append(Mount(path_prefix, app=a2a_app))
                    self._a2a_apps[agent_id] = a2a_app
                    logger.info(f"Created A2A app for agent '{agent_id}' at '{path_prefix}'")
            except Exception as e:
                logger.error(f"Failed to create A2A app for agent '{agent_id}': {e}")

        self._app = Starlette(routes=routes)
        return self._app

    async def _create_a2a_app(self, agent_id: str, config: A2AAgentConfig) -> Optional[Starlette]:
        if self._agent_factory is None:
            logger.error("Agent factory is not initialized")
            return None

        try:
            agent = self._agent_factory.get_agent(agent_id)
        except KeyError:
            logger.error(f"Agent '{agent_id}' not found in factory")
            return None

        service_id = self._agent_factory.get_agent_service_id(agent_id)

        executor = SemanticKernelAgentExecutor(
            agent,
            chat_history_threshold=config.chat_history_threshold,
            chat_history_target=config.chat_history_target,
            service_id=service_id,
            enable_token_streaming=config.enable_token_streaming,
        )

        self._executors[agent_id] = executor

        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(agent_executor=executor, task_store=task_store)

        a2a_app_factory = A2AStarletteApplication(
            agent_card=config.card.to_agent_card(), http_handler=request_handler
        )

        return a2a_app_factory.build()

    def get_executor(self, agent_id: str) -> Optional[SemanticKernelAgentExecutor]:
        return self._executors.get(agent_id)

    async def cleanup_session(self, agent_id: str, session_id: str) -> None:
        executor = self._executors.get(agent_id)
        if executor:
            await executor.cleanup_session(session_id)
