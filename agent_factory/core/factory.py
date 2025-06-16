from __future__ import annotations

import logging
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.functions import KernelArguments
from typing_extensions import Literal

from ..mcp_server.provider import MCPProvider
from .config import AgentConfig, AgentFactoryConfig
from .registry import ServiceRegistry

logger = logging.getLogger(__name__)


class AgentFactory:
    def __init__(self, config: AgentFactoryConfig):
        self._config = config
        self._stack = AsyncExitStack()
        self._kernel: Optional[Kernel] = None
        self._agents: Dict[str, ChatCompletionAgent] = {}
        self._response_models: Dict[str, Optional[Type[BaseModel]]] = {}
        self._service_ids: Dict[str, str] = {}
        self._provider: Optional[MCPProvider] = None
        self._registry = ServiceRegistry(config.openai_models)

    async def __aenter__(self):
        await self._stack.__aenter__()
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            self._agents.clear()
            self._response_models.clear()
            self._service_ids.clear()
            self._kernel = None
        except Exception as e:
            logger.error(f"Error during agent factory cleanup: {e}")
        finally:
            return await self._stack.__aexit__(exc_type, exc_val, exc_tb)

    def get_agent(self, name: str) -> ChatCompletionAgent:
        return self._agents[name]

    def get_agent_service_id(self, name: str) -> Optional[str]:
        return self._service_ids.get(name)

    def get_agent_response_model(self, name: str) -> Optional[Type[BaseModel]]:
        return self._response_models.get(name)

    def get_all_agents(self) -> Dict[str, ChatCompletionAgent]:
        return self._agents.copy()

    async def _initialize(self):
        if self._kernel is not None:
            return

        self._kernel = self._registry.build_kernel()

        if (
            self._config.mcp
            and self._config.mcp.servers
            and self._config.mcp.auth
            and self._config.mcp.auth.on_behalf_of
            and self._config.mcp.auth.on_behalf_of.azure_ad
        ):
            from ..mcp_server.auth.filters import create_on_behalf_of_auth_filter

            obo_filter = create_on_behalf_of_auth_filter(
                self._config.mcp.servers, self._config.mcp.auth.on_behalf_of.azure_ad
            )
            self._kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, obo_filter)

        self._provider = await self._stack.enter_async_context(
            MCPProvider(
                self._config.mcp.servers if self._config.mcp else {},
                (
                    self._config.mcp.auth.on_behalf_of.azure_ad
                    if (
                        self._config.mcp
                        and self._config.mcp.auth
                        and self._config.mcp.auth.on_behalf_of
                    )
                    else None
                ),
            )
        )

        for config in self._config.agents.values():
            await self._create_agent(config)

    async def _create_agent(self, config: AgentConfig):
        if config.name is None:
            raise ValueError("Agent config must have a name")

        service_id = config.model or self._registry.select(self._config.model_selection)

        self._response_models[config.name] = self._create_response_model(config)
        self._service_ids[config.name] = service_id

        settings = self._create_execution_settings(config, service_id)
        plugins: Optional[List[Any]] = None
        if config.mcp_servers and self._provider is not None:
            plugins = []
            failed_plugins = []

            for name in config.mcp_servers:
                plugin = self._provider.get_plugin(name)
                if plugin is not None:
                    plugins.append(plugin)
                else:
                    failed_plugins.append(name)

            if failed_plugins:
                logger.warning(
                    f"Agent '{config.name}': MCP servers {failed_plugins} are not connected and will be skipped. "
                    f"Connected plugins: {[name for name in config.mcp_servers if name not in failed_plugins]}"
                )

            # If no plugins are available, set plugins to None
            if not plugins:
                plugins = None

        start_time = time.perf_counter()
        agent = ChatCompletionAgent(
            arguments=KernelArguments(settings),
            name=config.name,
            instructions=config.instructions,
            kernel=self._kernel,
            plugins=plugins,
        )

        self._agents[config.name] = agent
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"agent {config.name} via {service_id} ready in {elapsed_ms:.0f} ms")

    def _create_execution_settings(
        self, config: AgentConfig, service_id: str
    ) -> AzureChatPromptExecutionSettings:
        settings = AzureChatPromptExecutionSettings(
            service_id=service_id,
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )

        if config.model_settings:
            if config.model_settings.temperature is not None:
                settings.temperature = config.model_settings.temperature
            if config.model_settings.top_p is not None:
                settings.top_p = config.model_settings.top_p
            if config.model_settings.frequency_penalty is not None:
                settings.frequency_penalty = config.model_settings.frequency_penalty
            if config.model_settings.presence_penalty is not None:
                settings.presence_penalty = config.model_settings.presence_penalty
            if config.model_settings.max_tokens is not None:
                settings.max_tokens = config.model_settings.max_tokens

            if config.model_settings.response_json_schema:
                if config.name is None:
                    raise ValueError("Agent config must have a name for response schema")
                response_model = self._response_models.get(config.name)
                if response_model:
                    settings.response_format = response_model
                else:
                    raise ValueError(
                        f"JSON schema specified for agent '{config.name}' but model creation failed"
                    )

        return settings

    def _create_response_model(self, config: AgentConfig) -> Optional[Type[BaseModel]]:
        if not (config.model_settings and config.model_settings.response_json_schema):
            return None

        if config.name is None:
            raise ValueError("Agent config must have a name for response model")

        schema_dict = config.model_settings.response_json_schema.json_schema_definition
        return self._create_pydantic_model(schema_dict, config.name)

    def _create_pydantic_model(
        self, schema: Dict[str, Any], agent_name: str
    ) -> Optional[Type[BaseModel]]:
        try:
            properties = schema.get("properties", {})
            if not properties:
                return None

            required = schema.get("required", [])
            field_definitions: Dict[str, Any] = {}

            for field_name, field_def in properties.items():
                try:
                    field_type = self._get_python_type(field_def)

                    if field_name in required and "default" not in field_def:
                        # Required field without default
                        if "description" in field_def:
                            field_definitions[field_name] = (
                                field_type,
                                Field(description=field_def["description"]),
                            )
                        else:
                            field_definitions[field_name] = field_type
                    else:
                        # Optional field or field with default
                        default_value = field_def.get("default", None)

                        if "description" in field_def:
                            field_definitions[field_name] = (
                                field_type,
                                Field(default=default_value, description=field_def["description"]),
                            )
                        else:
                            field_definitions[field_name] = (field_type, default_value)

                except Exception:
                    continue

            if not field_definitions:
                return None

            model_name = f"ResponseModel_{agent_name}".replace(" ", "_")
            # Pass field definitions as keyword arguments to create_model
            model_class: Type[BaseModel] = create_model(model_name, **field_definitions)
            return model_class

        except Exception:
            return None

    def _get_python_type(self, field_def: Dict[str, Any]) -> Type[Any]:
        field_type = field_def.get("type", "string")
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List[Any],
            "object": Dict[str, Any],
        }
        python_type = type_mapping.get(field_type, str)

        if "enum" in field_def:
            enum_values = field_def["enum"]
            if all(isinstance(v, str) for v in enum_values):
                return Literal[tuple(enum_values)]  # type: ignore

        return python_type
