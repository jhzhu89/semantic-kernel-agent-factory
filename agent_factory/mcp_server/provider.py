import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict

from semantic_kernel.connectors.mcp import MCPSsePlugin, MCPStdioPlugin

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPProvider:
    def __init__(self, configs: Dict[str, MCPServerConfig]):
        self._configs = configs
        self._plugins: Dict[str, Any] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        await self._stack.__aenter__()
        for name, config in self._configs.items():
            plugin = self._create_plugin(name, config)
            plugin_instance = await self._stack.enter_async_context(plugin)
            self._plugins[name] = plugin_instance
            logger.info(f"Plugin '{name}' connected")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            self._plugins.clear()
        except Exception as e:
            logger.error(f"Error clearing plugins: {e}")

        try:
            return await self._stack.__aexit__(exc_type, exc_val, exc_tb)
        except asyncio.CancelledError:
            return True
        except Exception as e:
            logger.error(f"Error during stack cleanup: {e}")
            return False

    def _create_plugin(self, name: str, config: MCPServerConfig):
        if config.type == "sse" or (config.type is None and config.url):
            return MCPSsePlugin(
                name=name,
                url=str(config.url),
                request_timeout=config.timeout,
                headers=getattr(config, "headers", None),
                description=config.description or f"SSE plugin {name}",
            )

        if config.command is None:
            raise ValueError(f"Command is required for stdio MCP server '{name}'")

        env = os.environ.copy()
        env.update(config.env)
        return MCPStdioPlugin(
            name=name,
            command=config.command,
            args=config.args,
            env=env,
            request_timeout=config.timeout,
            description=config.description or f"Stdio plugin {name}",
        )
