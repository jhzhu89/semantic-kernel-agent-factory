import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict

from semantic_kernel.connectors.mcp import MCPSsePlugin, MCPStdioPlugin
from semantic_kernel.exceptions import KernelPluginInvalidConfigurationError

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPProvider:
    def __init__(self, configs: Dict[str, MCPServerConfig]):
        self._configs = configs
        self._plugins: Dict[str, Any] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        await self._stack.__aenter__()
        failed_plugins = []

        for name, config in self._configs.items():
            try:
                plugin = self._create_plugin(name, config)
                plugin_instance = await self._stack.enter_async_context(plugin)
                self._plugins[name] = plugin_instance
                logger.info(f"Plugin '{name}' connected successfully")
            except KernelPluginInvalidConfigurationError as e:
                logger.error(f"Failed to connect plugin '{name}': Invalid configuration - {e}")
                failed_plugins.append(name)
            except ConnectionError as e:
                logger.error(f"Failed to connect plugin '{name}': Connection error - {e}")
                failed_plugins.append(name)
            except OSError as e:
                logger.error(f"Failed to connect plugin '{name}': OS error - {e}")
                failed_plugins.append(name)
            except asyncio.TimeoutError as e:
                logger.error(f"Failed to connect plugin '{name}': Connection timeout - {e}")
                failed_plugins.append(name)
            except Exception as e:
                logger.error(f"Failed to connect plugin '{name}': Unexpected error - {e}")
                failed_plugins.append(name)

        if failed_plugins:
            logger.warning(
                f"Failed to connect {len(failed_plugins)} plugin(s): {', '.join(failed_plugins)}"
            )

        if not self._plugins:
            raise RuntimeError("No MCP plugins could be connected. Check your configuration.")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        errors = []

        try:
            self._plugins.clear()
        except Exception as e:
            logger.error(f"Error clearing plugins: {e}")
            errors.append(str(e))

        try:
            await self._stack.__aexit__(exc_type, exc_val, exc_tb)
        except asyncio.CancelledError:
            logger.debug("Stack cleanup cancelled")
            return True
        except Exception as e:
            logger.error(f"Error during stack cleanup: {e}")
            errors.append(str(e))

        if errors and exc_type is None:
            logger.warning(f"Cleanup completed with {len(errors)} error(s)")

        return False

    def _create_plugin(self, name: str, config: MCPServerConfig):
        try:
            if config.type == "sse" or (config.type is None and config.url):
                if not config.url:
                    raise ValueError(f"URL is required for SSE MCP server '{name}'")
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
        except Exception as e:
            logger.error(f"Error creating plugin '{name}': {e}")
            raise

    def get_connected_plugins(self) -> Dict[str, Any]:
        return self._plugins.copy()

    def is_plugin_connected(self, name: str) -> bool:
        return name in self._plugins

    def get_plugin_count(self) -> int:
        return len(self._plugins)

    def get_plugin_names(self) -> list[str]:
        return list(self._plugins.keys())

    def get_plugin(self, name: str) -> Any | None:
        """Safely get a plugin by name, returning None if not connected."""
        return self._plugins.get(name)
