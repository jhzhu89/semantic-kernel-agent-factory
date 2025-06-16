import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from semantic_kernel.connectors.mcp import MCPStdioPlugin, MCPStreamableHttpPlugin

from .config import AzureAdConfig, MCPServerConfig

logger = logging.getLogger(__name__)


class MCPProvider:
    def __init__(
        self,
        configs: Dict[str, MCPServerConfig],
        azure_ad_config: Optional[AzureAdConfig] = None,
        cert_directory: str = ".",
    ):
        self._configs = configs
        self._azure_ad_config = azure_ad_config
        self._cert_directory = cert_directory
        self._plugins: Dict[str, Any] = {}
        self._stack = AsyncExitStack()
        logger.info(f"MCPProvider initialized with {len(configs)} server configs")

    async def __aenter__(self):
        await self._stack.__aenter__()
        failed_plugins = []

        for name, config in self._configs.items():
            plugin = None
            try:
                plugin = self._create_plugin(name, config)
                plugin_instance = await self._stack.enter_async_context(plugin)
                self._plugins[name] = plugin_instance
                logger.info(f"Plugin '{name}' connected successfully")

            except (Exception, asyncio.CancelledError) as e:
                logger.error(f"Failed to connect plugin '{name}': {e}")
                if plugin is not None:
                    await self._safe_cleanup_plugin(plugin, name)
                failed_plugins.append(name)

        # Report connection results
        if failed_plugins:
            logger.warning(
                f"Failed to connect {len(failed_plugins)} plugin(s): {', '.join(failed_plugins)}"
            )

        # Only raise error if plugins were configured but none could be connected
        if self._configs and not self._plugins:
            raise RuntimeError(
                f"No MCP plugins could be connected. "
                f"Failed plugins: {', '.join(failed_plugins)}. "
                f"Check your configuration and ensure MCP servers are accessible."
            )

        logger.info(
            f"MCPProvider initialized: {len(self._plugins)}/{len(self._configs)} plugins connected"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        cleanup_errors: list[str] = []
        should_suppress = False

        await self._cleanup_plugins_dict(cleanup_errors)
        should_suppress = (
            await self._cleanup_async_stack(exc_type, exc_val, exc_tb, cleanup_errors)
            or should_suppress
        )

        self._log_cleanup_summary(cleanup_errors, exc_type)

        if should_suppress or exc_type is asyncio.CancelledError:
            logger.debug("Suppressing exception to allow graceful shutdown")
            return True

        return False

    async def _cleanup_plugins_dict(self, cleanup_errors: list[str]):
        try:
            self._plugins.clear()
            logger.debug("Plugins dictionary cleared")
        except Exception as e:
            logger.error(f"Error clearing plugins dictionary: {e}")
            cleanup_errors.append(f"Plugin cleanup: {e}")

    async def _cleanup_async_stack(
        self, exc_type, exc_val, exc_tb, cleanup_errors: list[str]
    ) -> bool:
        try:
            await self._stack.__aexit__(exc_type, exc_val, exc_tb)
            logger.debug("AsyncExitStack cleanup completed successfully")
            return False
        except asyncio.CancelledError:
            logger.debug("AsyncExitStack cleanup was cancelled - expected during shutdown")
            return True
        except RuntimeError as e:
            if "cancel scope" in str(e) and "different task" in str(e):
                logger.debug(f"Cross-task cleanup issue (safe to ignore): {e}")
            else:
                logger.error(f"RuntimeError during AsyncExitStack cleanup: {e}")
                cleanup_errors.append(f"Stack cleanup: {e}")
            return True
        except Exception as e:
            logger.error(f"Error during AsyncExitStack cleanup: {e}")
            cleanup_errors.append(f"Stack cleanup: {e}")
            return False

    def _log_cleanup_summary(self, cleanup_errors: list[str], exc_type):
        if cleanup_errors:
            if exc_type is None:
                logger.warning(
                    f"Cleanup completed with {len(cleanup_errors)} error(s): {'; '.join(cleanup_errors)}"
                )
            else:
                logger.debug(
                    f"Cleanup had {len(cleanup_errors)} error(s) during exception handling: {'; '.join(cleanup_errors)}"
                )

    async def _safe_cleanup_plugin(self, plugin, name: str):
        if plugin is None:
            return
        try:
            if hasattr(plugin, "__aexit__"):
                await plugin.__aexit__(None, None, None)
                logger.debug(f"Plugin '{name}' cleaned up successfully")
        except Exception as e:
            logger.debug(f"Plugin '{name}' cleanup failed (this is usually safe to ignore): {e}")

    def _validate_auth_config(self, name: str, config: MCPServerConfig):
        if config.auth and config.auth.enabled:
            if not self._azure_ad_config:
                raise ValueError(f"Azure auth config required for MCP '{name}' but not provided")
            if not config.auth.scope:
                raise ValueError(f"Auth scope required for MCP '{name}'")
            if (
                not self._azure_ad_config.certificate_name
                and not self._azure_ad_config.client_secret
            ):
                raise ValueError(
                    f"Either certificate name or client secret required for MCP '{name}'"
                )

    def _create_plugin(self, name: str, config: MCPServerConfig):
        try:
            self._validate_auth_config(name, config)
            if config.type == "streamable_http" or (config.type is None and config.url):
                if not config.url:
                    raise ValueError(f"URL is required for Streamable HTTP MCP server '{name}'")

                logger.debug(f"Creating Streamable HTTP plugin: {name} -> {config.url}")
                return MCPStreamableHttpPlugin(
                    name=name,
                    url=str(config.url),
                    request_timeout=config.timeout,
                    headers=getattr(config, "headers", None),
                    description=config.description or f"Streamable HTTP plugin {name}",
                )

            if config.command is None:
                raise ValueError(f"Command is required for stdio MCP server '{name}'")

            logger.debug(f"Creating stdio plugin: {name} -> {config.command}")
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
