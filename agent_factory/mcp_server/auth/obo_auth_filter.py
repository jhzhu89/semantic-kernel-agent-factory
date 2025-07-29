from contextvars import ContextVar
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata

from ..config import AzureAdConfig, MCPServerConfig
from .azure_credential_factory import FactoryManager
from .obo_credential_cache import OboCredentialCache


@dataclass
class AuthContext:
    user_token: str


CURRENT_AUTH_CONTEXT: ContextVar[Optional[AuthContext]] = ContextVar(
    "current_auth_context", default=None
)


def create_obo_auth_filter(
    mcp_configs: Dict[str, MCPServerConfig],
    azure_ad_config: AzureAdConfig,
) -> Callable:
    factory = FactoryManager.get_factory(azure_ad_config)
    credential_cache = OboCredentialCache(factory)

    def _is_streamable_http_with_user_assertion(config: MCPServerConfig) -> bool:
        return (
            (config.type == "streamable_http" or (config.type is None and bool(config.url)))
            and config.auth is not None
            and bool(config.auth.enable_user_assertion)
        )

    async def _get_token(config: MCPServerConfig, auth_context) -> Optional[str]:
        if not auth_context:
            return None

        try:
            credential = await credential_cache.get_credential(
                user_assertion=auth_context.user_token,
            )
            if config.auth and config.auth.scope:
                return (await credential.get_token(config.auth.scope)).token
            return None
        except Exception:
            return None

    async def obo_auth_filter(
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ):
        config = mcp_configs.get(context.function.plugin_name)
        if not config or not _is_streamable_http_with_user_assertion(config):
            await next(context)
            return

        auth_context = CURRENT_AUTH_CONTEXT.get()
        token = await _get_token(config, auth_context)

        if token and context.arguments is not None:
            context.arguments["user_assertion"] = token
            context.function.parameters.append(
                KernelParameterMetadata(name="user_assertion", include_in_function_choices=False)
            )

        await next(context)

    return obo_auth_filter
