from typing import Awaitable, Callable, Dict, Optional

from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata

from ..config import AzureAdConfig, MCPServerConfig
from .azure_obo import CURRENT_AUTH_CONTEXT
from .certificate_managers import FileCertificateManager
from .credential_cache import CertificateCredentialCache, CredentialCache, SecretCredentialCache


def create_on_behalf_of_auth_filter(
    mcp_configs: Dict[str, MCPServerConfig],
    azure_ad_config: AzureAdConfig,
    cert_directory: str = ".",
) -> Callable:
    def _create_credential_cache() -> CredentialCache:
        if azure_ad_config.certificate_name:
            cert_manager = FileCertificateManager(cert_directory)
            return CertificateCredentialCache(cert_manager, azure_ad_config.certificate_name)

        if not azure_ad_config.client_secret:
            raise ValueError("Either certificate_name or client_secret must be provided")
        return SecretCredentialCache(azure_ad_config.client_secret)

    def _is_streamable_http_with_auth(config: MCPServerConfig) -> bool:
        return (
            (config.type == "streamable_http" or (config.type is None and bool(config.url)))
            and config.auth is not None
            and bool(config.auth.enabled)
        )

    async def _get_obo_token(config: MCPServerConfig, user_token: str) -> Optional[str]:
        try:
            credential_cache = _create_credential_cache()
            credential = await credential_cache.get_credential(user_token)
            if not credential or not config.auth:
                return None
            token_response = await credential.get_token(config.auth.scope)
            return token_response.token
        except Exception:
            return None

    async def on_behalf_of_auth_filter(
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ):
        config = mcp_configs.get(context.function.plugin_name)
        if not config or not _is_streamable_http_with_auth(config):
            await next(context)
            return

        auth_context = CURRENT_AUTH_CONTEXT.get()
        if not auth_context:
            await next(context)
            return

        obo_token = await _get_obo_token(config, auth_context.user_token)
        if obo_token and context.arguments is not None:
            context.arguments["access_token"] = obo_token
            context.function.parameters.append(
                KernelParameterMetadata(name="access_token", include_in_function_choices=False)
            )

        await next(context)

    return on_behalf_of_auth_filter
