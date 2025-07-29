from .app_credential_cache import AppCredentialCache
from .azure_credential_factory import (
    AzureCredentialFactory,
    FactoryManager,
    create_azure_credential_factory,
)
from .obo_auth_filter import CURRENT_AUTH_CONTEXT, AuthContext, create_obo_auth_filter
from .obo_credential_cache import OboCredentialCache
from .s2s_auth import S2SAuthHandler, S2SAuthManager

# Create alias for backward compatibility if needed
S2SAuth = S2SAuthHandler
from .token_parser import TokenInfo, TokenParser

__all__ = [
    "AuthContext",
    "CURRENT_AUTH_CONTEXT",
    "create_obo_auth_filter",
    "S2SAuthHandler",
    "S2SAuthManager",
    "S2SAuth",  # Alias for backward compatibility
    "TokenInfo",
    "TokenParser",
    "OboCredentialCache",
    "AppCredentialCache",
    "AzureCredentialFactory",
    "create_azure_credential_factory",
    "FactoryManager",
]
