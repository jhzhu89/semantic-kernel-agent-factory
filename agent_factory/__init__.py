from importlib.metadata import version as _v

# Import core functionality (always available)
from .core import (
    AgentConfig,
    AgentFactory,
    AgentFactoryConfig,
    AzureOpenAIConfig,
    ModelSelectStrategy,
    ModelSettings,
    ResponseSchema,
    ServiceRegistry,
)

# Import MCP server auth functionality
from .mcp_server.auth import CURRENT_AUTH_CONTEXT, AuthContext, S2SAuth

try:
    __version__ = _v("semantic-kernel-agent-factory")
except Exception:
    # Fallback to version file when package is not installed
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.1"

# Core exports (always available)
__all__ = [
    # Core factory classes
    "AgentFactory",
    "ServiceRegistry",
    # Configuration classes
    "AgentConfig",
    "AgentFactoryConfig",
    "AzureOpenAIConfig",
    "ModelSettings",
    "ModelSelectStrategy",
    "ResponseSchema",
    # Auth classes
    "AuthContext",
    "CURRENT_AUTH_CONTEXT",
    "S2SAuth",
    # Version
    "__version__",
]
