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
    # Version
    "__version__",
]
