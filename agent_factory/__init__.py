from importlib.metadata import version as _v

from .config import (
    A2AAgentConfig,
    A2AServiceConfig,
    AgentConfig,
    AgentFactoryConfig,
    AgentServiceFactoryConfig,
    ConfigurableAgentCard,
)
from .executor import SemanticKernelAgentExecutor
from .factory import AgentFactory
from .function_events import (
    FunctionCallEvent,
    FunctionEvent,
    FunctionEventType,
    FunctionResultEvent,
)
from .service_factory import AgentServiceFactory

try:
    __version__ = _v("semantic-kernel-agent-factory")
except Exception:
    # Fallback to version file when package is not installed
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.1"

__all__ = [
    "AgentFactory",
    "AgentServiceFactory",
    "AgentFactoryConfig",
    "AgentConfig",
    "ConfigurableAgentCard",
    "A2AServiceConfig",
    "A2AAgentConfig",
    "AgentServiceFactoryConfig",
    "SemanticKernelAgentExecutor",
    "FunctionCallEvent",
    "FunctionResultEvent",
    "FunctionEvent",
    "FunctionEventType",
    "__version__",
]
