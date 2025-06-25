"""Service factory functionality (optional service integration)."""

try:
    from .config import (
        A2AAgentConfig,
        A2AServiceConfig,
        AgentServiceFactoryConfig,
        ConfigurableAgentCard,
    )
    from .executor import SemanticKernelAgentExecutor
    from .service_factory import AgentServiceFactory

    def is_a2a_available():
        return True

    _SERVICE_AVAILABLE = True
    __all__ = [
        "AgentServiceFactory",
        "AgentServiceFactoryConfig",
        "A2AAgentConfig",
        "A2AServiceConfig",
        "ConfigurableAgentCard",
        "SemanticKernelAgentExecutor",
        "is_a2a_available",
    ]

except ImportError as e:
    _SERVICE_AVAILABLE = False
    _import_error = e

    def _raise_service_import_error(*args, **kwargs):
        raise ImportError(
            "Service functionality requires additional dependencies. "
            "Install with: pip install "
            "'semantic-kernel-agent-factory[service]'"
        ) from _import_error

    # Create dummy classes that raise helpful errors
    class _ServiceImportError:
        def __init__(self, *args, **kwargs):
            _raise_service_import_error()

    AgentServiceFactory = _ServiceImportError  # type: ignore[misc,assignment]
    A2AAgentConfig = _ServiceImportError  # type: ignore[misc,assignment]
    A2AServiceConfig = _ServiceImportError  # type: ignore[misc,assignment]
    ConfigurableAgentCard = _ServiceImportError  # type: ignore[misc,assignment]
    SemanticKernelAgentExecutor = _ServiceImportError  # type: ignore[misc,assignment]  # noqa: E501

    def is_a2a_available():
        return False

    __all__ = [
        "AgentServiceFactory",
        "AgentServiceFactoryConfig",
        "A2AAgentConfig",
        "A2AServiceConfig",
        "ConfigurableAgentCard",
        "SemanticKernelAgentExecutor",
        "is_a2a_available",
    ]
