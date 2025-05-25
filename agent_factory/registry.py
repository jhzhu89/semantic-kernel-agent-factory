from __future__ import annotations

from typing import Dict

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from .config import AzureOpenAIConfig, ModelSelectStrategy


class ServiceRegistry:
    def __init__(self, configs: Dict[str, AzureOpenAIConfig]):
        self._configs = configs
        self._services: Dict[str, AzureChatCompletion] = {}

    def build_kernel(self) -> Kernel:
        kernel = Kernel()
        for config in self._configs.values():
            service = self._create_service(config)
            kernel.add_service(service)
            if config.model is not None:
                self._services[config.model] = service
        return kernel

    def _create_service(self, config: AzureOpenAIConfig) -> AzureChatCompletion:
        return AzureChatCompletion(
            service_id=config.model,
            endpoint=str(config.endpoint),
            deployment_name=config.model,
            api_version=config.api_version,
            api_key=config.api_key.get_secret_value() if config.api_key else None,
        )

    def select(self, strategy: ModelSelectStrategy) -> str:
        service_names = list(self._services.keys())

        if strategy == ModelSelectStrategy.first or len(service_names) == 1:
            return service_names[0]

        if strategy == ModelSelectStrategy.latency:
            return next(
                (name for name in service_names if name.endswith(("-small", "-lite"))),
                service_names[0],
            )

        if strategy == ModelSelectStrategy.cost:
            return next(
                (name for name in service_names if "gpt-3.5" in name or "35" in name),
                service_names[0],
            )

        return service_names[0]
