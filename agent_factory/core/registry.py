from __future__ import annotations

import logging
from typing import Dict

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from .config import AzureOpenAIConfig, ModelSelectStrategy

logger = logging.getLogger(__name__)


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

        self._add_builtin_plugins(kernel)
        return kernel

    def _create_service(self, config: AzureOpenAIConfig) -> AzureChatCompletion:
        if config.api_key is not None:
            logger.info(f"Using API key authentication for model {config.model}")
            return AzureChatCompletion(
                service_id=config.model,
                endpoint=str(config.endpoint),
                deployment_name=config.model,
                api_version=config.api_version,
                api_key=config.api_key.get_secret_value(),
            )
        else:
            logger.info(f"Using Azure credential authentication for model {config.model}")
            return AzureChatCompletion(
                service_id=config.model,
                endpoint=str(config.endpoint),
                deployment_name=config.model,
                api_version=config.api_version,
                ad_token_provider=self._create_azure_token_provider(),
            )

    def _create_azure_token_provider(self):
        try:
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                ManagedIdentityCredential,
            )

            cli_credential = AzureCliCredential()
            mi_credential = ManagedIdentityCredential()

            credential = ChainedTokenCredential(cli_credential, mi_credential)

            def get_token():
                access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
                return access_token.token

            logger.debug("Created Azure credential chain: CLI + Managed Identity")
            return get_token

        except ImportError as e:
            logger.info(
                "Azure Identity library not found. Install with: pip install azure-identity"
            )
            raise RuntimeError(
                "Azure Identity library required for credential authentication"
            ) from e
        except Exception as e:
            logger.info(f"Azure credential authentication failed: {e}")
            raise

    def _add_builtin_plugins(self, kernel: Kernel):
        try:
            from semantic_kernel.core_plugins import TimePlugin

            kernel.add_plugin(TimePlugin(), plugin_name="time")
            logger.debug("Added TimePlugin to kernel as 'time'")
        except ImportError:
            logger.debug("TimePlugin not available, skipping")
        except Exception as e:
            logger.debug(f"Failed to add TimePlugin: {e}")

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
