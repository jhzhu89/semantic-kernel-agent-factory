import logging
from abc import ABC, abstractmethod
from typing import Dict

import httpx

from ..config import AzureAdConfig
from .app_credential_cache import AppCredentialCache
from .azure_credential_factory import FactoryManager

logger = logging.getLogger(__name__)


class AuthHandler(httpx.Auth, ABC):
    requires_request_body = False

    @abstractmethod
    async def get_token(self) -> str:
        pass


class S2SAuthHandler(AuthHandler):
    def __init__(self, scope: str, app_cache: AppCredentialCache, azure_ad_config: AzureAdConfig):
        # Replace specific scope with /.default for S2S auth
        if "/" in scope and not scope.endswith("/.default"):
            # Extract the base URL part before the slash
            base_url = scope.rsplit("/", 1)[0]
            self._scope = base_url + "/.default"
        elif not scope.endswith("/.default"):
            self._scope = scope + "/.default"
        else:
            self._scope = scope
        self._app_cache = app_cache
        self._azure_ad_config = azure_ad_config
        logger.debug(
            f"S2SAuthHandler initialized: original scope='{scope}' -> final scope='{self._scope}'"
        )

    async def get_token(self) -> str:
        credential = await self._app_cache.get_credential(
            self._azure_ad_config.tenant_id, self._azure_ad_config.client_id
        )

        token_response = await credential.get_token(self._scope)
        return token_response.token

    def sync_auth_flow(self, request):
        raise NotImplementedError("Synchronous auth flow not supported for S2S")

    async def async_auth_flow(self, request):
        request_id = id(request)
        logger.debug(f"S2S auth flow starting for request: {request_id}")

        try:
            token = await self.get_token()
            request.headers["Authorization"] = f"Bearer {token}"

            response = yield request

            if response.status_code == 401:
                logger.warning(f"Request {request_id} got 401, retrying with fresh token")

                token = await self.get_token()
                request.headers["Authorization"] = f"Bearer {token}"
                yield request
            else:
                logger.debug(
                    f"Request {request_id} completed with status: {response.status_code}, body: {response}"
                )

        except Exception as e:
            logger.error(f"S2S auth flow failed for request {request_id}: {e}")
            raise


class S2SAuthManager:
    def __init__(self, azure_ad_config: AzureAdConfig):
        self._azure_ad_config = azure_ad_config
        factory = FactoryManager.get_factory(azure_ad_config)
        self._app_cache = AppCredentialCache(factory)
        self._handlers: Dict[str, S2SAuthHandler] = {}
        logger.info(f"S2SAuthManager initialized")

    def get_auth_handler(self, scope: str) -> S2SAuthHandler:
        if scope not in self._handlers:
            self._handlers[scope] = S2SAuthHandler(scope, self._app_cache, self._azure_ad_config)
        return self._handlers[scope]
