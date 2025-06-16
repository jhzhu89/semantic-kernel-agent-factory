import logging
import warnings
from contextvars import ContextVar
from typing import NamedTuple, Optional

from .base import AuthHandler
from .credential_cache import CredentialCache

logger = logging.getLogger(__name__)


class AuthContext(NamedTuple):
    user_token: str


CURRENT_AUTH_CONTEXT: ContextVar[Optional[AuthContext]] = ContextVar(
    "current_auth_context", default=None
)


class AzureOboAuth(AuthHandler):
    """
    DEPRECATED: This class cannot correctly retrieve access_token from contextvar.
    Use the filter-based approach in filters.py instead.
    """

    def __init__(self, scope: str, credential_cache: CredentialCache):
        warnings.warn(
            "AzureOboAuth is deprecated. This class cannot correctly retrieve access_token "
            "from contextvar. Use create_on_behalf_of_auth_filter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._scope = scope
        self._cache = credential_cache
        logger.info(f"AzureOboAuth initialized - scope: {scope}")

    async def get_token(self) -> str:
        auth_context = CURRENT_AUTH_CONTEXT.get()
        if not auth_context:
            logger.error("Auth context not available")
            raise RuntimeError("Auth context not set in CURRENT_AUTH_CONTEXT")

        logger.debug(f"Requesting OBO token")
        credential = await self._cache.get_credential(auth_context.user_token)

        if not credential:
            logger.error(f"Failed to get credential")
            raise RuntimeError(f"Failed to get credential")

        try:
            token_response = await credential.get_token(self._scope)
            logger.debug(f"OBO token obtained, expires: {token_response.expires_on}")
            return token_response.token
        except Exception as e:
            logger.error(f"OBO token request failed: {e}")
            raise

    def sync_auth_flow(self, request):
        raise NotImplementedError("Synchronous auth flow not supported for Azure OBO")

    async def async_auth_flow(self, request):
        request_id = id(request)
        logger.debug(f"Auth flow starting for request: {request_id}")

        try:
            token = await self.get_token()
            request.headers["Authorization"] = f"Bearer {token}"

            response = yield request

            if response.status_code == 401:
                logger.warning(f"Request {request_id} got 401, invalidating cache and retrying")
                auth_context = CURRENT_AUTH_CONTEXT.get()

                if auth_context:
                    await self._cache.invalidate(auth_context.user_token)

                token = await self.get_token()
                request.headers["Authorization"] = f"Bearer {token}"
                yield request
            else:
                logger.debug(f"Request {request_id} completed with status: {response.status_code}")

        except Exception as e:
            logger.error(f"Auth flow failed for request {request_id}: {e}")
            raise
