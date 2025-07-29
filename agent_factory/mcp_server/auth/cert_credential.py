from __future__ import annotations

import asyncio
import functools
from concurrent.futures import Executor
from types import TracebackType
from typing import Any, Awaitable, Optional, Type, TypeVar, Union

from azure.core.credentials import AccessToken, AccessTokenInfo
from azure.core.credentials_async import AsyncContextManager
from azure.identity import CertificateCredential
from azure.identity.aio._internal.get_token_mixin import GetTokenMixin

T = TypeVar("T", bound="AsyncCertificateCredential")


class AsyncCertificateCredential(AsyncContextManager, GetTokenMixin):
    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        certificate_data: bytes,
        *,
        password: Optional[str] = None,
        send_certificate_chain: bool = True,
        executor: Optional[Executor] = None,
    ):
        self._sync_cred = CertificateCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            certificate_data=certificate_data,
            password=password,
            send_certificate_chain=send_certificate_chain,
        )
        self._executor = executor
        super().__init__()

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._sync_cred.close)

    async def _acquire_token_silently(
        self, *scopes: str, **kwargs: Any
    ) -> Optional[AccessTokenInfo]:
        return None

    async def _request_token(self, *scopes: str, **kwargs: Any) -> AccessTokenInfo:
        loop = asyncio.get_running_loop()
        token: AccessToken = await loop.run_in_executor(
            self._executor,
            functools.partial(self._sync_cred.get_token, *scopes, **kwargs),
        )
        return AccessTokenInfo(token=token.token, expires_on=token.expires_on)

    def get_token_sync(self, *scopes: str, **kwargs: Any) -> AccessToken:
        return asyncio.run(self.get_token(*scopes, **kwargs))
