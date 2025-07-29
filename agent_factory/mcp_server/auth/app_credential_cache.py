from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Union

from aiocache import SimpleMemoryCache
from azure.identity.aio._internal.get_token_mixin import GetTokenMixin

from .azure_credential_factory import AzureCredentialFactory

logger = logging.getLogger(__name__)


@dataclass
class CachedCredential:
    credential: GetTokenMixin


class AppCredentialCache:
    def __init__(self, factory: AzureCredentialFactory):
        self._factory = factory
        self._cache = SimpleMemoryCache()
        self._fail_backoff = 5
        self._builder_timeout = 10
        self._ttl = 43_200
        self._buffer_seconds = 300

    def _k(self, tenant_id: str, client_id: str) -> str:
        return f"{tenant_id}:{client_id}:app"

    async def get_credential(self, tenant_id: str, client_id: str) -> GetTokenMixin:
        k = self._k(tenant_id, client_id)
        logger.debug(f"Getting app credential for key: {k}")
        entry = await self._cache.get(
            k
        )  # type: Union[Exception, asyncio.Task[GetTokenMixin], CachedCredential]
        if entry:
            if isinstance(entry, Exception):
                logger.debug(f"Cache hit with exception for {k}")
                raise entry
            if isinstance(entry, CachedCredential):
                logger.debug(f"Cache hit with credential for {k}")
                return entry.credential
            logger.debug(f"Cache hit with task for {k}")
            result = await asyncio.shield(entry)
            return result

        logger.debug(f"Cache miss for {k}, creating credential")

        async def _builder():
            try:
                cred = await asyncio.wait_for(
                    self._factory.create_app_credential(),
                    timeout=self._builder_timeout,
                )
                ttl = self._ttl
                if ttl > self._buffer_seconds:
                    ttl -= self._buffer_seconds

                logger.debug(f"Created app credential for {k}, caching with ttl={ttl}s")
                await self._cache.set(k, CachedCredential(cred), ttl=ttl)
                return cred
            except Exception as exc:
                logger.debug(f"Failed to create credential for {k}: {exc}")
                await self._cache.set(k, exc, ttl=self._fail_backoff)
                raise

        task = asyncio.create_task(_builder())  # type: asyncio.Task[GetTokenMixin]

        await self._cache.set(k, task, ttl=self._fail_backoff)
        return await task

    async def invalidate(self, tenant_id: str, client_id: str):
        k = self._k(tenant_id, client_id)
        logger.debug(f"Invalidating app credential for {k}")
        await self._cache.delete(k)

    async def clear(self):
        await self._cache.clear()
