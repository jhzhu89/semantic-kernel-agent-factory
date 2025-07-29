from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Union

from aiocache import SimpleMemoryCache
from azure.identity.aio._internal.get_token_mixin import GetTokenMixin

from .azure_credential_factory import AzureCredentialFactory
from .token_parser import TokenInfo, TokenParser

logger = logging.getLogger(__name__)


@dataclass
class CachedCredential:
    credential: GetTokenMixin
    token_info: TokenInfo


class OboCredentialCache:
    def __init__(self, factory: AzureCredentialFactory):
        self._factory = factory
        self._cache = SimpleMemoryCache()
        self._fail_backoff = 5
        self._builder_timeout = 10
        self._buffer_seconds = 30
        self._parser = TokenParser()

    def _k(self, info: TokenInfo) -> str:
        return f"{info.tenant_id}:{info.client_id}:obo:{info.user_id}"

    async def get_credential(self, user_assertion: str) -> GetTokenMixin:
        info = self._parser.parse_token(user_assertion)
        k = self._k(info)
        logger.debug(f"Getting OBO credential for key: {k}")
        entry = await self._cache.get(
            k
        )  # type: Union[Exception, asyncio.Task[GetTokenMixin], CachedCredential]
        if entry:
            if isinstance(entry, Exception):
                logger.debug(f"Cache hit with exception for {k}")
                raise entry
            if isinstance(entry, CachedCredential):
                if entry.token_info.expiry <= datetime.utcnow():
                    logger.debug(f"Token expired for {k}, removing from cache")
                    await self._cache.delete(k)
                else:
                    ttl = int((entry.token_info.expiry - datetime.utcnow()).total_seconds())
                    logger.debug(f"Cache hit {k}: token_exp={entry.token_info.expiry}, ttl={ttl}s")
                    return entry.credential
            if isinstance(entry, asyncio.Task):
                logger.debug(f"Cache hit with task for {k}")
                result = await asyncio.shield(entry)
                return result

        logger.debug(f"Cache miss for {k}, creating OBO credential")

        async def _builder():
            try:
                cred = await asyncio.wait_for(
                    self._factory.create_obo_credential(user_assertion),
                    timeout=self._builder_timeout,
                )
                ttl = int((info.expiry - datetime.utcnow()).total_seconds())
                if ttl > self._buffer_seconds:
                    ttl -= self._buffer_seconds

                if ttl <= 0:
                    logger.warning(f"Token for {k} is expired, not caching")
                    await self._cache.delete(k)
                    return cred

                logger.debug(f"Created OBO {k}: token_exp={info.expiry}, caching with ttl={ttl}s")
                await self._cache.set(k, CachedCredential(cred, info), ttl=ttl)
                return cred
            except Exception as exc:
                logger.debug(f"Failed to create OBO credential for {k}: {exc}")
                await self._cache.set(k, exc, ttl=self._fail_backoff)
                raise

        task = asyncio.create_task(_builder())  # type: asyncio.Task[GetTokenMixin]
        await self._cache.set(k, task, ttl=self._fail_backoff)
        return await task

    async def invalidate(self, user_assertion: str):
        info = self._parser.parse_token(user_assertion)
        k = self._k(info)
        logger.debug(f"Invalidating OBO credential for {k}")
        await self._cache.delete(k)

    async def clear(self):
        await self._cache.clear()
