import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict

import jwt
from aiocache import Cache
from azure.identity.aio import OnBehalfOfCredential

from .base import CertificateManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenInfo:
    user_id: str
    tenant_id: str
    client_id: str
    expiry: datetime

    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expiry


@dataclass
class CachedCredential:
    credential: OnBehalfOfCredential
    token_info: TokenInfo

    def is_valid(self) -> bool:
        return not self.token_info.is_expired()


class TokenParser:
    @staticmethod
    def parse_token(user_assertion: str) -> TokenInfo:
        try:
            decoded = jwt.decode(user_assertion, options={"verify_signature": False})
            user_id = decoded.get("oid") or decoded.get("sub") or decoded.get("upn") or "unknown"
            tenant_id = decoded.get("tid") or "unknown"
            client_id = decoded.get("aud") or decoded.get("appid") or "unknown"
            expiry = datetime.fromtimestamp(decoded["exp"])

            return TokenInfo(
                user_id=user_id, tenant_id=tenant_id, client_id=client_id, expiry=expiry
            )
        except Exception as e:
            logger.warning(f"Failed to parse token, using fallback values: {e}")
            fallback_id = hashlib.sha256(user_assertion.encode()).hexdigest()[:16]
            fallback_expiry = datetime.utcnow() + timedelta(hours=1)
            return TokenInfo(
                user_id=fallback_id,
                tenant_id="unknown",
                client_id="unknown",
                expiry=fallback_expiry,
            )


class CredentialCache(ABC):

    def __init__(self, auth_type: str):
        self._auth_type = auth_type
        self._cache: Cache = Cache(Cache.MEMORY)
        self._creation_locks: Dict[str, asyncio.Lock] = {}
        self._creation_locks_lock = asyncio.Lock()
        self._token_parser = TokenParser()
        logger.info(f"CredentialCache ({auth_type}) initialized")

    def _make_cache_key(self, tenant_id: str, client_id: str, user_id: str) -> str:
        return f"{tenant_id}:{client_id}:{self._auth_type}:{user_id}"

    async def get_credential(self, user_assertion: str) -> OnBehalfOfCredential:
        token_info = self._token_parser.parse_token(user_assertion)
        cache_key = self._make_cache_key(
            token_info.tenant_id, token_info.client_id, token_info.user_id
        )

        cached_credential = await self._cache.get(cache_key)
        if cached_credential is not None and isinstance(cached_credential, CachedCredential):
            if cached_credential.is_valid():
                logger.debug(f"Cache hit for key: {cache_key}")
                credential: OnBehalfOfCredential = cached_credential.credential
                return credential
            else:
                logger.debug(f"Cached credential expired, creating new one")

        return await self._create_and_cache(cache_key, token_info, user_assertion)

    async def _get_creation_lock(self, cache_key: str) -> asyncio.Lock:
        async with self._creation_locks_lock:
            if cache_key not in self._creation_locks:
                self._creation_locks[cache_key] = asyncio.Lock()
            return self._creation_locks[cache_key]

    async def _create_and_cache(
        self, cache_key: str, token_info: TokenInfo, user_assertion: str
    ) -> OnBehalfOfCredential:
        creation_lock = await self._get_creation_lock(cache_key)

        async with creation_lock:
            # Double-check cache to avoid race conditions
            cached_credential = await self._cache.get(cache_key)
            if (
                cached_credential is not None
                and isinstance(cached_credential, CachedCredential)
                and cached_credential.is_valid()
            ):
                credential: OnBehalfOfCredential = cached_credential.credential
                return credential

            try:
                credential = await self._create_credential(
                    token_info.tenant_id, token_info.client_id, user_assertion
                )
                if not credential:
                    logger.error(f"Credential creation returned None for key: {cache_key}")
                    raise ValueError(f"Failed to create credential for key: {cache_key}")

                cached_credential = CachedCredential(credential=credential, token_info=token_info)
                ttl = int(token_info.expiry.timestamp())
                await self._cache.set(cache_key, cached_credential, ttl=ttl)

                logger.debug(f"Credential created and cached for key: {cache_key}")
                return credential

            except Exception as e:
                logger.error(f"Failed to create credential for key {cache_key}: {e}", exc_info=True)
                raise RuntimeError(f"Credential creation failed for cache key: {cache_key}") from e

    @abstractmethod
    async def _create_credential(
        self, tenant_id: str, client_id: str, user_assertion: str
    ) -> OnBehalfOfCredential:
        pass

    async def invalidate(self, user_assertion: str):
        token_info = self._token_parser.parse_token(user_assertion)
        cache_key = self._make_cache_key(
            token_info.tenant_id, token_info.client_id, token_info.user_id
        )
        await self._cache.delete(cache_key)

    async def clear(self):
        await self._cache.clear()
        async with self._creation_locks_lock:
            self._creation_locks.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        return {"creation_locks": len(self._creation_locks)}


class CertificateCredentialCache(CredentialCache):

    def __init__(self, cert_manager: CertificateManager, cert_name: str):
        super().__init__("certificate")
        self._cert_manager = cert_manager
        self._cert_name = cert_name

    async def _create_credential(
        self, tenant_id: str, client_id: str, user_assertion: str
    ) -> OnBehalfOfCredential:
        try:
            cert_data = await self._cert_manager.get_certificate(self._cert_name)
        except Exception as e:
            raise RuntimeError(
                f"Certificate retrieval failed for certificate: {self._cert_name}"
            ) from e

        return OnBehalfOfCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_certificate=cert_data,
            user_assertion=user_assertion,
        )


class SecretCredentialCache(CredentialCache):

    def __init__(self, client_secret: str):
        super().__init__("secret")
        self._client_secret = client_secret

    async def _create_credential(
        self, tenant_id: str, client_id: str, user_assertion: str
    ) -> OnBehalfOfCredential:
        return OnBehalfOfCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=self._client_secret,
            user_assertion=user_assertion,
        )
