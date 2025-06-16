from abc import ABC, abstractmethod

import httpx


class AuthHandler(httpx.Auth, ABC):
    requires_request_body = False

    @abstractmethod
    async def get_token(self) -> str:
        pass


class CertificateManager(ABC):

    @abstractmethod
    async def get_certificate(self, cert_name: str) -> bytes:
        pass

    @abstractmethod
    async def get_certificate_version(self, cert_name: str) -> str:
        pass
