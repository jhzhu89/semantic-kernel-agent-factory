import base64
import logging
import time
import uuid
from typing import Dict, Optional

import jwt
from azure.identity.aio import ClientSecretCredential, OnBehalfOfCredential
from azure.identity.aio._internal.get_token_mixin import GetTokenMixin
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding, load_pem_private_key

from agent_factory.mcp_server.auth.cert_credential import AsyncCertificateCredential

logger = logging.getLogger(__name__)


class AzureCredentialFactory:
    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: Optional[str] = None,
        certificate_pem: Optional[str] = None,
    ):
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._certificate_pem = certificate_pem

        if not client_secret and not certificate_pem:
            raise ValueError("Either client_secret or certificate_pem must be provided")

    async def create_app_credential(self) -> GetTokenMixin:
        if self._client_secret:
            logger.debug(f"Creating ClientSecretCredential for tenant {self._tenant_id}")
            return ClientSecretCredential(
                tenant_id=self._tenant_id,
                client_id=self._client_id,
                client_secret=self._client_secret,
            )
        elif self._certificate_pem:
            logger.debug(f"Creating CertificateCredential for tenant {self._tenant_id}")
            try:
                pem_bytes = self._certificate_pem.encode("utf-8")
                return AsyncCertificateCredential(
                    tenant_id=self._tenant_id,
                    client_id=self._client_id,
                    certificate_data=pem_bytes,
                    send_certificate_chain=True,
                )
            except Exception as e:
                logger.error(f"Failed to create CertificateCredential: {e}")
                raise
        else:
            raise ValueError("Either client_secret or certificate_pem must be provided")

    async def create_obo_credential(self, user_assertion: str) -> GetTokenMixin:
        if self._client_secret:
            return OnBehalfOfCredential(
                tenant_id=self._tenant_id,
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_assertion=user_assertion,
            )
        elif self._certificate_pem:
            assertion_func = self._build_assertion_func()
            return OnBehalfOfCredential(
                tenant_id=self._tenant_id,
                client_id=self._client_id,
                client_assertion_func=assertion_func,
                user_assertion=user_assertion,
            )
        else:
            raise ValueError("Either client_secret or certificate_pem must be provided")

    def _build_assertion_func(self):
        if self._certificate_pem is None:
            raise ValueError("Certificate PEM is required for certificate-based authentication")
        private_key, certificate = self._parse_pem(self._certificate_pem)

        cert_der = certificate.public_bytes(Encoding.DER)
        x5c_chain = [base64.b64encode(cert_der).decode()]

        sha1_digest = hashes.Hash(hashes.SHA1())
        sha1_digest.update(cert_der)
        sha1_thumbprint = base64.urlsafe_b64encode(sha1_digest.finalize()).decode().rstrip("=")

        def build_assertion() -> str:
            now = int(time.time())
            payload = {
                "aud": f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token",
                "iss": self._client_id,
                "sub": self._client_id,
                "jti": str(uuid.uuid4()),
                "iat": now,
                "exp": now + 600,
            }
            headers = {"alg": "RS256", "x5c": x5c_chain, "x5t": sha1_thumbprint}
            return jwt.encode(payload, private_key, algorithm="RS256", headers=headers)

        return build_assertion

    def _parse_pem(self, cert_pem: str):
        pem_bytes = cert_pem.encode("utf-8")
        private_key = load_pem_private_key(pem_bytes, password=None)
        certificate = x509.load_pem_x509_certificate(pem_bytes)
        return private_key, certificate


class FactoryManager:
    _instances: Dict[str, AzureCredentialFactory] = {}

    @classmethod
    def get_factory(cls, azure_ad_config) -> AzureCredentialFactory:
        key = f"{azure_ad_config.tenant_id}:{azure_ad_config.client_id}"
        if key not in cls._instances:
            cls._instances[key] = AzureCredentialFactory(
                azure_ad_config.tenant_id,
                azure_ad_config.client_id,
                client_secret=azure_ad_config.client_secret,
                certificate_pem=azure_ad_config.certificate_pem,
            )
        return cls._instances[key]


def create_azure_credential_factory(azure_ad_config) -> AzureCredentialFactory:
    return FactoryManager.get_factory(azure_ad_config)
