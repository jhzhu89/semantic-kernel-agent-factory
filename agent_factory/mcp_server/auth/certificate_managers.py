import hashlib
import logging
from pathlib import Path
from typing import Dict

from .base import CertificateManager

logger = logging.getLogger(__name__)


class FileCertificateManager(CertificateManager):

    def __init__(self, cert_directory: str = "."):
        self.cert_directory = Path(cert_directory).resolve()
        self._file_hashes: Dict[str, str] = {}
        logger.info(f"FileCertificateManager initialized with directory: {self.cert_directory}")

    async def get_certificate(self, cert_name: str) -> bytes:
        logger.debug(f"Searching for certificate: {cert_name}")
        cert_extensions = [".pfx", ".p12", ".pem", ".crt", ".cer"]
        cert_path = None

        for ext in cert_extensions:
            potential_path = self.cert_directory / f"{cert_name}{ext}"
            if potential_path.exists():
                cert_path = potential_path
                logger.debug(f"Found certificate at: {cert_path}")
                break

        if not cert_path:
            logger.error(f"Certificate not found: {cert_name} in {self.cert_directory}")
            raise FileNotFoundError(
                f"Certificate file not found for '{cert_name}' in {self.cert_directory}. "
                f"Tried extensions: {cert_extensions}"
            )

        try:
            with open(cert_path, "rb") as f:
                cert_data = f.read()
            logger.debug(f"Successfully read certificate {cert_name}, size: {len(cert_data)} bytes")
            return cert_data
        except Exception as e:
            logger.error(f"Failed to read certificate file {cert_path}: {e}")
            raise ValueError(f"Failed to read certificate file {cert_path}: {e}")

    async def get_certificate_version(self, cert_name: str) -> str:
        cert_data = await self.get_certificate(cert_name)
        current_hash = hashlib.sha256(cert_data).hexdigest()

        previous_hash = self._file_hashes.get(cert_name)
        if previous_hash != current_hash:
            self._file_hashes[cert_name] = current_hash
            if previous_hash:
                logger.info(
                    f"Certificate {cert_name} version changed: {previous_hash[:8]} -> {current_hash[:8]}"
                )
            else:
                logger.debug(
                    f"First version tracking for certificate {cert_name}: {current_hash[:8]}"
                )

        return current_hash
