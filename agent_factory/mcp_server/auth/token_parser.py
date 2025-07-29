import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import jwt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenInfo:
    user_id: str
    tenant_id: str
    client_id: str
    expiry: datetime

    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expiry


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
