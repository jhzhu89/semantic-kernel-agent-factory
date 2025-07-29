from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class AuthConfig(BaseModel):
    enable_s2s: bool = False
    enable_user_assertion: bool = True
    scope: str


class MCPServerConfig(BaseModel):
    type: Optional[Literal["streamable_http", "stdio"]] = None
    timeout: int = 5
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = []
    env: Dict[str, str] = {}
    description: Optional[str] = None
    auth: Optional[AuthConfig] = None


class AzureAdConfig(BaseModel):
    certificate_pem: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: str
    client_id: str


class MCPAuthConfig(BaseModel):
    azure_ad: Optional[AzureAdConfig] = None


class MCPConfig(BaseModel):
    servers: Dict[str, MCPServerConfig] = {}
    auth: Optional[MCPAuthConfig] = None
