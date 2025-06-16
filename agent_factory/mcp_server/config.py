from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class AuthConfig(BaseModel):
    enabled: bool = False
    type: str = "bearer"
    scope: str


class AzureAdConfig(BaseModel):
    certificate_name: Optional[str] = None
    client_secret: Optional[str] = None
    client_id: Optional[str] = None
    tenant_id: Optional[str] = None


class OnBehalfOfAuth(BaseModel):
    azure_ad: Optional[AzureAdConfig] = None


class MCPAuthConfig(BaseModel):
    on_behalf_of: Optional[OnBehalfOfAuth] = None


class MCPServerConfig(BaseModel):
    type: Optional[Literal["streamable_http", "stdio"]] = None
    timeout: int = 5
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = []
    env: Dict[str, str] = {}
    description: Optional[str] = None
    auth: Optional[AuthConfig] = None


class MCPConfig(BaseModel):
    servers: Dict[str, MCPServerConfig] = {}
    auth: Optional[MCPAuthConfig] = None
