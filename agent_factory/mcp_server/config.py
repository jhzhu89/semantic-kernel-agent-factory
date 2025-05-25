from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class MCPServerConfig(BaseModel):
    type: Optional[Literal["sse", "stdio"]] = None
    timeout: int = 5
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = []
    env: Dict[str, str] = {}
    description: Optional[str] = None
