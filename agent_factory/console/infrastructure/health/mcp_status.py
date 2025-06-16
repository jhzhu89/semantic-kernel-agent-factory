import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from ....mcp_server.config import MCPServerConfig


class MCPStatus(Enum):
    UNKNOWN = "unknown"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class MCPServerStatus:
    name: str
    status: MCPStatus
    server_type: str
    last_check: Optional[datetime] = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    connection_time: Optional[datetime] = None


class MCPHealthChecker:
    def __init__(self, configs: Dict[str, MCPServerConfig]):
        self.configs = configs

    async def check_server_health(self, name: str, config: MCPServerConfig) -> bool:
        server_type = config.type or ("streamable_http" if config.url else "stdio")
        if server_type == "streamable_http":
            if config.url is None:
                return False
            return await self._check_streamable_http_health(config.url)
        else:
            return await self._check_stdio_health(config)

    async def _check_streamable_http_health(self, url: str) -> bool:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=2.0)
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, OSError, ConnectionError):
            return False
        except Exception:
            return False

    async def _check_stdio_health(self, config: MCPServerConfig) -> bool:
        process = None
        try:
            if config.command is None:
                return False
            process = await asyncio.create_subprocess_exec(
                config.command,
                *config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **config.env},
            )
            try:
                await asyncio.wait_for(process.communicate(b""), timeout=3.0)
                return process.returncode == 0
            except asyncio.TimeoutError:
                return False
        except Exception:
            return False
        finally:
            if process and process.returncode is None:
                process.terminate()
                await process.wait()
