from __future__ import annotations

import enum
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..mcp_server.config import MCPConfig

AGENT_NAME_PATTERN = r"^[0-9A-Za-z_-]+$"


class AzureOpenAIConfig(BaseSettings):
    model: Optional[str] = Field(default=None)
    api_key: Optional[SecretStr] = Field(default=None)
    api_version: str = Field(default="2024-06-01")
    endpoint: Optional[HttpUrl] = Field(default=None)

    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )


class ResponseSchema(BaseModel):
    json_schema_definition: Dict[str, Any]
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "schema": self.json_schema_definition,
                "name": self.name,
                "strict": True,
            },
        }


class ModelSettings(BaseModel):
    temperature: float = 1.0
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    response_json_schema: Optional[ResponseSchema] = None


class ModelSelectStrategy(str, enum.Enum):
    first = "first"
    cost = "cost"
    latency = "latency"
    quality = "quality"


class AgentConfig(BaseModel):
    name: Optional[str] = Field(default=None, pattern=AGENT_NAME_PATTERN)
    instructions: str
    mcp_servers: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    model_settings: Optional[ModelSettings] = None


class AgentFactoryConfig(BaseSettings):
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    mcp: Optional[MCPConfig] = None
    openai_models: Dict[str, AzureOpenAIConfig] = Field(default_factory=dict)
    model_selection: ModelSelectStrategy = ModelSelectStrategy.first
    mcp_failure_strategy: Literal["strict", "lenient"] = Field(
        default="strict",
        description="How to handle MCP plugin connection failures. 'strict': exit on any failure, 'lenient': continue with working plugins",
    )

    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_prefix="AGENT_FACTORY_", env_file=".env", extra="ignore"
    )

    @model_validator(mode="after")
    def validate_references(self):
        self._validate_mcp_servers()
        self._validate_agent_names()
        self._validate_model_names()
        return self

    def _validate_mcp_servers(self):
        if not self.mcp or not self.mcp.servers:
            return

        missing_servers = {
            server
            for agent in self.agents.values()
            for server in agent.mcp_servers
            if server not in self.mcp.servers
        }
        if missing_servers:
            raise ValueError(f"Undefined MCP servers: {missing_servers}")

    def _validate_agent_names(self):
        name_pattern = re.compile(AGENT_NAME_PATTERN)

        for key, agent in self.agents.items():
            if not name_pattern.match(key):
                raise ValueError(f"Agent key '{key}' must match pattern '{AGENT_NAME_PATTERN}'")

            if agent.name is None:
                agent.name = key
            elif agent.name != key:
                raise ValueError(f"Agent key '{key}' does not match name '{agent.name}'")

    def _validate_model_names(self):
        for key, model_config in self.openai_models.items():
            if model_config.model is None:
                model_config.model = key
            elif model_config.model != key:
                raise ValueError(
                    f"Model key '{key}' does not match model name '{model_config.model}'"
                )

    @classmethod
    def from_file(cls, path: str | Path):
        import json

        import yaml

        file_path = Path(path)

        with open(file_path, "r") as f:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif file_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}. Use .yaml, .yml, or .json"
                )

        return cls.model_validate(data)
