from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ....core.config import AgentFactoryConfig


class AgentHistoryConfig(BaseModel):
    threshold_count: int = Field(
        default=1000, description="Chat history threshold for summarization"
    )
    target_count: int = Field(
        default=10, description="Target number of messages after summarization"
    )


class ChatHistoryConfig(BaseModel):
    agents: Dict[str, AgentHistoryConfig] = Field(
        default_factory=dict, description="Per-agent history configuration"
    )


class AgentFactoryCliConfig(BaseSettings):
    agent_factory: AgentFactoryConfig = Field(..., description="Agent factory configuration")
    chat_history: ChatHistoryConfig = Field(
        default_factory=ChatHistoryConfig, description="Chat history configuration"
    )

    model_config = SettingsConfigDict(
        env_prefix="CLI_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    @model_validator(mode="after")
    def validate_agent_history_consistency(self):
        self._validate_history_agent_mapping()
        return self

    def _validate_history_agent_mapping(self):
        history_keys = set(self.chat_history.agents.keys())
        agent_keys = set(self.agent_factory.agents.keys())

        missing_agents = history_keys - agent_keys
        if missing_agents:
            raise ValueError(
                f"Chat history keys {missing_agents} do not have corresponding agents in agent_factory. Available agents: {sorted(agent_keys)}"
            )

    def get_agent_history_config(self, agent_name: str) -> Optional[AgentHistoryConfig]:
        return self.chat_history.agents.get(agent_name)

    @classmethod
    def from_file(cls, path: str | Path):
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
