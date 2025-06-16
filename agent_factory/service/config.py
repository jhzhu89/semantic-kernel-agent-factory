from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from a2a.types import AgentCapabilities, AgentCard, AgentProvider, AgentSkill, SecurityScheme
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..core.config import AgentFactoryConfig


class ConfigurableAgentCard(BaseModel):
    name: str = Field(default="UnnamedAgent")
    description: str = Field(default="No description provided")
    url: str = Field(default="http://localhost:8000")
    version: str = Field(default="1.0.0")
    capabilities: Optional[AgentCapabilities] = None
    defaultInputModes: List[str] = Field(default_factory=lambda: ["text/plain"])
    defaultOutputModes: List[str] = Field(default_factory=lambda: ["text/plain"])
    skills: List[AgentSkill] = Field(default_factory=list)
    provider: Optional[AgentProvider] = None
    documentationUrl: Optional[str] = None
    securitySchemes: Optional[Dict[str, SecurityScheme]] = None
    security: Optional[List[Dict[str, List[str]]]] = None
    supportsAuthenticatedExtendedCard: Optional[bool] = None

    def to_agent_card(self) -> "AgentCard":
        from a2a.types import AgentCapabilities, AgentCard

        capabilities = self.capabilities
        if capabilities is None:
            capabilities = AgentCapabilities(
                pushNotifications=False, stateTransitionHistory=False, streaming=True
            )

        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            version=self.version,
            capabilities=capabilities,
            defaultInputModes=self.defaultInputModes,
            defaultOutputModes=self.defaultOutputModes,
            skills=self.skills,
            provider=self.provider,
            documentationUrl=self.documentationUrl,
            securitySchemes=self.securitySchemes,
            security=self.security,
            supportsAuthenticatedExtendedCard=self.supportsAuthenticatedExtendedCard,
        )


class A2AAgentConfig(BaseModel):
    card: ConfigurableAgentCard = Field(..., description="Agent card configuration")
    chat_history_threshold: int = Field(
        default=1000, description="Chat history threshold for summarization"
    )
    chat_history_target: int = Field(
        default=10, description="Target number of messages after summarization"
    )
    path_prefix: Optional[str] = Field(default=None, description="URL path prefix for this agent")
    enable_token_streaming: bool = Field(
        default=False, description="Enable token-level streaming for responses"
    )


class A2AServiceConfig(BaseModel):
    services: Dict[str, A2AAgentConfig] = Field(..., description="Configuration for each agent")


class AgentServiceFactoryConfig(BaseSettings):
    service_factory: A2AServiceConfig = Field(..., description="A2A service configuration")
    agent_factory: AgentFactoryConfig = Field(..., description="Agent factory configuration")

    model_config = SettingsConfigDict(
        env_prefix="SERVICE_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    @model_validator(mode="after")
    def validate_agent_service_consistency(self):
        self._validate_service_agent_mapping()
        self._validate_agent_skills_limit()
        return self

    def _validate_service_agent_mapping(self):
        service_keys = set(self.service_factory.services.keys())
        agent_keys = set(self.agent_factory.agents.keys())

        missing_agents = service_keys - agent_keys
        if missing_agents:
            raise ValueError(
                f"A2A service keys {missing_agents} do not have corresponding agents in agent_factory. Available agents: {sorted(agent_keys)}"
            )

    def _validate_agent_skills_limit(self):
        for service_name, agent_config in self.service_factory.services.items():
            skills = agent_config.card.skills
            if len(skills) > 1:
                raise ValueError(
                    f"Agent '{service_name}' has {len(skills)} skills, but only one skill per agent is currently supported. Skills: {[skill.name if hasattr(skill, 'name') else str(skill) for skill in skills]}"
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
