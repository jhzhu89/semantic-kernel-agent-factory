from typing import Any, List


class AgentService:
    def __init__(self, factory: Any, config: Any) -> None:
        self.factory = factory
        self.config = config

    def get_agent_names(self) -> List[str]:
        return list(self.factory.get_all_agents().keys())

    def get_agent_instructions(self, agent_name: str) -> str:
        instructions = self.config.agent_factory.agents[agent_name].instructions
        if not instructions:
            raise ValueError(f"Agent {agent_name} has no instructions")
        return f"\n{instructions}"

    def validate_agent_exists(self, agent_name: str) -> bool:
        return agent_name in self.factory.get_all_agents()
