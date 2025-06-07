from typing import Dict, Optional

from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents import ChatHistorySummarizationReducer

from .agent_service import AgentService
from .message_service import MessageService


class SessionManager:
    def __init__(self, factory, config):
        self.factory = factory
        self.config = config
        self.threads: Dict[str, ChatHistoryAgentThread] = {}
        self.agent_service = AgentService(factory, config)
        self.message_service = MessageService(factory, config)

    def create_chat_session(self, agent_name: str) -> str:
        if not self.agent_service.validate_agent_exists(agent_name):
            raise ValueError(f"Agent {agent_name} does not exist")

        if agent_name not in self.threads:
            self._create_thread(agent_name)
        return agent_name

    def get_thread(self, agent_name: str) -> Optional[ChatHistoryAgentThread]:
        return self.threads.get(agent_name)

    async def send_message(self, agent_name: str, message: str):
        if agent_name not in self.threads:
            self.create_chat_session(agent_name)

        thread = self.threads[agent_name]
        async for event in self.message_service.send_message(agent_name, message, thread):
            yield event

    def get_agent_names(self) -> list:
        return self.agent_service.get_agent_names()

    def get_agent_instructions(self, agent_name: str) -> str:
        return self.agent_service.get_agent_instructions(agent_name)

    def _create_thread(self, agent_name: str) -> ChatHistoryAgentThread:
        if agent_name in self.threads:
            return self.threads[agent_name]

        agent = self.factory.get_agent(agent_name)
        model = self.config.agent_factory.agents[agent_name].model or agent_name
        agent_history_config = self.config.get_agent_history_config(agent_name)

        try:
            if agent_history_config and agent_history_config.threshold_count > 0:
                reducer = ChatHistorySummarizationReducer(
                    service=agent.kernel.get_service(model),
                    threshold_count=agent_history_config.threshold_count,
                    target_count=agent_history_config.target_count,
                    auto_reduce=True,
                )
                thread = ChatHistoryAgentThread(chat_history=reducer)
            else:
                thread = ChatHistoryAgentThread()
        except Exception:
            thread = ChatHistoryAgentThread()

        self.threads[agent_name] = thread
        return thread
