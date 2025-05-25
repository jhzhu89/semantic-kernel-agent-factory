import asyncio
import json
from typing import Dict, List, Union

from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents import (
    ChatHistorySummarizationReducer,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)

from .history_config import AgentFactoryCliConfig
from .json_formatter import StreamingJSONFormatter, is_json_output_expected, serialize_for_json
from .models import MessageType


class ChatService:
    def __init__(self, factory, config: AgentFactoryCliConfig):
        self.factory = factory
        self.config = config
        self.agents = factory.get_all_agents()
        self.threads: Dict[str, ChatHistoryAgentThread] = {}

    def get_agent_names(self):
        return list(self.agents.keys())

    def create_chat_session(self, agent_name: str) -> str:
        if agent_name not in self.threads:
            self._create_thread(agent_name)
        return agent_name

    def get_agent_instructions(self, agent_name: str) -> str:
        instructions = self.config.agent_factory.agents[agent_name].instructions
        if not instructions:
            raise ValueError(f"Agent {agent_name} has no instructions")
        return f"\n{instructions}"

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

    async def send_message(self, agent_name: str, message: str):
        from .models import ErrorMessage, StreamingChunk, StreamingEnded, StreamingStarted

        if not agent_name:
            yield ErrorMessage(agent_name, "No agent specified")
            return

        expect_json = is_json_output_expected(self.config.agent_factory, agent_name)
        json_formatter = StreamingJSONFormatter() if expect_json else None

        event_queue: asyncio.Queue = asyncio.Queue()

        async def handle_intermediate(msg):
            events = self._process_intermediate_message(agent_name, msg)
            for event in events:
                await event_queue.put(event)

        try:
            agent = self.factory.get_agent(agent_name)
            thread = self.threads[agent_name]

            stream_iterator = agent.invoke_stream(
                messages=message,
                thread=thread,
                on_intermediate_message=handle_intermediate,
            )

            assistant_started = False
            async for chunk in stream_iterator:
                while not event_queue.empty():
                    event = await event_queue.get()
                    yield event
                    await asyncio.sleep(0.02)

                if (
                    chunk
                    and hasattr(chunk, "message")
                    and chunk.message
                    and hasattr(chunk.message, "content")
                    and chunk.message.content
                ):
                    if not assistant_started:
                        yield StreamingStarted(agent_name)
                        assistant_started = True

                    content = chunk.message.content
                    if json_formatter:
                        formatted = json_formatter.add_chunk(content)
                        if formatted:
                            yield StreamingChunk(agent_name, formatted)
                    else:
                        if content:
                            yield StreamingChunk(agent_name, content)

            if assistant_started:
                yield StreamingEnded(agent_name)

            if not event_queue.empty():
                raise RuntimeError("event queue is not empty after streaming completion")

        except Exception as e:
            yield ErrorMessage(agent_name, f"Error communicating with agent: {str(e)}")
            if assistant_started:
                yield StreamingEnded(agent_name)

    def _process_intermediate_message(self, agent_name: str, message: ChatMessageContent):
        """Process intermediate message and return list of events to yield."""
        from .models import ErrorMessage, IntermediateMessage

        events: List[Union[IntermediateMessage, ErrorMessage]] = []
        if not message or not message.items:
            return events

        try:
            for item in message.items:
                if isinstance(item, FunctionCallContent):
                    try:
                        arguments = (
                            json.loads(item.arguments)
                            if isinstance(item.arguments, str)
                            else item.arguments
                        )
                    except (json.JSONDecodeError, ValueError):
                        arguments = item.arguments

                    call_data = {
                        "call_id": item.id,
                        "function_name": item.name,
                        "arguments": arguments,
                    }
                    formatted_data = json.dumps(
                        serialize_for_json(call_data), indent=2, ensure_ascii=False
                    )
                    events.append(
                        IntermediateMessage(agent_name, MessageType.FUNCTION_CALL, formatted_data)
                    )

                elif isinstance(item, FunctionResultContent):
                    processed_result = self._process_function_result(item)
                    result_data = {
                        "call_id": item.id,
                        "function_name": item.name,
                        "result": processed_result,
                    }
                    formatted_data = json.dumps(
                        serialize_for_json(result_data), indent=2, ensure_ascii=False
                    )
                    events.append(
                        IntermediateMessage(agent_name, MessageType.FUNCTION_RESULT, formatted_data)
                    )
        except Exception as e:
            events.append(
                ErrorMessage(agent_name, f"Error processing intermediate message: {str(e)}")
            )

        return events

    def _process_function_result(self, item):
        def process_result(result):
            if hasattr(result, "text"):
                try:
                    return json.loads(result.text)
                except (json.JSONDecodeError, ValueError):
                    return result.text
            return str(result)

        if isinstance(item.result, list):
            processed_result = [process_result(r) for r in item.result]
            return processed_result[0] if len(processed_result) == 1 else processed_result
        return process_result(item.result)
