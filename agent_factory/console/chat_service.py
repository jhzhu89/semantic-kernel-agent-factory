import asyncio
import json
import traceback
from typing import Dict, List, Union

from textual import log
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents import (
    ChatHistorySummarizationReducer,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
    AuthorRole,
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
        from .models import ErrorMessage, StreamingChunk, StreamingEnded, StreamingStarted, HandoffAgentMessage

        # Send debug info via log
        log.debug(f"ChatService.send_message called with agent_name={agent_name}, message={message[:100]}...")

        if not agent_name:
            log.error("No agent specified")
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
            # Check if this agent has handoff orchestration
            has_handoff = self.factory.has_handoff_orchestration(agent_name)
            log.debug(f"Agent {agent_name} has handoff orchestration: {has_handoff}")
            
            if has_handoff:
                log.debug(f"Using handoff orchestration for agent {agent_name}")
                async for event in self._send_message_with_handoff(agent_name, message, event_queue, handle_intermediate):
                    yield event
            else:
                log.debug(f"Using direct messaging for agent {agent_name}")
                async for event in self._send_message_direct(agent_name, message, event_queue, handle_intermediate, json_formatter):
                    yield event

        except Exception as e:
            log.error(f"ERROR in send_message: {str(e)}", exc_info=True)
            yield ErrorMessage(agent_name, f"Error communicating with agent: {str(e)}")

    async def _send_message_with_handoff(self, agent_name: str, message: str, event_queue: asyncio.Queue, handle_intermediate):
        from .models import ErrorMessage, HandoffAgentMessage, HandoffFinalOutput

        log.debug(f"_send_message_with_handoff called for agent {agent_name}")
        yield HandoffAgentMessage("DEBUG", agent_name, f"_send_message_with_handoff called for agent {agent_name}")
        
        orchestration = self.factory.get_handoff_orchestration(agent_name)
        runtime = self.factory.get_runtime()
        
        log.debug(f"Orchestration: {orchestration}")
        log.debug(f"Runtime: {runtime}")
        yield HandoffAgentMessage("DEBUG", agent_name, f"Orchestration: {orchestration}")
        yield HandoffAgentMessage("DEBUG", agent_name, f"Runtime: {runtime}")
        
        if not orchestration or not runtime:
            log.error(f"Missing orchestration or runtime. orchestration={orchestration}, runtime={runtime}")
            yield HandoffAgentMessage("ERROR", agent_name, f"Missing orchestration or runtime. orchestration={orchestration}, runtime={runtime}")
            yield ErrorMessage(agent_name, "Handoff orchestration not available")
            return

        # Ensure thread exists
        if agent_name not in self.threads:
            log.debug(f"Creating thread for agent {agent_name}")
            yield HandoffAgentMessage("DEBUG", agent_name, f"Creating thread for agent {agent_name}")
            self._create_thread(agent_name)
        
        thread = self.threads[agent_name]
        log.debug(f"Thread for {agent_name}: {thread}")
        yield HandoffAgentMessage("DEBUG", agent_name, f"Thread for {agent_name}: {thread}")
        
        user_message = ChatMessageContent(role=AuthorRole.USER, content=message)
        await thread.on_new_message(user_message)
        log.debug(f"Added user message to thread: {message[:100]}...")
        yield HandoffAgentMessage("DEBUG", agent_name, f"Added user message to thread: {message[:100]}...")
        
        async def agent_response_callback(response: ChatMessageContent):
            log.debug(f"Agent response callback called with response from {response.name or 'unknown'}")
            yield HandoffAgentMessage("DEBUG", agent_name, f"Agent response callback called with response from {response.name or 'unknown'}")
            source_agent = response.name or agent_name
            content = response.content or ""
            log.debug(f"Callback: source_agent={source_agent}, content={content[:100]}...")
            yield HandoffAgentMessage("DEBUG", agent_name, f"Callback: source_agent={source_agent}, content={content[:100]}...")
            await event_queue.put(HandoffAgentMessage(source_agent, agent_name, content))

        orchestration.agent_response_callback = agent_response_callback
        log.debug(f"Set agent_response_callback on orchestration")
        yield HandoffAgentMessage("DEBUG", agent_name, f"Set agent_response_callback on orchestration")

        try:
            messages = [message async for message in thread.get_messages()]
            yield HandoffAgentMessage("DEBUG", agent_name, f"Retrieved [{messages}] from thread")
            result = await orchestration.invoke(user_message, runtime)
            log.debug(f"Orchestration invoke returned: {result}")
            yield HandoffAgentMessage("DEBUG", agent_name, f"Orchestration invoke returned: {result}")
            
            # Process any queued events
            while not event_queue.empty():
                event = await event_queue.get()
                log.debug(f"Yielding event: {type(event).__name__}")
                yield HandoffAgentMessage("DEBUG", agent_name, f"Yielding event: {type(event).__name__}")
                yield event
                await asyncio.sleep(0.02)
            
            # Get final result
            log.debug(f"Getting final result from orchestration")
            yield HandoffAgentMessage("DEBUG", agent_name, f"Getting final result from orchestration")
            final_output = await result.get()
            log.debug(f"Final output: {final_output}")
            yield HandoffAgentMessage("DEBUG", agent_name, f"Final output: {final_output}")
            
            if final_output:
                thread.on_new_message(final_output)
                log.debug(f"Added assistant message to thread, yielding HandoffFinalOutput")
                yield HandoffAgentMessage("DEBUG", agent_name, f"Added assistant message to thread")
                #yield HandoffFinalOutput(agent_name, final_output)
            else:
                log.warning(f"No final output from orchestration")
                yield HandoffAgentMessage("WARNING", agent_name, f"No final output from orchestration")

        except Exception as e:
            # Get detailed exception information
            exception_details = f"Exception Type: {type(e).__name__}\nException Message: {str(e)}\nFull Traceback:\n{traceback.format_exc()}"
            log.error(f"ERROR in handoff orchestration: {exception_details}", exc_info=True)
            yield HandoffAgentMessage("ERROR", agent_name, f"ERROR in handoff orchestration: {exception_details}")
            yield ErrorMessage(agent_name, f"Error in handoff orchestration: {str(e)}")

    async def _send_message_direct(self, agent_name: str, message: str, event_queue: asyncio.Queue, handle_intermediate, json_formatter):
        from .models import ErrorMessage, StreamingChunk, StreamingEnded, StreamingStarted

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
            log.error(f"Error processing intermediate message for {agent_name}: {str(e)}", exc_info=True)

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
