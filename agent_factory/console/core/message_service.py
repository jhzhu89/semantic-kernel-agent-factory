import asyncio
import json
import logging
from typing import AsyncGenerator, List, Union

logger = logging.getLogger(__name__)

from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents import (
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)

from ..domain.models import MessageType
from ..domain.processors.json_formatter import (
    StreamingJSONFormatter,
    is_json_output_expected,
    serialize_for_json,
)


class MessageService:
    def __init__(self, factory, config):
        self.factory = factory
        self.config = config

    async def send_message(
        self, agent_name: str, message: str, thread: ChatHistoryAgentThread
    ) -> AsyncGenerator:
        from ..domain.models import ErrorMessage, StreamingChunk, StreamingEnded, StreamingStarted

        if not agent_name:
            yield ErrorMessage(agent_name, "No agent specified")
            return

        expect_json = is_json_output_expected(self.config.agent_factory, agent_name)
        json_formatter = StreamingJSONFormatter() if expect_json else None

        event_queue: asyncio.Queue = asyncio.Queue()

        class EndMarker:
            pass

        async def handle_intermediate(msg):
            logger.debug("Processing intermediate message")
            events = self._process_intermediate_message(agent_name, msg)
            logger.debug(f"Generated {len(events)} intermediate events")
            for event in events:
                await event_queue.put(event)

        async def process_stream():
            assistant_started = False
            try:
                agent = self.factory.get_agent(agent_name)
                stream_iterator = agent.invoke_stream(
                    messages=message,
                    thread=thread,
                    on_intermediate_message=handle_intermediate,
                )

                async for chunk in stream_iterator:
                    if (
                        chunk
                        and hasattr(chunk, "message")
                        and chunk.message
                        and hasattr(chunk.message, "content")
                        and chunk.message.content
                    ):
                        if not assistant_started:
                            logger.debug("Starting streaming response")
                            await event_queue.put(StreamingStarted(agent_name))
                            assistant_started = True

                        content = chunk.message.content
                        if json_formatter:
                            formatted = json_formatter.add_chunk(content)
                            if formatted:
                                await event_queue.put(StreamingChunk(agent_name, formatted))
                        else:
                            if content:
                                await event_queue.put(StreamingChunk(agent_name, content))

                if assistant_started:
                    logger.debug("Ending streaming response")
                    await event_queue.put(StreamingEnded(agent_name))

            except Exception as e:
                await event_queue.put(
                    ErrorMessage(agent_name, f"Error communicating with agent: {str(e)}")
                )
                if assistant_started:
                    await event_queue.put(StreamingEnded(agent_name))
            finally:
                await event_queue.put(EndMarker())

        stream_task = asyncio.create_task(process_stream())

        try:
            while True:
                event = await event_queue.get()
                if isinstance(event, EndMarker):
                    break
                yield event
        finally:
            await stream_task

    def _process_intermediate_message(self, agent_name: str, message: ChatMessageContent):
        from ..domain.models import ErrorMessage, IntermediateMessage

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
