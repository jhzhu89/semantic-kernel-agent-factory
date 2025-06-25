from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent
from a2a.utils import new_agent_text_message, new_data_artifact, new_task, new_text_artifact
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.contents import (
    ChatHistorySummarizationReducer,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)

from .function_events import FunctionCallEvent, FunctionEvent, FunctionResultEvent

logger = logging.getLogger(__name__)


class SemanticKernelAgentExecutor(AgentExecutor):
    def __init__(
        self,
        agent: ChatCompletionAgent,
        chat_history_threshold: int = 0,
        chat_history_target: int = 0,
        service_id: Optional[str] = None,
        enable_token_streaming: bool = False,
    ) -> None:
        self.agent = agent
        self.name = getattr(agent, "name", "SemanticKernelAgent")
        self._chat_history_threshold = chat_history_threshold
        self._chat_history_target = chat_history_target
        self._service_id = service_id
        self._enable_token_streaming = enable_token_streaming
        self._cancelled_tasks: set = set()
        self._threads_lock = asyncio.Lock()
        self._active_threads: Dict[str, ChatHistoryAgentThread] = {}
        logger.info(f"Initialized executor for agent: {self.name}")

    async def _create_thread(self, session_id: str) -> ChatHistoryAgentThread:
        if (
            self._chat_history_threshold <= 0
            or self._chat_history_target <= 0
            or not self._service_id
            or not getattr(self.agent, "kernel", None)
        ):
            return ChatHistoryAgentThread(thread_id=session_id)

        try:
            chat_service: Any = self.agent.kernel.get_service(self._service_id)
            reducer = ChatHistorySummarizationReducer(
                service=chat_service,
                threshold_count=self._chat_history_threshold,
                target_count=self._chat_history_target,
                auto_reduce=True,
            )
            return ChatHistoryAgentThread(thread_id=session_id, chat_history=reducer)
        except Exception as e:
            logger.warning(f"Failed to create chat history reducer: {e}")
            return ChatHistoryAgentThread(thread_id=session_id)

    async def _get_thread(self, session_id: str) -> ChatHistoryAgentThread:
        if session_id in self._active_threads:
            return self._active_threads[session_id]

        async with self._threads_lock:
            if session_id not in self._active_threads:
                self._active_threads[session_id] = await self._create_thread(session_id)
                logger.debug(f"Created new thread for session: {session_id}")
            return self._active_threads[session_id]

    async def cleanup_session(self, session_id: str) -> None:
        async with self._threads_lock:
            thread = self._active_threads.pop(session_id, None)

        if thread and hasattr(thread, "delete") and callable(thread.delete):
            try:
                await thread.delete()
                logger.debug(f"Cleaned up thread for session: {session_id}")
            except Exception as e:
                logger.warning(f"Error deleting thread for session {session_id}: {e}")

    def _create_function_event(self, item, task, event_type: str):
        event: FunctionEvent
        if event_type == "call":
            event = FunctionCallEvent.create(
                call_id=item.id,
                function_name=item.name,
                arguments=item.arguments,
                metadata={"task_id": task.id, "context_id": task.contextId},
            )
            logger.debug(f"Function call initiated: {item.name} [id: {item.id}]")
            message = f"Calling: {item.name}"
        else:
            event = FunctionResultEvent.create(
                call_id=item.id,
                function_name=item.name,
                result=item.result,
                metadata={"task_id": task.id, "context_id": task.contextId},
            )
            logger.info(f"Function result received: {item.name} [id: {item.id}]")
            message = f"Function result: {item.name}"

        return new_data_artifact(name="function_event", description=message, data=event.to_dict())

    async def _send_status_update(
        self, event_queue: EventQueue, task, state: TaskState, message: str, final: bool = False
    ):
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=state, message=new_agent_text_message(message, task.contextId, task.id)
                ),
                contextId=task.contextId,
                taskId=task.id,
                final=final,
            )
        )

    async def _enqueue_artifact(
        self,
        event_queue: EventQueue,
        task,
        artifact,
        append: bool = False,
        last_chunk: bool = False,
    ):
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                append=append,
                artifact=artifact,
                contextId=task.contextId,
                taskId=task.id,
                lastChunk=last_chunk,
            )
        )

    def _create_intermediate_handler(self, event_queue: EventQueue, task):
        async def handle_intermediate(message: ChatMessageContent) -> None:
            if not message.items:
                return
            for item in message.items:
                if isinstance(item, FunctionCallContent):
                    artifact = self._create_function_event(item, task, "call")
                    await self._enqueue_artifact(event_queue, task, artifact)
                    await self._send_status_update(
                        event_queue, task, TaskState.working, f"Calling: {item.name}"
                    )
                elif isinstance(item, FunctionResultContent):
                    artifact = self._create_function_event(item, task, "result")
                    await self._enqueue_artifact(event_queue, task, artifact)

        return handle_intermediate

    async def _process_token_level_streaming_response(
        self, query, task, event_queue: EventQueue, thread: ChatHistoryAgentThread
    ):
        handle_intermediate = self._create_intermediate_handler(event_queue, task)
        artifact_id = f"streamed_text_{task.id}"
        is_first_chunk = True
        has_sent_any_chunk = False

        async for chunk in self.agent.invoke_stream(
            messages=query, thread=thread, on_intermediate_message=handle_intermediate
        ):
            if task.id in self._cancelled_tasks:
                break

            if (
                chunk
                and hasattr(chunk, "message")
                and chunk.message
                and hasattr(chunk.message, "content")
                and chunk.message.content
            ):
                content = chunk.message.content
                logger.debug(f"Streaming chunk: {content}")

                artifact = new_text_artifact(name="streamed_text", text=content)
                artifact.artifactId = artifact_id
                await self._enqueue_artifact(event_queue, task, artifact, append=not is_first_chunk)

                is_first_chunk = False
                has_sent_any_chunk = True

        if has_sent_any_chunk and task.id not in self._cancelled_tasks:
            artifact = new_text_artifact(name="streamed_text", text="")
            artifact.artifactId = artifact_id
            await self._enqueue_artifact(event_queue, task, artifact, append=True, last_chunk=True)

        self._log_thread_messages(thread)

    async def _process_message_level_streaming_response(
        self, query, task, event_queue: EventQueue, thread: ChatHistoryAgentThread
    ):
        handle_intermediate = self._create_intermediate_handler(event_queue, task)
        has_sent_any_message = False

        async for response_item in self.agent.invoke(
            messages=query, thread=thread, on_intermediate_message=handle_intermediate
        ):
            if task.id in self._cancelled_tasks:
                break

            if (
                response_item
                and hasattr(response_item, "message")
                and response_item.message
                and hasattr(response_item.message, "content")
                and response_item.message.content
            ):
                content = response_item.message.content
                logger.debug(f"Message-level streaming: {content}")

                artifact = new_text_artifact(name="streamed_text", text=content)
                await self._enqueue_artifact(event_queue, task, artifact)

                has_sent_any_message = True

        if has_sent_any_message and task.id not in self._cancelled_tasks:
            artifact = new_text_artifact(name="streamed_text", text="")
            await self._enqueue_artifact(event_queue, task, artifact, last_chunk=True)

        self._log_thread_messages(thread)

    def _log_thread_messages(self, thread):
        def format_content(content):
            if not content:
                return "no content"
            content_str = str(content).strip()
            try:
                if content_str.startswith(("{", "[")):
                    parsed = json.loads(content_str)
                    return f"\n{json.dumps(parsed, indent=2, ensure_ascii=False)}"
            except (json.JSONDecodeError, ValueError):
                pass
            return content_str

        async def log_messages():
            message_count = 0
            logger.info(f"=== Thread Messages Debug () === {thread.id}")
            async for message in thread.get_messages():
                logger.info(f"Message {message_count}: Role={getattr(message, 'role', 'unknown')}")
                content = getattr(message, "content", None)
                logger.info(f"  Content: {format_content(content)}")

                if hasattr(message, "items") and message.items:
                    logger.info(f"  Items: {len(message.items)}")
                    for j, item in enumerate(message.items):
                        item_content = getattr(item, "content", getattr(item, "text", str(item)))
                        logger.info(f"    Item {j}: {type(item).__name__}")
                        logger.info(f"      Content: {format_content(item_content)}")
                logger.info("  " + "=" * 50)
                message_count += 1
            logger.info(f"=== End Thread Messages (Total: {message_count}) ===\n")

        asyncio.create_task(log_messages())

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        if context.current_task is not None:
            task = context.current_task
        else:
            if context.message is None:
                raise ValueError("Cannot create task: both current_task and message are None")
            task = new_task(context.message)

        if not context.current_task:
            await event_queue.enqueue_event(task)

        if task.id in self._cancelled_tasks:
            await self._send_status_update(
                event_queue, task, TaskState.canceled, "Task was cancelled", final=True
            )
            return

        logger.info(f"Agent '{self.name}' executing task {task.id} in session {task.id}")
        await self._send_status_update(event_queue, task, TaskState.working, "Processing...")

        try:
            thread = await self._get_thread(task.id)
            if self._enable_token_streaming:
                await self._process_token_level_streaming_response(query, task, event_queue, thread)
            else:
                await self._process_message_level_streaming_response(
                    query, task, event_queue, thread
                )

            if task.id in self._cancelled_tasks:
                await self._send_status_update(
                    event_queue, task, TaskState.canceled, "Task was cancelled", final=True
                )
            else:
                await self._send_status_update(
                    event_queue, task, TaskState.completed, "Task completed", final=True
                )

            self._cancelled_tasks.discard(task.id)

        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            await self._send_status_update(
                event_queue, task, TaskState.failed, f"Error: {str(e)}", final=True
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is not None:
            task_id = context.current_task.id
        else:
            logger.warning("Cancel requested but no current task in context")
            return

        logger.info(f"Cancel requested for task {task_id}")
        self._cancelled_tasks.add(task_id)

    async def cleanup(self) -> None:
        async with self._threads_lock:
            threads_to_cleanup = list(self._active_threads.values())
            self._active_threads.clear()

        for thread in threads_to_cleanup:
            if hasattr(thread, "delete") and callable(thread.delete):
                try:
                    await thread.delete()
                except Exception as e:
                    logger.warning(f"Error deleting thread: {e}")
