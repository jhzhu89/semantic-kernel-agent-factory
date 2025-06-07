from .models import (
    ErrorMessage,
    IntermediateMessage,
    MessageType,
    StreamingChunk,
    StreamingEnded,
    StreamingStarted,
)


class MessageProcessor:
    def __init__(self, chat_container, add_message_callback, update_status_callback):
        self.chat_container = chat_container
        self.add_message_callback = add_message_callback
        self.update_status_callback = update_status_callback

    async def process_event(self, event):
        if isinstance(event, StreamingStarted):
            self.chat_container.start_streaming(event.agent_name)
        elif isinstance(event, StreamingChunk):
            self.chat_container.add_streaming_chunk(event.agent_name, event.chunk)
        elif isinstance(event, StreamingEnded):
            self.chat_container.end_streaming(event.agent_name)
            self.update_status_callback(event.agent_name)
        elif isinstance(event, IntermediateMessage):
            self.add_message_callback(event.agent_name, event.message_type, event.content)
        elif isinstance(event, ErrorMessage):
            self.add_message_callback(event.agent_name, MessageType.ERROR, event.error)
