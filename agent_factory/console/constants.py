from .models import MessageType

MESSAGE_TYPE_ICONS = {
    MessageType.USER: ("👤", "You"),
    MessageType.ASSISTANT: ("🤖", "Assistant"),
    MessageType.SYSTEM: ("💡", "System"),
    MessageType.FUNCTION_CALL: ("⚡", "Function Call"),
    MessageType.FUNCTION_RESULT: ("✅", "Function Result"),
    MessageType.ERROR: ("❌", "Error"),
    MessageType.AGENT_INSTRUCTIONS: ("📋", "Agent Instructions"),
}

MESSAGE_COUNT_TYPES = {MessageType.USER, MessageType.ASSISTANT}
