from __future__ import annotations

from typing import Any


class StreamingJSONFormatter:
    def __init__(self):
        self.indent_level = 0
        self.in_string = False
        self.escaped = False

    def add_chunk(self, chunk: str) -> str:
        result = ""
        for char in chunk:
            if self.escaped:
                result += char
                self.escaped = False
                continue

            if char == "\\" and self.in_string:
                result += char
                self.escaped = True
                continue

            if char == '"':
                result += char
                self.in_string = not self.in_string
                continue

            if self.in_string:
                result += char
                continue

            if char in " \t\n\r":
                continue

            if char in "{[":
                result += char + "\n"
                self.indent_level += 1
                result += "  " * self.indent_level
            elif char in "}]":
                if result.endswith("  "):
                    result = result[:-2]
                if not result.endswith("\n"):
                    result += "\n"
                self.indent_level -= 1
                result += "  " * self.indent_level + char
                if self.indent_level > 0:
                    result += "\n" + "  " * self.indent_level
            elif char == ":":
                result += char + " "
            elif char == ",":
                result += char + "\n" + "  " * self.indent_level
            else:
                result += char

        return result


def is_json_output_expected(config, agent_name: str) -> bool:
    agent_config = config.agents.get(agent_name)
    return (
        agent_config
        and agent_config.model_settings
        and agent_config.model_settings.response_json_schema is not None
    )


def serialize_for_json(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return {k: serialize_for_json(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    return str(obj)
