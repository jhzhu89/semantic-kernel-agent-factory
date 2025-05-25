from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, Field


class FunctionEventType(str, Enum):
    CALL = "function_call"
    RESULT = "function_result"


class BaseFunctionEvent(BaseModel):
    call_id: str
    function_name: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class FunctionCallEvent(BaseFunctionEvent):
    event_type: FunctionEventType = FunctionEventType.CALL
    arguments: str | Mapping[str, Any] | None = None

    @classmethod
    def create(
        cls,
        call_id: str,
        function_name: str,
        arguments: str | Mapping[str, Any] | None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FunctionCallEvent":
        return cls(
            call_id=call_id, function_name=function_name, arguments=arguments, metadata=metadata
        )


class FunctionResultEvent(BaseFunctionEvent):
    event_type: FunctionEventType = FunctionEventType.RESULT
    result: Any
    execution_time_ms: Optional[float] = None

    @classmethod
    def create(
        cls,
        call_id: str,
        function_name: str,
        result: Any,
        execution_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FunctionResultEvent":
        return cls(
            call_id=call_id,
            function_name=function_name,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )


FunctionEvent = FunctionCallEvent | FunctionResultEvent


__all__ = [
    "FunctionEventType",
    "BaseFunctionEvent",
    "FunctionCallEvent",
    "FunctionResultEvent",
    "FunctionEvent",
]
