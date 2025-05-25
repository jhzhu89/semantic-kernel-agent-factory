from agent_factory.function_events import (
    FunctionCallEvent,
    FunctionResultEvent,
    FunctionEventType
)


class TestFunctionEventType:
    
    def test_function_event_types(self):
        assert FunctionEventType.CALL.value == "function_call"
        assert FunctionEventType.RESULT.value == "function_result"


class TestFunctionEvent:
    
    def test_function_event_union_type(self):
        # Test that FunctionEvent is a union type
        call_event = FunctionCallEvent(
            call_id="call-123",
            function_name="test_function"
        )
        
        result_event = FunctionResultEvent(
            call_id="call-123",
            function_name="test_function",
            result="test result"
        )
        
        # Both should be instances of the union type
        assert isinstance(call_event, (FunctionCallEvent, FunctionResultEvent))
        assert isinstance(result_event, (FunctionCallEvent, FunctionResultEvent))


class TestFunctionCallEvent:
    
    def test_function_call_event_creation(self):
        event = FunctionCallEvent(
            call_id="call-123",
            function_name="test_function",
            arguments={"arg1": "value1", "arg2": "value2"}
        )
        
        assert event.event_type == FunctionEventType.CALL
        assert event.call_id == "call-123"
        assert event.function_name == "test_function"
        assert event.arguments == {"arg1": "value1", "arg2": "value2"}
        assert event.timestamp is not None

    def test_function_call_event_without_arguments(self):
        event = FunctionCallEvent(
            call_id="call-123",
            function_name="test_function"
        )
        
        assert event.arguments is None

    def test_function_call_event_to_dict(self):
        event = FunctionCallEvent(
            call_id="call-123",
            function_name="test_function",
            arguments={"arg1": "value1"}
        )
        
        result = event.to_dict()
        
        assert result["event_type"] == "function_call"
        assert result["call_id"] == "call-123"
        assert result["function_name"] == "test_function"
        assert result["arguments"] == {"arg1": "value1"}
        assert "timestamp" in result

    def test_function_call_event_with_none_arguments(self):
        event = FunctionCallEvent(
            call_id="call-123",
            function_name="test_function",
            arguments=None
        )
        
        assert event.arguments is None

    def test_function_call_event_create_method(self):
        event = FunctionCallEvent.create(
            call_id="call-123",
            function_name="test_function",
            arguments={"arg1": "value1"}
        )
        
        assert event.call_id == "call-123"
        assert event.function_name == "test_function"
        assert event.arguments == {"arg1": "value1"}


class TestFunctionResultEvent:
    
    def test_function_result_event_creation(self):
        event = FunctionResultEvent(
            call_id="call-123",
            function_name="test_function",
            result="Function executed successfully"
        )
        
        assert event.event_type == FunctionEventType.RESULT
        assert event.call_id == "call-123"
        assert event.function_name == "test_function"
        assert event.result == "Function executed successfully"
        assert event.timestamp is not None

    def test_function_result_event_with_none_result(self):
        event = FunctionResultEvent(
            call_id="call-123",
            function_name="test_function",
            result=None
        )
        
        assert event.result is None

    def test_function_result_event_to_dict(self):
        event = FunctionResultEvent(
            call_id="call-123",
            function_name="test_function",
            result="Function result"
        )
        
        result = event.to_dict()
        
        assert result["event_type"] == "function_result"
        assert result["call_id"] == "call-123"
        assert result["function_name"] == "test_function"
        assert result["result"] == "Function result"
        assert "timestamp" in result

    def test_function_result_event_with_complex_result(self):
        complex_result = {"status": "success", "data": [1, 2, 3]}
        
        event = FunctionResultEvent(
            call_id="call-123",
            function_name="test_function",
            result=complex_result
        )
        
        assert event.result == complex_result

    def test_function_result_event_with_execution_time(self):
        event = FunctionResultEvent(
            call_id="call-123",
            function_name="test_function",
            result="test result",
            execution_time_ms=150.5
        )
        
        assert event.execution_time_ms == 150.5

    def test_function_result_event_create_method(self):
        event = FunctionResultEvent.create(
            call_id="call-123",
            function_name="test_function",
            result="test result",
            execution_time_ms=100.0
        )
        
        assert event.call_id == "call-123"
        assert event.function_name == "test_function"
        assert event.result == "test result"
        assert event.execution_time_ms == 100.0
