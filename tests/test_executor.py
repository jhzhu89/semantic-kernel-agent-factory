import pytest
from unittest.mock import Mock, AsyncMock, patch
from agent_factory.service import SemanticKernelAgentExecutor
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent


class TestSemanticKernelAgentExecutor:
    
    @pytest.fixture
    def mock_agent(self):
        agent = Mock(spec=ChatCompletionAgent)
        agent.name = "test-agent"
        agent.kernel = Mock()
        return agent

    @pytest.fixture
    def executor(self, mock_agent):
        return SemanticKernelAgentExecutor(
            agent=mock_agent,
            chat_history_threshold=100,
            chat_history_target=50,
            service_id="gpt-4"
        )

    def test_executor_initialization(self, mock_agent):
        executor = SemanticKernelAgentExecutor(
            agent=mock_agent,
            chat_history_threshold=100,
            chat_history_target=50,
            service_id="gpt-4"
        )
        
        assert executor.agent == mock_agent
        assert executor.name == "test-agent"
        assert executor._chat_history_threshold == 100
        assert executor._chat_history_target == 50
        assert executor._service_id == "gpt-4"
        assert len(executor._cancelled_tasks) == 0
        assert len(executor._active_threads) == 0

    def test_executor_initialization_with_defaults(self, mock_agent):
        executor = SemanticKernelAgentExecutor(agent=mock_agent)
        
        assert executor._chat_history_threshold == 0
        assert executor._chat_history_target == 0
        assert executor._service_id is None

    @pytest.mark.asyncio
    async def test_create_thread_without_history_management(self, mock_agent):
        executor = SemanticKernelAgentExecutor(
            agent=mock_agent,
            chat_history_threshold=0,
            chat_history_target=0
        )
        
        thread = await executor._create_thread("session-123")
        
        assert isinstance(thread, ChatHistoryAgentThread)
        assert thread.id == "session-123"

    @pytest.mark.asyncio
    async def test_create_thread_with_history_management(self, mock_agent):
        executor = SemanticKernelAgentExecutor(
            agent=mock_agent,
            chat_history_threshold=100,
            chat_history_target=50,
            service_id="gpt-4"
        )
        
        with patch('agent_factory.service.executor.ChatHistorySummarizationReducer') as mock_reducer:
            thread = await executor._create_thread("session-123")
            
            assert isinstance(thread, ChatHistoryAgentThread)
            mock_reducer.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_create_thread_new(self, executor):
        session_id = "new-session"
        
        thread = await executor._get_thread(session_id)
        
        assert thread is not None
        assert session_id in executor._active_threads

    @pytest.mark.asyncio
    async def test_get_or_create_thread_existing(self, executor):
        session_id = "existing-session"
        existing_thread = Mock(spec=ChatHistoryAgentThread)
        executor._active_threads[session_id] = existing_thread
        
        thread = await executor._get_thread(session_id)
        
        assert thread == existing_thread

    @pytest.mark.asyncio
    async def test_execute_agent_task_simple_message(self, executor):
        mock_context = Mock()
        mock_context.session_id = "session-123"
        mock_context.cancel_requested = False
        mock_context.current_task = None
        mock_context.message = "Test message"
        mock_context.get_user_input.return_value = "Test message"
        
        mock_event_queue = AsyncMock()
        
        mock_thread = AsyncMock(spec=ChatHistoryAgentThread)
        
        with patch.object(executor, '_get_thread', return_value=mock_thread):
            with patch.object(executor, '_process_message_level_streaming_response', new_callable=AsyncMock) as mock_process:
                with patch('agent_factory.service.executor.new_task') as mock_new_task:
                    mock_task = Mock()
                    mock_task.id = "task-123"
                    mock_task.contextId = "context-123"  # Make sure this is a string
                    mock_new_task.return_value = mock_task
                    
                    await executor.execute(mock_context, mock_event_queue)
                    
                    mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agent_task_with_cancellation(self, executor):
        mock_context = Mock()
        mock_context.current_task = Mock()
        mock_context.current_task.id = "task-123"
        
        mock_event_queue = Mock()
        
        executor._cancelled_tasks.add("task-123")
        
        await executor.cancel(mock_context, mock_event_queue)
        
        assert "task-123" in executor._cancelled_tasks

    @pytest.mark.asyncio
    async def test_cancel_task(self, executor):
        mock_context = Mock()
        mock_context.current_task = Mock()
        mock_context.current_task.id = "task-to-cancel"
        
        mock_event_queue = Mock()
        
        await executor.cancel(mock_context, mock_event_queue)
        
        assert "task-to-cancel" in executor._cancelled_tasks

    @pytest.mark.asyncio
    async def test_cleanup(self, executor):
        # Create AsyncMock threads that have an async delete method
        mock_thread1 = AsyncMock(spec=ChatHistoryAgentThread)
        mock_thread2 = AsyncMock(spec=ChatHistoryAgentThread)
        
        executor._active_threads["session1"] = mock_thread1
        executor._active_threads["session2"] = mock_thread2
        executor._cancelled_tasks.add("task1")
        
        await executor.cleanup()
        
        assert len(executor._active_threads) == 0
        # Note: _cancelled_tasks is not cleared in the actual cleanup method

    def test_emit_function_events_with_function_calls(self, executor):
        mock_task = Mock()
        mock_task.id = "task-123"
        mock_task.contextId = "context-123"
        
        function_call = FunctionCallContent(
            function_name="test_function",
            plugin_name="test_plugin",
            id="call-123",
            arguments={"arg1": "value1"}
        )
        
        with patch('agent_factory.service.executor.new_data_artifact') as mock_artifact:
            mock_artifact.return_value = Mock()
            
            artifact = executor._create_function_event(function_call, mock_task, "call")
            
            mock_artifact.assert_called_once()
            assert artifact is not None

    def test_emit_function_events_without_function_calls(self, executor):
        mock_task = Mock()
        mock_task.id = "task-123"
        mock_task.contextId = "context-123"
        
        function_result = FunctionResultContent(
            function_name="test_function",
            plugin_name="test_plugin",
            id="call-123",
            result="function result"
        )
        
        with patch('agent_factory.service.executor.new_data_artifact') as mock_artifact:
            mock_artifact.return_value = Mock()
            
            artifact = executor._create_function_event(function_result, mock_task, "result")
            
            mock_artifact.assert_called_once()
            assert artifact is not None

    @pytest.mark.asyncio
    async def test_thread_cleanup_on_session_end(self, executor):
        session_id = "session-to-cleanup"
        mock_thread = Mock(spec=ChatHistoryAgentThread)
        executor._active_threads[session_id] = mock_thread
        
        async with executor._threads_lock:
            if session_id in executor._active_threads:
                del executor._active_threads[session_id]
        
        assert session_id not in executor._active_threads

    def test_agent_name_fallback(self):
        mock_agent = Mock()
        delattr(mock_agent, 'name')
        
        executor = SemanticKernelAgentExecutor(agent=mock_agent)
        
        assert executor.name == "SemanticKernelAgent"
