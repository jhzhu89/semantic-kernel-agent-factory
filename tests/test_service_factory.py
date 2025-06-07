import pytest
from unittest.mock import Mock, AsyncMock, patch
from agent_factory.config import (
    AgentConfig,
    A2AAgentConfig,
    A2AServiceConfig,
    ConfigurableAgentCard,
    AgentServiceFactoryConfig
)
from agent_factory.service_factory import AgentServiceFactory
from agent_factory.factory import AgentFactory


class TestAgentServiceFactory:
    
    @pytest.fixture
    def mock_agent_factory(self):
        factory = Mock(spec=AgentFactory)
        factory.get_agent.return_value = Mock()
        factory.get_agent_service_id.return_value = "gpt-4"
        return factory

    @pytest.fixture
    def a2a_agent_config(self):
        return A2AAgentConfig(
            card=ConfigurableAgentCard(
                name="test-agent",
                description="Test agent"
            ),
            chat_history_threshold=100,
            chat_history_target=50
        )

    @pytest.fixture
    def a2a_service_config(self, a2a_agent_config):
        return A2AServiceConfig(
            services={"test-agent": a2a_agent_config}
        )

    @pytest.fixture
    def service_factory_config(self):
        return Mock(spec=AgentServiceFactoryConfig)

    def test_service_factory_init_with_factory_and_config(self, mock_agent_factory, a2a_service_config):
        service_factory = AgentServiceFactory(mock_agent_factory, a2a_service_config)
        
        assert service_factory._agent_factory == mock_agent_factory
        assert service_factory._a2a_config == a2a_service_config
        assert not service_factory._owns_agent_factory

    def test_service_factory_init_with_factory_config(self, service_factory_config):
        service_factory_config.service_factory = Mock()
        service_factory_config.agent_factory = Mock()
        
        service_factory = AgentServiceFactory(service_factory_config)
        
        assert service_factory._agent_factory_config == service_factory_config.agent_factory
        assert service_factory._owns_agent_factory

    def test_service_factory_init_without_a2a_config_raises_error(self, mock_agent_factory):
        with pytest.raises(ValueError, match="a2a_config required"):
            AgentServiceFactory(mock_agent_factory, None)

    @pytest.mark.asyncio
    async def test_service_factory_context_manager(self, mock_agent_factory, a2a_service_config):
        service_factory = AgentServiceFactory(mock_agent_factory, a2a_service_config)
        
        async with service_factory as sf:
            assert sf == service_factory
            assert sf._agent_factory == mock_agent_factory

    @pytest.mark.asyncio
    async def test_service_factory_owns_factory_context_manager(self):
        # Create a proper config object that will trigger the _owns_agent_factory path
        agent_config = AgentConfig(
            name="test-agent",
            instructions="Test instructions"
        )
        
        from agent_factory.config import AzureOpenAIConfig, AgentFactoryConfig
        openai_config = {
            "gpt-4": AzureOpenAIConfig(
                model="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/"
            )
        }
        
        agent_factory_config = AgentFactoryConfig(
            agents={"test-agent": agent_config},
            openai_models=openai_config
        )
        
        a2a_agent_config = A2AAgentConfig(
            card=ConfigurableAgentCard(
                name="test-agent",
                description="Test description"
            )
        )
        
        service_factory_config_obj = A2AServiceConfig(
            services={"test-agent": a2a_agent_config}
        )
        
        service_factory_config = AgentServiceFactoryConfig(
            agent_factory=agent_factory_config,
            service_factory=service_factory_config_obj
        )
        
        # Create the service factory with a config (this should set _owns_agent_factory=True)
        service_factory = AgentServiceFactory(service_factory_config)
        
        # Verify the service factory was initialized correctly
        assert service_factory._agent_factory_config == service_factory_config.agent_factory
        assert service_factory._owns_agent_factory == True
        assert service_factory._agent_factory is None  # Should be None before entering context
        
        # Mock MCPProvider to avoid connection issues
        with patch('agent_factory.factory.MCPProvider') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance._plugins = {}
            mock_provider.return_value.__aenter__.return_value = mock_provider_instance
            
            # Test that entering the context creates the agent factory
            async with service_factory as sf:
                # The _agent_factory should now be set to a real AgentFactory instance
                assert sf._agent_factory is not None
                assert isinstance(sf._agent_factory, AgentFactory)
                assert sf._owns_agent_factory == True

    @pytest.mark.asyncio
    async def test_create_application(self, mock_agent_factory, a2a_service_config):
        with patch('agent_factory.service_factory.Mount') as mock_mount:
            with patch('agent_factory.service_factory.Starlette') as mock_starlette:
                mock_app = Mock()
                mock_starlette.return_value = mock_app
                
                service_factory = AgentServiceFactory(mock_agent_factory, a2a_service_config)
                service_factory._agent_factory = mock_agent_factory
                
                with patch.object(service_factory, '_create_a2a_app', return_value=Mock()) as mock_create:
                    app = await service_factory.create_application()
                    
                    assert app == mock_app
                    mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_a2a_app_success(self, mock_agent_factory, a2a_agent_config):
        mock_agent = Mock()
        mock_agent_factory.get_agent.return_value = mock_agent
        mock_agent_factory.get_agent_service_id.return_value = "gpt-4"
        
        with patch('agent_factory.service_factory.SemanticKernelAgentExecutor') as mock_executor:
            with patch('agent_factory.service_factory.InMemoryTaskStore') as mock_task_store:
                with patch('agent_factory.service_factory.DefaultRequestHandler') as mock_handler:
                    with patch('agent_factory.service_factory.A2AStarletteApplication') as mock_app_factory:
                        mock_a2a_app = Mock()
                        mock_app_factory.return_value.build.return_value = mock_a2a_app
                        
                        service_factory = AgentServiceFactory(Mock(), Mock())
                        service_factory._agent_factory = mock_agent_factory
                        
                        result = await service_factory._create_a2a_app("test-agent", a2a_agent_config)
                        
                        assert result == mock_a2a_app
                        mock_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_a2a_app_with_token_streaming_enabled(self, mock_agent_factory, a2a_agent_config):
        mock_agent = Mock()
        mock_agent_factory.get_agent.return_value = mock_agent
        mock_agent_factory.get_agent_service_id.return_value = "gpt-4"
        
        # Set token streaming to enabled in A2A config
        a2a_agent_config.enable_token_streaming = True
        
        with patch('agent_factory.service_factory.SemanticKernelAgentExecutor') as mock_executor:
            with patch('agent_factory.service_factory.InMemoryTaskStore'):
                with patch('agent_factory.service_factory.DefaultRequestHandler'):
                    with patch('agent_factory.service_factory.A2AStarletteApplication') as mock_app_factory:
                        mock_app_factory.return_value.build.return_value = Mock()
                        
                        service_factory = AgentServiceFactory(Mock(), Mock())
                        service_factory._agent_factory = mock_agent_factory
                        
                        await service_factory._create_a2a_app("test-agent", a2a_agent_config)
                        
                        # Verify that enable_token_streaming=True was passed to the executor
                        mock_executor.assert_called_once_with(
                            mock_agent,
                            chat_history_threshold=a2a_agent_config.chat_history_threshold,
                            chat_history_target=a2a_agent_config.chat_history_target,
                            service_id="gpt-4",
                            enable_token_streaming=True,
                        )

    @pytest.mark.asyncio
    async def test_create_a2a_app_with_token_streaming_disabled(self, mock_agent_factory, a2a_agent_config):
        mock_agent = Mock()
        mock_agent_factory.get_agent.return_value = mock_agent
        mock_agent_factory.get_agent_service_id.return_value = "gpt-4"
        
        # Set token streaming to disabled in A2A config
        a2a_agent_config.enable_token_streaming = False
        
        with patch('agent_factory.service_factory.SemanticKernelAgentExecutor') as mock_executor:
            with patch('agent_factory.service_factory.InMemoryTaskStore'):
                with patch('agent_factory.service_factory.DefaultRequestHandler'):
                    with patch('agent_factory.service_factory.A2AStarletteApplication') as mock_app_factory:
                        mock_app_factory.return_value.build.return_value = Mock()
                        
                        service_factory = AgentServiceFactory(Mock(), Mock())
                        service_factory._agent_factory = mock_agent_factory
                        
                        await service_factory._create_a2a_app("test-agent", a2a_agent_config)
                        
                        # Verify that enable_token_streaming=False was passed to the executor
                        mock_executor.assert_called_once_with(
                            mock_agent,
                            chat_history_threshold=a2a_agent_config.chat_history_threshold,
                            chat_history_target=a2a_agent_config.chat_history_target,
                            service_id="gpt-4",
                            enable_token_streaming=False,
                        )

    @pytest.mark.asyncio
    async def test_create_a2a_app_defaults_to_streaming_disabled(self, mock_agent_factory, a2a_agent_config):
        mock_agent = Mock()
        mock_agent_factory.get_agent.return_value = mock_agent
        mock_agent_factory.get_agent_service_id.return_value = "gpt-4"
        
        # Default value should be False for enable_token_streaming
        assert a2a_agent_config.enable_token_streaming == False
        
        with patch('agent_factory.service_factory.SemanticKernelAgentExecutor') as mock_executor:
            with patch('agent_factory.service_factory.InMemoryTaskStore'):
                with patch('agent_factory.service_factory.DefaultRequestHandler'):
                    with patch('agent_factory.service_factory.A2AStarletteApplication') as mock_app_factory:
                        mock_app_factory.return_value.build.return_value = Mock()
                        
                        service_factory = AgentServiceFactory(Mock(), Mock())
                        service_factory._agent_factory = mock_agent_factory
                        
                        await service_factory._create_a2a_app("test-agent", a2a_agent_config)
                        
                        # Verify that enable_token_streaming defaults to False
                        mock_executor.assert_called_once_with(
                            mock_agent,
                            chat_history_threshold=a2a_agent_config.chat_history_threshold,
                            chat_history_target=a2a_agent_config.chat_history_target,
                            service_id="gpt-4",
                            enable_token_streaming=False,
                        )

    @pytest.mark.asyncio
    async def test_create_a2a_app_agent_not_found(self, mock_agent_factory, a2a_agent_config):
        mock_agent_factory.get_agent.side_effect = KeyError("Agent not found")
        
        service_factory = AgentServiceFactory(Mock(), Mock())
        service_factory._agent_factory = mock_agent_factory
        
        result = await service_factory._create_a2a_app("non-existent", a2a_agent_config)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_create_a2a_app_no_factory(self, a2a_agent_config):
        service_factory = AgentServiceFactory(Mock(), Mock())
        service_factory._agent_factory = None
        
        result = await service_factory._create_a2a_app("test-agent", a2a_agent_config)
        
        assert result is None

    def test_get_executor(self, mock_agent_factory, a2a_service_config):
        service_factory = AgentServiceFactory(mock_agent_factory, a2a_service_config)
        mock_executor = Mock()
        service_factory._executors["test-agent"] = mock_executor
        
        result = service_factory.get_executor("test-agent")
        
        assert result == mock_executor

    def test_get_executor_not_found(self, mock_agent_factory, a2a_service_config):
        service_factory = AgentServiceFactory(mock_agent_factory, a2a_service_config)
        
        result = service_factory.get_executor("non-existent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_executors(self, mock_agent_factory, a2a_service_config):
        service_factory = AgentServiceFactory(mock_agent_factory, a2a_service_config)
        
        mock_executor1 = AsyncMock()
        mock_executor2 = AsyncMock()
        service_factory._executors["agent1"] = mock_executor1
        service_factory._executors["agent2"] = mock_executor2
        
        async with service_factory:
            pass
        
        mock_executor1.cleanup.assert_called_once()
        mock_executor2.cleanup.assert_called_once()
        assert len(service_factory._executors) == 0
