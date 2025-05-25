import pytest
from unittest.mock import Mock, AsyncMock, patch
from agent_factory.config import (
    AgentConfig, 
    AgentFactoryConfig, 
    AzureOpenAIConfig,
    ModelSettings,
    ResponseSchema,
    ModelSelectStrategy
)
from agent_factory.factory import AgentFactory
from agent_factory.registry import ServiceRegistry
from semantic_kernel import Kernel


class TestAgentFactory:
    
    @pytest.fixture
    def openai_config(self):
        return {
            "gpt-4": AzureOpenAIConfig(
                model="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                api_version="2024-06-01"
            )
        }

    @pytest.fixture
    def agent_config(self):
        return AgentConfig(
            name="test-agent",
            instructions="You are a test agent",
            model="gpt-4",
            model_settings=ModelSettings(temperature=0.7)
        )

    @pytest.fixture
    def factory_config(self, openai_config, agent_config):
        return AgentFactoryConfig(
            agents={"test-agent": agent_config},
            openai_models=openai_config,
            model_selection=ModelSelectStrategy.first
        )

    @pytest.mark.asyncio
    async def test_factory_initialization(self, factory_config):
        factory = AgentFactory(factory_config)
        assert factory._config == factory_config
        assert factory._agents == {}
        assert factory._kernel is None

    @pytest.mark.asyncio
    async def test_factory_context_manager(self, factory_config):
        with patch('agent_factory.factory.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(factory_config) as factory:
                assert factory._kernel is not None
                assert isinstance(factory._kernel, Kernel)

    @pytest.mark.asyncio
    async def test_get_agent(self, factory_config):
        with patch('agent_factory.factory.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(factory_config) as factory:
                agent = factory.get_agent("test-agent")
                assert agent is not None
                assert agent.name == "test-agent"

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, factory_config):
        factory = AgentFactory(factory_config)
        
        with pytest.raises(KeyError):
            factory.get_agent("non-existent")

    @pytest.mark.asyncio
    async def test_get_agent_service_id(self, factory_config):
        with patch('agent_factory.factory.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(factory_config) as factory:
                service_id = factory.get_agent_service_id("test-agent")
                assert service_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_create_response_model_with_schema(self, openai_config):
        schema_config = AgentConfig(
            name="schema-agent",
            instructions="Test",
            model="gpt-4",
            model_settings=ModelSettings(
                response_json_schema=ResponseSchema(
                    name="TestResponse",
                    json_schema_definition={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        },
                        "required": ["message"]
                    }
                )
            )
        )
        
        factory_config = AgentFactoryConfig(
            agents={"schema-agent": schema_config},
            openai_models=openai_config,
            model_selection=ModelSelectStrategy.first
        )
        
        with patch('agent_factory.factory.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(factory_config) as factory:
                response_model = factory.get_agent_response_model("schema-agent")
                assert response_model is not None

    @pytest.mark.asyncio
    async def test_create_agent_without_name_raises_error(self, openai_config):
        invalid_config = AgentConfig(
            name=None,
            instructions="Test",
            model="gpt-4"
        )
        
        # Create a factory without going through AgentFactoryConfig validation
        # which would automatically set the name
        factory_config = AgentFactoryConfig(
            agents={},  # Empty agents dict to avoid validation
            openai_models=openai_config,
            model_selection=ModelSelectStrategy.first
        )
        
        factory = AgentFactory(factory_config)
        
        # Now the invalid_config still has name=None
        with pytest.raises(ValueError, match="Agent config must have a name"):
            await factory._create_agent(invalid_config)


class TestServiceRegistry:
    
    @pytest.fixture
    def openai_configs(self):
        return {
            "gpt-4": AzureOpenAIConfig(
                model="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/"
            ),
            "gpt-3.5-turbo": AzureOpenAIConfig(
                model="gpt-3.5-turbo",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/"
            )
        }

    def test_registry_initialization(self, openai_configs):
        registry = ServiceRegistry(openai_configs)
        assert registry._configs == openai_configs
        assert registry._services == {}

    def test_build_kernel(self, openai_configs):
        registry = ServiceRegistry(openai_configs)
        kernel = registry.build_kernel()
        
        assert isinstance(kernel, Kernel)
        assert len(registry._services) == 2

    def test_select_first_strategy(self, openai_configs):
        registry = ServiceRegistry(openai_configs)
        registry.build_kernel()
        
        selected = registry.select(ModelSelectStrategy.first)
        assert selected in ["gpt-4", "gpt-3.5-turbo"]

    def test_select_cost_strategy(self, openai_configs):
        registry = ServiceRegistry(openai_configs)
        registry.build_kernel()
        
        selected = registry.select(ModelSelectStrategy.cost)
        assert selected == "gpt-3.5-turbo"

    def test_select_latency_strategy_no_small_model(self, openai_configs):
        registry = ServiceRegistry(openai_configs)
        registry.build_kernel()
        
        selected = registry.select(ModelSelectStrategy.latency)
        assert selected in ["gpt-4", "gpt-3.5-turbo"]

    def test_select_single_service(self):
        single_config = {
            "gpt-4": AzureOpenAIConfig(
                model="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/"
            )
        }
        
        registry = ServiceRegistry(single_config)
        registry.build_kernel()
        
        selected = registry.select(ModelSelectStrategy.cost)
        assert selected == "gpt-4"
