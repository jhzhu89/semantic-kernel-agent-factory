import pytest
import asyncio
from unittest.mock import Mock, patch
from agent_factory import (
    AgentFactory,
    AgentConfig,
    AgentFactoryConfig,
    AzureOpenAIConfig,
    ModelSelectStrategy
)
from agent_factory.mcp_server.config import MCPConfig, MCPServerConfig

from agent_factory.service import (
    AgentServiceFactory,
    SemanticKernelAgentExecutor
)


class TestIntegration:
    
    @pytest.fixture
    def complete_config(self):
        return AgentFactoryConfig(
            agents={
                "general": AgentConfig(
                    name="general",
                    instructions="You are a helpful assistant",
                    model="gpt-4"
                ),
                "coder": AgentConfig(
                    name="coder", 
                    instructions="You are a coding assistant",
                    model="gpt-4"
                )
            },
            openai_models={
                "gpt-4": AzureOpenAIConfig(
                    model="gpt-4",
                    api_key="test-key",
                    endpoint="https://test.openai.azure.com/"
                )
            },
            model_selection=ModelSelectStrategy.first
        )

    @pytest.mark.asyncio
    async def test_agent_factory_full_lifecycle(self, complete_config):
        with patch('agent_factory.mcp_server.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(complete_config) as factory:
                assert len(factory.get_all_agents()) == 2
                
                general_agent = factory.get_agent("general")
                coder_agent = factory.get_agent("coder")
                
                assert general_agent.name == "general"
                assert coder_agent.name == "coder"
                
                assert factory.get_agent_service_id("general") == "gpt-4"
                assert factory.get_agent_service_id("coder") == "gpt-4"

    @pytest.mark.asyncio
    async def test_executor_with_real_agent(self, complete_config):
        with patch('agent_factory.mcp_server.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(complete_config) as factory:
                agent = factory.get_agent("general")
                service_id = factory.get_agent_service_id("general")
                
                executor = SemanticKernelAgentExecutor(
                    agent=agent,
                    service_id=service_id,
                    chat_history_threshold=100,
                    chat_history_target=50
                )
                
                assert executor.name == "general"
                assert executor._service_id == "gpt-4"
                
                await executor.cleanup()

    @pytest.mark.asyncio
    async def test_service_factory_integration(self, complete_config):
        from agent_factory.service.config import A2AServiceConfig, A2AAgentConfig, ConfigurableAgentCard
        
        a2a_config = A2AServiceConfig(
            services={
                "general": A2AAgentConfig(
                    card=ConfigurableAgentCard(
                        name="general",
                        description="General assistant"
                    )
                )
            }
        )
        
        with patch('agent_factory.mcp_server.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(complete_config) as factory:
                with patch('agent_factory.service.service_factory.InMemoryTaskStore'):
                    with patch('agent_factory.service.service_factory.DefaultRequestHandler'):
                        with patch('agent_factory.service.service_factory.A2AStarletteApplication') as mock_app_factory:
                            mock_app = Mock()
                            mock_app_factory.return_value.build.return_value = mock_app
                            
                            service_factory = AgentServiceFactory(factory, a2a_config)
                            
                            async with service_factory as sf:
                                app = await sf.create_application()
                                
                                assert app is not None
                                executor = sf.get_executor("general")
                                assert executor is not None

    @pytest.mark.asyncio
    async def test_multiple_agents_different_models(self):
        config = AgentFactoryConfig(
            agents={
                "fast": AgentConfig(
                    name="fast",
                    instructions="You are fast",
                    model="gpt-3.5-turbo"
                ),
                "smart": AgentConfig(
                    name="smart",
                    instructions="You are smart", 
                    model="gpt-4"
                )
            },
            openai_models={
                "gpt-3.5-turbo": AzureOpenAIConfig(
                    model="gpt-3.5-turbo",
                    api_key="test-key",
                    endpoint="https://test.openai.azure.com/"
                ),
                "gpt-4": AzureOpenAIConfig(
                    model="gpt-4",
                    api_key="test-key",
                    endpoint="https://test.openai.azure.com/"
                )
            }
        )
        
        with patch('agent_factory.mcp_server.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(config) as factory:
                fast_service_id = factory.get_agent_service_id("fast")
                smart_service_id = factory.get_agent_service_id("smart")
                
                assert fast_service_id == "gpt-3.5-turbo"
                assert smart_service_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_mcp_server_integration(self):
        config = AgentFactoryConfig(
            agents={
                "time_agent": AgentConfig(
                    name="time_agent",
                    instructions="You can tell time",
                    model="gpt-4",
                    mcp_servers=["time"]
                )
            },
            openai_models={
                "gpt-4": AzureOpenAIConfig(
                    model="gpt-4",
                    api_key="test-key",
                    endpoint="https://test.openai.azure.com/"
                )
            },
            mcp=MCPConfig(
                servers={
                    "time": MCPServerConfig(
                        type="stdio",
                        command="python",
                        args=["-m", "mcp_server_time"]
                    )
                }
            )
        )
        
        with patch('agent_factory.mcp_server.MCPProvider') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance._plugins = {}
            mock_provider_instance.get_plugin.return_value = None
            mock_provider.return_value.__aenter__.return_value = mock_provider_instance
            
            async with AgentFactory(config) as factory:
                agent = factory.get_agent("time_agent")
                assert agent is not None

    @pytest.mark.asyncio
    async def test_error_handling_invalid_agent(self, complete_config):
        with patch('agent_factory.mcp_server.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(complete_config) as factory:
                with pytest.raises(KeyError):
                    factory.get_agent("non_existent")

    @pytest.mark.asyncio
    async def test_concurrent_agent_access(self, complete_config):
        with patch('agent_factory.mcp_server.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            async with AgentFactory(complete_config) as factory:
                async def get_agent_task(name):
                    return factory.get_agent(name)
                
                tasks = [
                    get_agent_task("general"),
                    get_agent_task("coder"),
                    get_agent_task("general"),
                    get_agent_task("coder")
                ]
                
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 4
                assert all(agent is not None for agent in results)

    @pytest.mark.asyncio
    async def test_service_factory_error_recovery(self):
        from agent_factory.service.config import A2AServiceConfig, A2AAgentConfig, ConfigurableAgentCard, AgentServiceFactoryConfig
        
        bad_config = AgentServiceFactoryConfig(
            agent_factory=AgentFactoryConfig(
                agents={
                    "bad_agent": AgentConfig(
                        name="bad_agent",
                        instructions="Test",
                        model="gpt-4"
                    )
                },
                openai_models={
                    "gpt-4": AzureOpenAIConfig(
                        model="gpt-4",
                        api_key="test-key",
                        endpoint="https://test.openai.azure.com/"
                    )
                }
            ),
            service_factory=A2AServiceConfig(
                services={
                    "bad_agent": A2AAgentConfig(
                        card=ConfigurableAgentCard(
                            name="bad_agent",
                            description="Bad agent"
                        )
                    )
                }
            )
        )
        
        with patch('agent_factory.mcp_server.MCPProvider') as mock_provider:
            mock_provider.return_value.__aenter__.return_value = Mock()
            
            service_factory = AgentServiceFactory(bad_config)
            
            async with service_factory as sf:
                app = await sf.create_application()
                assert app is not None
