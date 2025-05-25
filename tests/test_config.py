import pytest
from pydantic import ValidationError
from agent_factory.config import (
    AgentConfig,
    AgentFactoryConfig,
    AzureOpenAIConfig,
    ModelSettings,
    ResponseSchema,
    ModelSelectStrategy,
    A2AAgentConfig,
    A2AServiceConfig,
    ConfigurableAgentCard,
    AgentServiceFactoryConfig
)


class TestAzureOpenAIConfig:
    
    def test_azure_openai_config_valid(self):
        config = AzureOpenAIConfig(
            model="gpt-4",
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
            api_version="2024-06-01"
        )
        
        assert config.model == "gpt-4"
        assert config.api_key.get_secret_value() == "test-key"
        assert str(config.endpoint) == "https://test.openai.azure.com/"
        assert config.api_version == "2024-06-01"

    def test_azure_openai_config_defaults(self):
        config = AzureOpenAIConfig()
        
        assert config.model is None
        assert config.api_key is None
        assert config.api_version == "2024-06-01"
        assert config.endpoint is None


class TestResponseSchema:
    
    def test_response_schema_creation(self):
        schema_def = {
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        }
        
        schema = ResponseSchema(
            name="TestResponse",
            json_schema_definition=schema_def
        )
        
        assert schema.name == "TestResponse"
        assert schema.json_schema_definition == schema_def

    def test_response_schema_to_dict(self):
        schema_def = {
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            }
        }
        
        schema = ResponseSchema(
            name="TestResponse",
            json_schema_definition=schema_def
        )
        
        result = schema.to_dict()
        
        expected = {
            "type": "json_schema",
            "json_schema": {
                "schema": schema_def,
                "name": "TestResponse",
                "strict": True
            }
        }
        
        assert result == expected


class TestModelSettings:
    
    def test_model_settings_defaults(self):
        settings = ModelSettings()
        
        assert settings.temperature == 1.0
        assert settings.top_p is None
        assert settings.frequency_penalty is None
        assert settings.presence_penalty is None
        assert settings.max_tokens is None
        assert settings.response_json_schema is None

    def test_model_settings_with_values(self):
        schema = ResponseSchema(
            name="TestResponse",
            json_schema_definition={"type": "object"}
        )
        
        settings = ModelSettings(
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            max_tokens=1000,
            response_json_schema=schema
        )
        
        assert settings.temperature == 0.7
        assert settings.top_p == 0.9
        assert settings.frequency_penalty == 0.1
        assert settings.presence_penalty == 0.2
        assert settings.max_tokens == 1000
        assert settings.response_json_schema == schema


class TestAgentConfig:
    
    def test_agent_config_minimal(self):
        config = AgentConfig(
            name="test-agent",
            instructions="Test instructions"
        )
        
        assert config.name == "test-agent"
        assert config.instructions == "Test instructions"
        assert config.model is None
        assert config.model_settings is None
        assert config.mcp_servers == []

    def test_agent_config_full(self):
        model_settings = ModelSettings(temperature=0.5)
        
        config = AgentConfig(
            name="test-agent",
            instructions="Test instructions",
            model="gpt-4",
            model_settings=model_settings,
            mcp_servers=["time", "weather"]
        )
        
        assert config.name == "test-agent"
        assert config.instructions == "Test instructions"
        assert config.model == "gpt-4"
        assert config.model_settings == model_settings
        assert config.mcp_servers == ["time", "weather"]

    def test_agent_config_name_validation(self):
        with pytest.raises(ValidationError):
            AgentConfig(
                name="invalid name!",
                instructions="Test"
            )

    def test_agent_config_valid_names(self):
        valid_names = ["test-agent", "test_agent", "TestAgent123", "agent-1"]
        
        for name in valid_names:
            config = AgentConfig(name=name, instructions="Test")
            assert config.name == name


class TestConfigurableAgentCard:
    
    def test_configurable_agent_card_minimal(self):
        card = ConfigurableAgentCard(
            name="test-agent",
            description="Test description"
        )
        
        assert card.name == "test-agent"
        assert card.description == "Test description"

    def test_configurable_agent_card_to_agent_card(self):
        card = ConfigurableAgentCard(
            name="test-agent",
            description="Test description",
            instructions="Test instructions"
        )
        
        agent_card = card.to_agent_card()
        
        assert agent_card.name == "test-agent"
        assert agent_card.description == "Test description"


class TestA2AAgentConfig:
    
    def test_a2a_agent_config_defaults(self):
        card = ConfigurableAgentCard(
            name="test-agent",
            description="Test description"
        )
        
        config = A2AAgentConfig(card=card)
        
        assert config.card == card
        assert config.chat_history_threshold == 1000
        assert config.chat_history_target == 10

    def test_a2a_agent_config_with_values(self):
        card = ConfigurableAgentCard(
            name="test-agent",
            description="Test description"
        )
        
        config = A2AAgentConfig(
            card=card,
            chat_history_threshold=100,
            chat_history_target=50
        )
        
        assert config.card == card
        assert config.chat_history_threshold == 100
        assert config.chat_history_target == 50


class TestA2AServiceConfig:
    
    def test_a2a_service_config(self):
        agent_config = A2AAgentConfig(
            card=ConfigurableAgentCard(
                name="test-agent",
                description="Test description"
            )
        )
        
        config = A2AServiceConfig(
            services={"test-agent": agent_config}
        )
        
        assert "test-agent" in config.services
        assert config.services["test-agent"] == agent_config


class TestAgentFactoryConfig:
    
    def test_agent_factory_config_minimal(self):
        agent_config = AgentConfig(
            name="test-agent",
            instructions="Test instructions"
        )
        
        openai_config = {
            "gpt-4": AzureOpenAIConfig(
                model="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/"
            )
        }
        
        config = AgentFactoryConfig(
            agents={"test-agent": agent_config},
            openai_models=openai_config
        )
        
        assert "test-agent" in config.agents
        assert config.agents["test-agent"] == agent_config
        assert "gpt-4" in config.openai_models
        assert config.model_selection == ModelSelectStrategy.first

    def test_agent_factory_config_with_mcp_servers(self):
        from agent_factory.mcp_server.config import MCPServerConfig
        
        agent_config = AgentConfig(
            name="test-agent",
            instructions="Test instructions"
        )
        
        openai_config = {
            "gpt-4": AzureOpenAIConfig(
                model="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/"
            )
        }
        
        mcp_servers = {
            "time": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "mcp_server_time"]
            }
        }
        
        config = AgentFactoryConfig(
            agents={"test-agent": agent_config},
            openai_models=openai_config,
            mcp_servers=mcp_servers,
            model_selection=ModelSelectStrategy.cost
        )
        
        # Check that mcp_servers were converted to MCPServerConfig objects
        assert "time" in config.mcp_servers
        assert isinstance(config.mcp_servers["time"], MCPServerConfig)
        assert config.mcp_servers["time"].type == "stdio"
        assert config.mcp_servers["time"].command == "python"
        assert config.mcp_servers["time"].args == ["-m", "mcp_server_time"]
        assert config.model_selection == ModelSelectStrategy.cost


class TestAgentServiceFactoryConfig:
    
    def test_agent_service_factory_config(self):
        agent_config = AgentConfig(
            name="test-agent",
            instructions="Test instructions"
        )
        
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
        
        service_factory_config = A2AServiceConfig(
            services={"test-agent": a2a_agent_config}
        )
        
        config = AgentServiceFactoryConfig(
            agent_factory=agent_factory_config,
            service_factory=service_factory_config
        )
        
        assert config.agent_factory == agent_factory_config
        assert config.service_factory == service_factory_config
