from agent_factory.mcp_server.config import MCPServerConfig


class TestMCPServerConfig:
    
    def test_mcp_server_config_defaults(self):
        config = MCPServerConfig()
        
        assert config.type is None
        assert config.timeout == 5
        assert config.url is None
        assert config.command is None
        assert config.args == []
        assert config.env == {}
        assert config.description is None

    def test_mcp_server_config_stdio(self):
        config = MCPServerConfig(
            type="stdio",
            command="python",
            args=["-m", "mcp_server_time"],
            env={"DEBUG": "true"},
            description="Time server"
        )
        
        assert config.type == "stdio"
        assert config.command == "python"
        assert config.args == ["-m", "mcp_server_time"]
        assert config.env == {"DEBUG": "true"}
        assert config.description == "Time server"

    def test_mcp_server_config_sse(self):
        config = MCPServerConfig(
            type="sse",
            url="https://example.com/mcp",
            timeout=10,
            description="SSE server"
        )
        
        assert config.type == "sse"
        assert config.url == "https://example.com/mcp"
        assert config.timeout == 10
        assert config.description == "SSE server"

    def test_mcp_server_config_custom_timeout(self):
        config = MCPServerConfig(timeout=30)
        
        assert config.timeout == 30

    def test_mcp_server_config_complex_args(self):
        config = MCPServerConfig(
            type="stdio",
            command="node",
            args=["server.js", "--port", "3000", "--debug"],
            env={
                "NODE_ENV": "development",
                "PORT": "3000",
                "DEBUG": "true"
            }
        )
        
        assert config.command == "node"
        assert config.args == ["server.js", "--port", "3000", "--debug"]
        assert len(config.env) == 3
        assert config.env["NODE_ENV"] == "development"

    def test_mcp_server_config_empty_env(self):
        config = MCPServerConfig(
            type="stdio",
            command="python",
            env={}
        )
        
        assert config.env == {}

    def test_mcp_server_config_json_serialization(self):
        config = MCPServerConfig(
            type="stdio",
            command="python",
            args=["-m", "test"],
            timeout=15
        )
        
        config_dict = config.model_dump()
        
        assert config_dict["type"] == "stdio"
        assert config_dict["command"] == "python"
        assert config_dict["args"] == ["-m", "test"]
        assert config_dict["timeout"] == 15

    def test_mcp_server_config_from_dict(self):
        data = {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server"],
            "env": {"KEY": "value"},
            "timeout": 20
        }
        
        config = MCPServerConfig(**data)
        
        assert config.type == "stdio"
        assert config.command == "python" 
        assert config.args == ["-m", "mcp_server"]
        assert config.env == {"KEY": "value"}
        assert config.timeout == 20
