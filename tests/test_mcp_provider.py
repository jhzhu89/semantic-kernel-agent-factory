import pytest
from unittest.mock import Mock, AsyncMock, patch
from agent_factory.mcp_server.provider import MCPProvider
from agent_factory.mcp_server.config import MCPServerConfig


class TestMCPProvider:
    
    @pytest.fixture
    def mcp_configs(self):
        return {
            "time": MCPServerConfig(
                type="stdio",
                command="python",
                args=["-m", "mcp_server_time"]
            ),
            "weather": MCPServerConfig(
                type="stdio", 
                command="node",
                args=["weather-server.js"]
            )
        }

    def test_mcp_provider_initialization(self, mcp_configs):
        provider = MCPProvider(mcp_configs)
        
        assert provider._configs == mcp_configs
        assert provider._plugins == {}
        assert provider._stack is not None

    @pytest.mark.asyncio
    async def test_mcp_provider_context_manager(self, mcp_configs):
        with patch('agent_factory.mcp_server.provider.MCPStdioPlugin') as mock_stdio:
            mock_plugin = AsyncMock()
            mock_stdio.return_value = mock_plugin
            
            async with MCPProvider(mcp_configs) as provider:
                assert len(provider._plugins) == 2
                assert "time" in provider._plugins
                assert "weather" in provider._plugins
                assert mock_stdio.call_count == 2

    @pytest.mark.asyncio
    async def test_mcp_provider_connection_failure(self):
        failing_config = {
            "failing_server": MCPServerConfig(
                type="stdio",
                command="non-existent-command",
                args=[]
            )
        }
        
        with patch('agent_factory.mcp_server.provider.MCPStdioPlugin') as mock_stdio:
            mock_stdio.side_effect = Exception("Connection failed")
            
            # The exception should be raised during context entry
            with pytest.raises(Exception, match="Connection failed"):
                async with MCPProvider(failing_config) as provider:
                    pass

    @pytest.mark.asyncio
    async def test_mcp_provider_empty_configs(self):
        async with MCPProvider({}) as provider:
            assert len(provider._plugins) == 0

    @pytest.mark.asyncio
    async def test_mcp_provider_stack_cleanup_error(self):
        """Test that stack cleanup errors are handled gracefully"""
        config = {
            "test": MCPServerConfig(
                type="stdio",
                command="python",
                args=["-m", "test"]
            )
        }
        
        with patch('agent_factory.mcp_server.provider.MCPStdioPlugin') as mock_stdio:
            mock_plugin = AsyncMock()
            mock_stdio.return_value = mock_plugin
            
            # Mock the stack to simulate cleanup failure
            provider = MCPProvider(config)
            
            with patch.object(provider._stack, 'enter_async_context') as mock_enter:
                mock_enter.side_effect = Exception("Stack error")
                
                # This should raise an exception during context entry
                with pytest.raises(Exception, match="Stack error"):
                    async with provider:
                        pass

    @pytest.mark.asyncio 
    async def test_mcp_provider_plugin_creation(self, mcp_configs):
        with patch('agent_factory.mcp_server.provider.MCPStdioPlugin') as mock_stdio:
            mock_plugin = AsyncMock()
            mock_plugin_instance = AsyncMock()
            mock_plugin.__aenter__.return_value = mock_plugin_instance
            mock_stdio.return_value = mock_plugin
            
            async with MCPProvider({"test": mcp_configs["time"]}) as provider:
                mock_stdio.assert_called_once()
                assert "test" in provider._plugins
                # The provider stores the result of __aenter__, not the plugin itself
                assert provider._plugins["test"] == mock_plugin_instance

    @pytest.mark.asyncio
    async def test_mcp_provider_sse_plugin_creation(self):
        sse_config = {
            "sse_server": MCPServerConfig(
                type="sse",
                url="https://example.com/mcp",
                timeout=10,
                description="Test SSE server"
            )
        }
        
        with patch('agent_factory.mcp_server.provider.MCPSsePlugin') as mock_sse:
            mock_plugin = AsyncMock()
            mock_plugin_instance = AsyncMock()
            mock_plugin.__aenter__.return_value = mock_plugin_instance
            mock_sse.return_value = mock_plugin
            
            async with MCPProvider(sse_config) as provider:
                mock_sse.assert_called_once_with(
                    name="sse_server",
                    url="https://example.com/mcp",
                    request_timeout=10,
                    headers=None,
                    description="Test SSE server"
                )
                assert "sse_server" in provider._plugins
                # The provider stores the result of __aenter__, not the plugin itself
                assert provider._plugins["sse_server"] == mock_plugin_instance

    def test_mcp_provider_plugins_access(self, mcp_configs):
        provider = MCPProvider(mcp_configs)
        mock_plugin = Mock()
        provider._plugins["test"] = mock_plugin
        
        # Test direct access to plugins
        assert "test" in provider._plugins
        assert provider._plugins["test"] == mock_plugin
        assert provider._plugins.get("non-existent") is None

    def test_mcp_provider_plugins_list(self, mcp_configs):
        provider = MCPProvider(mcp_configs)
        provider._plugins["plugin1"] = Mock()
        provider._plugins["plugin2"] = Mock()
        
        plugin_names = list(provider._plugins.keys())
        
        assert "plugin1" in plugin_names
        assert "plugin2" in plugin_names
        assert len(plugin_names) == 2

    def test_mcp_provider_invalid_stdio_config(self):
        invalid_config = {
            "invalid": MCPServerConfig(
                type="stdio",
                command=None,  # This should cause an error
                args=[]
            )
        }
        
        provider = MCPProvider(invalid_config)
        
        # Test that _create_plugin raises ValueError for missing command
        with pytest.raises(ValueError, match="Command is required for stdio MCP server"):
            provider._create_plugin("invalid", invalid_config["invalid"])
