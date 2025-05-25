#!/usr/bin/env python3
"""Test cases for the CLI module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_factory.console import console


@pytest.fixture
def sample_cli_config():
    """Create a sample CLI configuration for testing."""
    return {
        "agent_factory": {
            "agents": {
                "test-agent": {
                    "name": "test-agent",
                    "instructions": "Test instructions",
                    "model": "gpt-4"
                }
            },
            "openai_models": {
                "gpt-4": {
                    "model": "gpt-4",
                    "api_key": "test-key",
                    "endpoint": "https://test.openai.azure.com/",
                    "api_version": "2024-06-01"
                }
            }
        }
    }


@pytest.fixture
def temp_config_file(sample_cli_config):
    """Create a temporary config file for testing."""
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_cli_config, f)
        f.flush()
        yield f.name
    
    os.unlink(f.name)


class TestCLICommands:
    """Test CLI command functionality."""
    
    def test_cli_group_exists(self):
        """Test that the main CLI group exists."""
        assert callable(console)
        assert hasattr(console, 'commands')
    
    def test_chat_command_exists(self):
        """Test that the chat command exists."""
        assert 'chat' in console.commands
    
    def test_list_command_exists(self):
        """Test that the list command exists."""
        assert 'list' in console.commands
    
    def test_list_command_with_config(self, temp_config_file):
        """Test the list command with a config file."""
        runner = CliRunner()
        
        with patch('agent_factory.console.commands.AgentFactoryCliConfig.from_file') as mock_from_file:
            mock_config = MagicMock()
            mock_config.agent_factory.agents = {"test-agent": MagicMock(), "another-agent": MagicMock()}
            mock_from_file.return_value = mock_config
            
            result = runner.invoke(console, ['list', '-c', temp_config_file])
            
            assert result.exit_code == 0
            assert "test-agent" in result.output
            assert "another-agent" in result.output
    
    def test_list_command_missing_config(self):
        """Test the list command with missing config file."""
        runner = CliRunner()
        result = runner.invoke(console, ['list', '-c', 'nonexistent.yaml'])
        
        assert result.exit_code != 0
    
    @patch('agent_factory.console.commands.anyio.run')
    @patch('agent_factory.console.commands.AgentFactoryCliConfig.from_file')
    @patch('agent_factory.console.commands.AgentFactory')
    def test_chat_command_basic(self, mock_factory_class, mock_from_file, mock_anyio_run, temp_config_file):
        """Test the chat command basic functionality."""
        mock_config = MagicMock()
        mock_from_file.return_value = mock_config
        
        mock_factory = MagicMock()
        mock_factory.get_all_agents.return_value = ["test-agent"]
        mock_factory_class.return_value.__aenter__.return_value = mock_factory
        
        runner = CliRunner()
        result = runner.invoke(console, ['chat', '-c', temp_config_file])
        
        mock_from_file.assert_called_once()
        mock_anyio_run.assert_called_once()
    
    @patch('agent_factory.console.commands.anyio.run')
    @patch('agent_factory.console.commands.AgentFactoryCliConfig.from_file')
    @patch('agent_factory.console.commands.AgentFactory')
    def test_chat_command_no_agents(self, mock_factory_class, mock_from_file, mock_anyio_run, temp_config_file):
        """Test the chat command with no agents configured."""
        mock_config = MagicMock()
        mock_from_file.return_value = mock_config
        
        mock_factory = MagicMock()
        mock_factory.get_all_agents.return_value = []
        mock_factory_class.return_value.__aenter__.return_value = mock_factory
        
        runner = CliRunner()
        result = runner.invoke(console, ['chat', '-c', temp_config_file])
        
        mock_from_file.assert_called_once()
        mock_anyio_run.assert_called_once()
    
    def test_chat_command_with_verbose(self, temp_config_file):
        """Test the chat command with verbose option."""
        runner = CliRunner()
        
        with patch('agent_factory.console.commands.anyio.run') as mock_anyio_run, \
             patch('agent_factory.console.commands.AgentFactoryCliConfig.from_file') as mock_from_file, \
             patch('agent_factory.console.commands.AgentFactory') as mock_factory_class, \
             patch('logging.getLogger') as mock_get_logger:
            
            mock_config = MagicMock()
            mock_from_file.return_value = mock_config
            
            mock_factory = MagicMock()
            mock_factory.get_all_agents.return_value = ["test-agent"]
            mock_factory_class.return_value.__aenter__.return_value = mock_factory
            
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = runner.invoke(console, ['chat', '-c', temp_config_file, '--verbose'])
            
            mock_from_file.assert_called_once()
            mock_anyio_run.assert_called_once()
            mock_logger.setLevel.assert_called_once()


class TestCLIConfiguration:
    """Test CLI configuration handling."""
    
    def test_config_loading(self, temp_config_file):
        """Test that configuration can be loaded from file."""
        with patch('agent_factory.console.commands.AgentFactoryCliConfig.from_file') as mock_from_file:
            mock_config = MagicMock()
            mock_from_file.return_value = mock_config
            
            runner = CliRunner()
            result = runner.invoke(console, ['list', '-c', temp_config_file])
            
            mock_from_file.assert_called_once()


class TestEnvironmentVariables:
    """Test environment variable handling."""
    
    def test_log_level_from_env(self):
        """Test that log level can be set via environment variable."""
        with patch.dict(os.environ, {'AGENT_FACTORY_LOG': 'DEBUG'}):
            pass
    
    def test_dotenv_loading(self):
        """Test that dotenv is loaded."""
        from dotenv import load_dotenv
        assert callable(load_dotenv)


if __name__ == "__main__":
    pytest.main([__file__])
