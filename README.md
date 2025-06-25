# Semantic Kernel Agent Factory

A comprehensive SDK for creating and managing AI agents powered by Microsoft Semantic Kernel with MCP (Model Context Protocol) server integration. Build sophisticated conversational agents with tool integration, deploy them as web services, or interact with them through a rich terminal interface.

## Features

- 🤖 **Agent Factory**: Create and manage multiple Semantic Kernel-based agents with different configurations
- 🔗 **MCP Integration**: Connect agents to external tools via Model Context Protocol (stdio and streamable HTTP servers)
- 🖥️ **Interactive Console**: Rich terminal-based chat interface with multi-agent support powered by Textual
- 🌐 **Web Service Factory**: Deploy agents as HTTP/REST APIs with A2A (Agent-to-Agent) protocol support
- ⚡ **Streaming Support**: Real-time response streaming for both console and web interfaces
- 📊 **Health Monitoring**: Built-in MCP server health checks and status monitoring
- 🔧 **Flexible Configuration**: YAML-based configuration with environment variable support
- 🎯 **Structured Outputs**: Support for JSON schema-based response formatting

## Installation

### Basic Installation

```bash
# Install core functionality only
pip install semantic-kernel-agent-factory
```

### Installation with Optional Features

```bash
# For console/CLI interface
pip install semantic-kernel-agent-factory[console]

# For web service deployment
pip install semantic-kernel-agent-factory[service]

# For development (includes testing, linting, and type checking tools)
pip install semantic-kernel-agent-factory[dev]

# For documentation generation
pip install semantic-kernel-agent-factory[docs]

# Install all optional features
pip install semantic-kernel-agent-factory[all]
```

### Development Installation

For local development:

```bash
# Clone the repository
git clone https://github.com/jhzhu89/semantic-kernel-agent-factory
cd semantic-kernel-agent-factory

# Install in editable mode with development dependencies only
pip install -e ".[dev]"

# OR install with all features for comprehensive development/testing
pip install -e ".[dev-all]"

# Use the Makefile for quick setup
make install-dev        # Basic development setup
make install-dev-all    # Development setup with all features
```

## Quick Start

### 1. Console Application

Create a configuration file `config.yaml`:

```yaml
agent_factory:
  agents:
    GeneralAssistant:
      name: "GeneralAssistant"
      instructions: |
        You are a helpful AI assistant.
        Answer questions clearly and concisely.
      model: "gpt-4"
      model_settings:
        temperature: 0.7

  openai_models:
    gpt-4:
      model: "gpt-4"
      api_key: "${OPENAI_API_KEY}"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
```

Run the interactive console:

```bash
# Note: Requires console dependencies
# Install with: pip install semantic-kernel-agent-factory[console]
agent-factory -c config.yaml
```

### 2. Python API - Agent Factory

```python
import asyncio
from agent_factory import AgentFactory, AgentFactoryConfig, AgentConfig

async def main():
    # Create configuration
    config = AgentFactoryConfig(
        agents={
            "assistant": AgentConfig(
                name="assistant",
                instructions="You are a helpful AI assistant",
                model="gpt-4"
            )
        },
        openai_models={
            "gpt-4": {
                "model": "gpt-4",
                "api_key": "your-api-key",
                "endpoint": "your-endpoint"
            }
        }
    )
    
    # Create and use agents
    async with AgentFactory(config) as factory:
        agent = factory.get_agent("assistant")
        # Use the agent for conversations
        
asyncio.run(main())
```

### 3. Web Service Deployment

Create a service configuration `service_config.yaml`:

```yaml
agent_factory:
  agents:
    ChatBot:
      name: "ChatBot"
      instructions: "You are a helpful chatbot"
      model: "gpt-4"
  
  openai_models:
    gpt-4:
      model: "gpt-4"
      api_key: "${OPENAI_API_KEY}"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"

service_factory:
  services:
    ChatBot:
      card:
        name: "ChatBot"
        description: "AI-powered chatbot service"
      enable_token_streaming: true
```

Deploy as web service:

```python
# Note: Requires service dependencies
# Install with: pip install semantic-kernel-agent-factory[service]
from agent_factory import AgentServiceFactory, AgentServiceFactoryConfig
import uvicorn

async def create_app():
    config = AgentServiceFactoryConfig.from_file("service_config.yaml")
    
    async with AgentServiceFactory(config) as factory:
        app = await factory.create_application()
        return app

if __name__ == "__main__":
    uvicorn.run("main:create_app", host="0.0.0.0", port=8000)
```

## MCP Server Integration

Connect agents to external tools using Model Context Protocol servers:

```yaml
agent_factory:
  agents:
    ToolAgent:
      name: "ToolAgent"
      instructions: "You have access to various tools"
      model: "gpt-4"
      mcp_servers: ["time", "kubernetes"]
  
  mcp:
    servers:
      time:
        type: "stdio"
        command: "python"
        args: ["-m", "mcp_server_time"]
      
      kubernetes:
        type: "streamable_http"
        url: "https://k8s-mcp-server.example.com/mcp"
        timeout: 10
```

### Access Token Authentication

For HTTP-based MCP servers that require authentication, there's a limitation with the MCP Python client SDK. While it's easy to obtain user access tokens in HTTP services, the underlying HTTP client utilization doesn't provide a straightforward way to add tokens to HTTP headers when communicating with MCP servers.

**Workaround: Filter-based Token Injection**

As a workaround, the system uses filters to inject access tokens before sending requests to MCP servers. Check out filters.py for details.

**Important Notes:**
- The server should consume the `access_token` for authentication purposes
- **Do not** include `access_token` in the tool's input schema definition
- The token is automatically injected by the filter before the request reaches the MCP server
- This is a temporary workaround until the MCP Python client SDK provides better header customization support

## Console Features

The interactive console provides:

- **Multi-Agent Chat**: Switch between different agents in tabbed interface
- **Real-time Streaming**: See responses as they're generated
- **MCP Status Monitoring**: Live health checks of connected MCP servers
- **Function Call Visibility**: See tool calls and results in real-time
- **Keyboard shortcuts**:
  - `Ctrl+Enter`: Send message
  - `Ctrl+L`: Clear chat
  - `F1`: Toggle agent panel
  - `F2`: Toggle logs
  - `Ctrl+W`: Close tab

## Configuration

### Agent Configuration

```yaml
agent_factory:
  agents:
    MyAgent:
      name: "MyAgent"
      instructions: "System prompt for the agent"
      model: "gpt-4"
      model_settings:
        temperature: 0.7
        max_tokens: 2000
        response_json_schema:  # Optional structured output
          type: "object"
          properties:
            answer: 
              type: "string"
      mcp:
        servers: ["tool1", "tool2"]
```

### OpenAI Models

```yaml
agent_factory:
  openai_models:
    gpt-4:
      model: "gpt-4"
      api_key: "${OPENAI_API_KEY}"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
    
    gpt-3.5-turbo:
      model: "gpt-3.5-turbo"
      api_key: "${OPENAI_API_KEY}"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
```

### MCP Server Types

**Stdio servers** (local processes):
```yaml
mcp:
  servers:
    local_tool:
      type: "stdio"
      command: "python"
      args: ["-m", "my_mcp_server"]
      env:
        DEBUG: "true"
```

**Streamable HTTP servers** (HTTP-based):
```yaml
mcp:
  servers:
    remote_tool:
      type: "streamable_http"
      url: "https://api.example.com/mcp"
      timeout: 15
```

## CLI Commands

```bash
# Start interactive chat (requires console dependencies)
agent-factory -c config.yaml

# List configured agents
agent-factory list -c config.yaml

# Enable verbose logging
agent-factory -c config.yaml --verbose

# Custom log directory
agent-factory -c config.yaml --log-dir /path/to/logs
```

## Environment Variables

Configure using environment variables:

```bash
# OpenAI Configuration
export OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Optional: Agent Factory Settings
export AGENT_FACTORY__MODEL_SELECTION="cost"  # first, cost, latency, quality
export AGENT_FACTORY__MCP_FAILURE_STRATEGY="lenient"  # strict, lenient
```

## Examples

See the `examples/` directory for:
- [`cli_example.yaml`](examples/cli_example.yaml) - Console application setup
- [`agent_service_factory_config.yaml`](examples/agent_service_factory_config.yaml) - Web service configuration
- [`web_service.py`](examples/web_service.py) - Web service deployment example

## Development

### Quick Setup

```bash
# Clone repository
git clone https://github.com/jhzhu89/semantic-kernel-agent-factory
cd semantic-kernel-agent-factory

# Install in editable mode with all development dependencies
pip install -e ".[dev]"
```

### Available Development Tools

The `[dev]` extra includes:
- **Testing**: pytest, pytest-asyncio, pytest-cov, pytest-mock
- **Code Formatting**: black, isort, ruff
- **Type Checking**: mypy with type stubs
- **Linting**: flake8, ruff
- **Coverage**: pytest-cov for test coverage reports

### Development Commands

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=agent_factory --cov-report=html

# Format code
black .
isort .

# Lint code
ruff check .
flake8 .

# Type checking
mypy agent_factory

# Run all quality checks
make test-cov  # Runs tests with coverage
make format    # Formats code
make type-check # Type checking
```

### Optional Development Features

```bash
# Install with console dependencies for development
pip install -e ".[dev,console]"

# For web service development
pip install -e ".[dev,service]"

# Install all features for development
pip install -e ".[dev,all]"
```

## Architecture

The Semantic Kernel Agent Factory consists of several key components:

- **AgentFactory**: Core factory for creating and managing Semantic Kernel agents
- **AgentServiceFactory**: Web service wrapper that exposes agents as HTTP APIs (requires `[service]` extra)
- **MCPProvider**: Manages connections to Model Context Protocol servers
- **Console Application**: Terminal-based interface for interactive agent chat (requires `[console]` extra)
- **Configuration System**: YAML-based configuration with validation

### Optional Components

Different installation options enable additional features:

- **`[console]`**: Interactive terminal interface with Textual UI, Click CLI commands
- **`[service]`**: A2A-based web services, Starlette server support
- **`[docs]`**: Sphinx-based documentation generation
- **`[dev]`**: Development tools for testing, linting, and type checking

## Requirements

- Python 3.10+
- Microsoft Semantic Kernel
- Azure OpenAI or OpenAI API access
- Optional: MCP-compatible tool servers

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Add tests for new functionality
5. Run the test suite and ensure all checks pass:
   ```bash
   # Run tests
   pytest
   
   # Format code
   black .
   isort .
   
   # Lint code
   ruff check .
   flake8 .
   
   # Type checking
   mypy agent_factory
   ```
6. Submit a pull request

### Development Environment Setup

```bash
# Install with console dependencies for development
pip install -e ".[dev,console]"

# For web service development
pip install -e ".[dev,service]"

# For full development environment
pip install -e ".[dev,all]"
```

### Project Structure

- `agent_factory/` - Core library code
- `tests/` - Test suite
- `examples/` - Usage examples
- `docs/` - Documentation (when using `[docs]` extra)

### Code Quality Standards

This project uses:
- **Black** for code formatting
- **isort** for import sorting  
- **Ruff** for linting
- **Flake8** for additional linting
- **mypy** for type checking
- **pytest** for testing with >80% coverage requirement

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support & Documentation

- 📖 [GitHub Repository](https://github.com/jhzhu89/semantic-kernel-agent-factory)
- 🐛 [Issue Tracker](https://github.com/jhzhu89/semantic-kernel-agent-factory/issues)
- 💬 [Discussions](https://github.com/jhzhu89/semantic-kernel-agent-factory/discussions)
- 📚 [Examples](https://github.com/jhzhu89/semantic-kernel-agent-factory/tree/main/examples)

## Related Projects

- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Textual](https://github.com/Textualize/textual) - Powers the console interface
