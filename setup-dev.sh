#!/bin/bash
# Development Environment Setup Script
# This script sets up the development environment for semantic-kernel-agent-factory

echo "🚀 Setting up development environment for semantic-kernel-agent-factory..."

# Install the package in editable mode
echo "📦 Installing package in editable mode..."
pip install -e .

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -r dev-requirements.txt

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Available commands:"
echo "  • Run tests: pytest"
echo "  • Format code: black ."
echo "  • Sort imports: isort ."
echo "  • Lint code: ruff check ."
echo "  • Type check: mypy agent_factory"
echo "  • Build docs: sphinx-build -b html docs docs/_build"
echo "  • Build package: python -m build"
echo ""
echo "🎯 Start developing with: cd agent_factory"
