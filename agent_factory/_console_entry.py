"""
Safe entry point for agent-factory console command.
This module provides a safe entry point that checks for console dependencies.
"""

import sys


def main():
    """
    Main entry point for the agent-factory command.
    Checks if console dependencies are installed before proceeding.
    """
    missing_deps = []

    # Check if console dependencies are available
    try:
        import click
    except ImportError:
        missing_deps.append("click")

    try:
        import anyio
    except ImportError:
        missing_deps.append("anyio")

    try:
        import textual
    except ImportError:
        missing_deps.append("textual")

    if missing_deps:
        print("❌ Console dependencies are not installed.")
        print(f"Missing packages: {', '.join(missing_deps)}")
        print("\nTo enable the console interface, install with:")
        print("  pip install semantic-kernel-agent-factory[console]")
        print("\nOr install all optional features:")
        print("  pip install semantic-kernel-agent-factory[all]")
        print("\nFor more information, see the project documentation.")
        sys.exit(1)

    # If all dependencies are available, import and run console
    try:
        from agent_factory.console.commands import console

        console()
    except Exception as e:
        print(f"❌ Error starting console: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
