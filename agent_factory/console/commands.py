import logging
import os
from pathlib import Path
from typing import Optional

import anyio
import click
from dotenv import load_dotenv

from ..factory import AgentFactory
from .app import AgentFactoryConsole
from .history_config import AgentFactoryCliConfig

logging.basicConfig(
    level=os.getenv("AGENT_FACTORY_LOG", "WARNING").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
load_dotenv()


@click.group(invoke_without_command=True)
@click.option(
    "-c", "--config", "config_path", type=click.Path(exists=True), help="Path to config file"
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def console(ctx, config_path: Optional[str] = None, verbose: bool = False):
    """Agent Factory Console - Interactive chat interface for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["verbose"] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If no subcommand is invoked and config is provided, default to chat
    if ctx.invoked_subcommand is None:
        if config_path:
            ctx.invoke(chat)
        else:
            click.echo(ctx.get_help())


@console.command()
@click.option(
    "-c", "--config", "config_path", type=click.Path(exists=True), help="Path to config file"
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def chat(ctx, config_path: Optional[str] = None, verbose: Optional[bool] = None):
    """Start interactive chat with agents."""
    # Use subcommand parameters if provided, otherwise fall back to parent
    final_config_path = config_path or ctx.obj.get("config_path")
    final_verbose = verbose if verbose is not None else ctx.obj.get("verbose", False)

    if not final_config_path:
        click.secho("❌ Config file is required. Use -c/--config option.", fg="red")
        return

    if final_verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cli_config = AgentFactoryCliConfig.from_file(Path(final_config_path))

    async def run_chat():
        async with AgentFactory(cli_config.agent_factory) as factory:
            if not factory.get_all_agents():
                click.secho("❌ No agents configured.", fg="red")
                return
            await AgentFactoryConsole(factory, cli_config).run_async()

    anyio.run(run_chat)


@console.command()
@click.option(
    "-c", "--config", "config_path", type=click.Path(exists=True), help="Path to config file"
)
@click.pass_context
def list(ctx, config_path: Optional[str] = None):
    """List all configured agents."""
    # Use subcommand parameter if provided, otherwise fall back to parent
    final_config_path = config_path or ctx.obj.get("config_path")

    if not final_config_path:
        click.secho("❌ Config file is required. Use -c/--config option.", fg="red")
        return

    cli_config = AgentFactoryCliConfig.from_file(Path(final_config_path))
    click.secho("\n🤖 Configured agents:", fg="bright_white", bold=True)
    for i, agent_name in enumerate(cli_config.agent_factory.agents, 1):
        click.secho(f"  {i}. {agent_name}", fg="bright_cyan")
