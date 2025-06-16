from pathlib import Path
from typing import Optional

import anyio
import click
from dotenv import load_dotenv

from ..core.factory import AgentFactory
from .infrastructure.config.history_config import AgentFactoryCliConfig
from .infrastructure.logging.manager import LoggingConfig
from .ui.console_app import AgentFactoryConsole

load_dotenv()


@click.group(invoke_without_command=True)
@click.option(
    "-c", "--config", "config_path", type=click.Path(exists=True), help="Path to config file"
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--log-dir", "log_dir", type=click.Path(), help="Custom log directory path")
@click.pass_context
def console(
    ctx, config_path: Optional[str] = None, verbose: bool = False, log_dir: Optional[str] = None
):
    ctx.ensure_object(dict)
    LoggingConfig.get_instance().setup_file_logging(verbose, log_dir)
    ctx.obj.update({"config_path": config_path, "verbose": verbose, "log_dir": log_dir})
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
@click.option("--log-dir", "log_dir", type=click.Path(), help="Custom log directory path")
@click.pass_context
def chat(
    ctx,
    config_path: Optional[str] = None,
    verbose: Optional[bool] = None,
    log_dir: Optional[str] = None,
):
    config_path = config_path or ctx.obj.get("config_path")
    verbose = verbose if verbose is not None else ctx.obj.get("verbose", False)
    log_dir = log_dir or ctx.obj.get("log_dir")

    if not config_path:
        click.secho("❌ Config file is required. Use -c/--config option.", fg="red")
        return

    if verbose != ctx.obj.get("verbose", False):
        LoggingConfig.get_instance().update_log_level(verbose)

    cli_config = AgentFactoryCliConfig.from_file(Path(config_path))
    anyio.run(_run_chat_session, cli_config)


async def _run_chat_session(cli_config: AgentFactoryCliConfig):
    async with AgentFactory(cli_config.agent_factory) as factory:
        if not factory.get_all_agents():
            click.secho("❌ No agents configured.", fg="red")
            return
        app = AgentFactoryConsole(factory, cli_config)
        await app.run_async()


@console.command()
@click.option(
    "-c", "--config", "config_path", type=click.Path(exists=True), help="Path to config file"
)
@click.pass_context
def list(ctx, config_path: Optional[str] = None):
    config_path = config_path or ctx.obj.get("config_path")
    if not config_path:
        click.secho("❌ Config file is required. Use -c/--config option.", fg="red")
        return
    cli_config = AgentFactoryCliConfig.from_file(Path(config_path))
    click.secho("\n🤖 Configured agents:", fg="bright_white", bold=True)
    for i, agent_name in enumerate(cli_config.agent_factory.agents, 1):
        click.secho(f"  {i}. {agent_name}", fg="bright_cyan")
