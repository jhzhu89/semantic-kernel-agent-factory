import logging
import os
from pathlib import Path

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


@click.group()
def cli():
    pass


@cli.command("chat")
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--verbose", is_flag=True)
def chat_cmd(config_path: str, verbose: bool):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cli_config = AgentFactoryCliConfig.from_file(Path(config_path))

    async def run_chat():
        async with AgentFactory(cli_config.agent_factory) as factory:
            if not factory.get_all_agents():
                click.secho("❌ No agents configured.", fg="red")
                return
            await AgentFactoryConsole(factory, cli_config).run_async()

    anyio.run(run_chat)


@cli.command("list")
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
def list_cmd(config_path: str):
    cli_config = AgentFactoryCliConfig.from_file(Path(config_path))
    click.secho("\n🤖 Configured agents:", fg="bright_white", bold=True)
    for i, agent_name in enumerate(cli_config.agent_factory.agents, 1):
        click.secho(f"  {i}. {agent_name}", fg="bright_cyan")
