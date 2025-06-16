#!/usr/bin/env python3
"""
Simple web service using the new AgentServiceFactoryConfig structure.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from agent_factory.service import AgentServiceFactoryConfig
from agent_factory.service import AgentServiceFactory

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def configure_uvicorn_logging():
    """Configure uvicorn loggers to use consistent format."""
    formatter = logging.Formatter(LOG_FORMAT)

    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.setLevel(logging.INFO)
        uvicorn_logger.propagate = False


@asynccontextmanager
async def create_app(config_path: str = "agent_service_factory_config.yaml"):
    """Create and manage the application lifecycle."""
    config = AgentServiceFactoryConfig.from_file(config_path)
    logger.info(f"Loaded configuration with {len(config.service_factory.services)} services")

    factory = AgentServiceFactory(config)

    try:
        async with factory:
            app = await factory.create_application()
            logger.info("Agent service factory created successfully")
            yield app
    except Exception as e:
        logger.error(f"Error in application lifecycle: {e}")
        raise


async def main():
    """Main entry point."""
    config_path = "./agent_service_factory_config.yaml"
    if not Path(config_path).exists():
        logger.error(f"Configuration file {config_path} not found")
        return

    configure_uvicorn_logging()

    try:
        async with create_app(config_path) as app:
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                use_colors=False,
                access_log=True,
                log_config=None,
            )
            server = uvicorn.Server(config)
            await server.serve()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")


if __name__ == "__main__":
    asyncio.run(main())
