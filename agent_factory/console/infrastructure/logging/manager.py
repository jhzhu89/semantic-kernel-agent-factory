import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual.widgets import RichLog


class TextualLogHandler(logging.Handler):
    def __init__(self, log_widget: RichLog):
        super().__init__()
        self.log_widget = log_widget
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record):
        if self.log_widget:
            try:
                self.log_widget.write(self.format(record))
            except:
                pass


class LoggingConfig:
    _instance = None

    def __init__(self):
        self.log_level = logging.WARNING
        self.file_handler = None
        self.ui_handler = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def setup_file_logging(self, verbose: bool = False, log_dir: Optional[str] = None) -> str:
        self.log_level = logging.DEBUG if verbose else logging.WARNING
        log_path = Path(log_dir) if log_dir else self._get_default_log_path()
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"agent-factory-{datetime.now().strftime('%Y%m%d')}.log"

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(self.log_level)

        from logging.handlers import RotatingFileHandler

        self.file_handler = RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5, encoding="utf-8"
        )
        self.file_handler.setLevel(self.log_level)
        self.file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(self.file_handler)
        return str(log_file)

    def _get_default_log_path(self) -> Path:
        system = platform.system().lower()
        if system == "linux":
            return Path.home() / ".local" / "share" / "agent-factory" / "logs"
        elif system == "darwin":
            return Path.home() / "Library" / "Logs" / "agent-factory"
        else:
            return Path.home() / ".agent-factory" / "logs"

    def add_ui_logging(self, log_widget):
        if self.ui_handler:
            return
        root_logger = logging.getLogger()
        self.ui_handler = TextualLogHandler(log_widget)
        self.ui_handler.setLevel(self.log_level)
        root_logger.addHandler(self.ui_handler)

    def update_log_level(self, verbose: bool):
        new_level = logging.DEBUG if verbose else logging.WARNING
        if new_level == self.log_level:
            return
        self.log_level = new_level
        root_logger = logging.getLogger()
        root_logger.setLevel(new_level)
        if self.file_handler:
            self.file_handler.setLevel(new_level)
        if self.ui_handler:
            self.ui_handler.setLevel(new_level)

    def get_log_file_path(self) -> Optional[str]:
        return self.file_handler.baseFilename if self.file_handler else None
