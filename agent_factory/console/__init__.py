"""Console CLI functionality (optional console integration)."""

try:
    from .commands import console

    def is_console_available():
        return True

    _CONSOLE_AVAILABLE = True
    __all__ = ["console", "is_console_available"]

except ImportError as e:
    _CONSOLE_AVAILABLE = False
    _import_error = e

    def _raise_console_import_error(*args, **kwargs):
        raise ImportError(
            "Console functionality requires additional dependencies. "
            "Install with: pip install "
            "'semantic-kernel-agent-factory[console]'"
        ) from _import_error

    # Create dummy function that raises helpful error
    console = _raise_console_import_error  # type: ignore[assignment]

    def is_console_available():
        return False

    __all__ = ["console", "is_console_available"]
