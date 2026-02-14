"""Smoke tests for the package CLI."""

from skycolor_locator.__main__ import main


def test_cli_import_smoke() -> None:
    """Ensure CLI entrypoint can be imported and executed."""
    assert main([]) == 0
