"""
Google Earth Engine client bootstrap (optional dependency).

- This module must NOT be imported by default test paths.
- Only import/use it when the `earthengine-api` extra is installed and the user opts in.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


class EarthEngineUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeeConfig:
    """Runtime configuration for Earth Engine initialization."""
    project: str
    # Optional: service account mode (non-interactive)
    service_account_email: str | None = None
    private_key_json_path: str | None = None


def _import_ee() -> Any:
    try:
        import ee  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise EarthEngineUnavailableError(
            "earthengine-api is not installed. Install with: pip install -e '.[gee]'"
        ) from exc
    return ee


def init_ee(cfg: GeeConfig) -> Any:
    """
    Initialize Earth Engine.

    Auth modes:
    - OAuth (interactive): user runs `earthengine authenticate` or `ee.Authenticate()` once,
      then we call `ee.Initialize(project=cfg.project)`.
    - Service account (non-interactive): uses ee.ServiceAccountCredentials(email, key_path).

    Note: credentials must never be committed to the repo.
    """
    ee = _import_ee()

    # If service account info is provided, use it.
    if cfg.service_account_email and cfg.private_key_json_path:
        credentials = ee.ServiceAccountCredentials(cfg.service_account_email, cfg.private_key_json_path)
        ee.Initialize(credentials, project=cfg.project)
        return ee

    # Otherwise rely on prior OAuth authentication on the machine/container.
    ee.Initialize(project=cfg.project)
    return ee


def config_from_env() -> GeeConfig:
    """
    Build GeeConfig from environment variables.

    Required:
      - SKYCOLOR_GEE_PROJECT

    Optional (service account):
      - SKYCOLOR_GEE_SERVICE_ACCOUNT_EMAIL
      - SKYCOLOR_GEE_PRIVATE_KEY_JSON
    """
    project = os.environ.get("SKYCOLOR_GEE_PROJECT")
    if not project:
        raise EarthEngineUnavailableError("Missing env SKYCOLOR_GEE_PROJECT")

    return GeeConfig(
        project=project,
        service_account_email=os.environ.get("SKYCOLOR_GEE_SERVICE_ACCOUNT_EMAIL"),
        private_key_json_path=os.environ.get("SKYCOLOR_GEE_PRIVATE_KEY_JSON"),
    )
