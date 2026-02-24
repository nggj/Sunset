"""Surface state resolver combining base surface provider and periodic constants."""

from __future__ import annotations

from datetime import datetime

from skycolor_locator.contracts import SurfaceState
from skycolor_locator.ingest.interfaces import SurfaceProvider
from skycolor_locator.state.periodic_resolver import PeriodicConstantsResolver


class SurfaceStateResolver:
    """Resolve merged surface state without mutating base provider responses."""

    def __init__(
        self,
        base_provider: SurfaceProvider,
        periodic: PeriodicConstantsResolver | None,
    ) -> None:
        self._base_provider = base_provider
        self._periodic = periodic

    def get_surface_state(self, dt: datetime, lat: float, lon: float) -> SurfaceState:
        """Return base surface state merged with periodic constants when configured."""
        base = self._base_provider.get_surface_state(lat, lon)
        landcover_mix = dict(base.landcover_mix)
        class_rgb = {k: list(v) for k, v in base.class_rgb.items()}
        periodic_meta = dict(base.periodic_meta)

        if self._periodic is not None:
            constants = self._periodic.get_periodic_surface_constants(dt, lat, lon)
            if constants.landcover_mix:
                landcover_mix = dict(constants.landcover_mix)
            if constants.class_rgb:
                class_rgb.update({k: list(v) for k, v in constants.class_rgb.items()})
            periodic_meta = {
                "tile_id": constants.tile_id,
                "period_start_utc": constants.period_start_utc.isoformat(),
                "period_end_utc": constants.period_end_utc.isoformat(),
                **dict(constants.meta),
            }

        return SurfaceState(
            surface_class=base.surface_class,
            dominant_albedo=base.dominant_albedo,
            landcover_mix=landcover_mix,
            class_rgb=class_rgb,
            periodic_meta=periodic_meta,
        )
