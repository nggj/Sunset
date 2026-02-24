"""Helpers to enrich base surface state with periodic constants."""

from __future__ import annotations

from skycolor_locator.contracts import PeriodicSurfaceConstants, SurfaceState


def merge_surface_with_periodic(
    surface: SurfaceState, constants: PeriodicSurfaceConstants
) -> SurfaceState:
    """Merge periodic constants into a surface state for signature computation.

    Periodic values override base values when provided:
    - landcover_mix: replace only when periodic mix is non-empty
    - class_rgb: merged as base + periodic overrides
    - periodic_meta: attach periodic provenance metadata
    """
    landcover_mix = (
        dict(constants.landcover_mix) if constants.landcover_mix else dict(surface.landcover_mix)
    )
    class_rgb = dict(surface.class_rgb)
    class_rgb.update(constants.class_rgb)

    periodic_meta = dict(surface.periodic_meta)
    periodic_meta.update(
        {
            "tile_id": constants.tile_id,
            "period_start_utc": constants.period_start_utc.isoformat(),
            "period_end_utc": constants.period_end_utc.isoformat(),
            "source": constants.meta.get("source", "unknown"),
            "meta": dict(constants.meta),
        }
    )

    return SurfaceState(
        surface_class=surface.surface_class,
        dominant_albedo=surface.dominant_albedo,
        landcover_mix=landcover_mix,
        class_rgb=class_rgb,
        periodic_meta=periodic_meta,
    )
