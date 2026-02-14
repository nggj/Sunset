"""Tests for perceptual_v1 embedding."""

from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.index.metrics import cosine_distance
from skycolor_locator.signature.perceptual import (
    H_BINS,
    PROFILE_BINS,
    S_BINS,
    SKY_BANDS,
    V_BINS,
    compute_perceptual_v1,
    compute_perceptual_v1_from_buffers,
)


def _inputs() -> tuple[datetime, AtmosphereState, SurfaceState]:
    dt = datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc)
    atmos = AtmosphereState(
        cloud_fraction=0.25,
        aerosol_optical_depth=0.12,
        total_ozone_du=305.0,
        visibility_km=22.0,
    )
    surface = SurfaceState(
        surface_class=SurfaceClass.LAND,
        dominant_albedo=0.3,
        landcover_mix={"land": 0.7, "forest": 0.3},
    )
    return dt, atmos, surface


def test_perceptual_v1_is_deterministic() -> None:
    """Same inputs should produce identical perceptual vectors and metadata."""
    dt, atmos, surface = _inputs()

    vec_a, meta_a = compute_perceptual_v1(dt, 35.0, 129.0, atmos, surface)
    vec_b, meta_b = compute_perceptual_v1(dt, 35.0, 129.0, atmos, surface)

    assert vec_a == vec_b
    assert meta_a == meta_b


def test_perceptual_v1_has_expected_shape_and_norm() -> None:
    """Embedding dimension and L2 norm should follow perceptual_v1 design."""
    dt, atmos, surface = _inputs()
    vector, meta = compute_perceptual_v1(dt, 35.0, 129.0, atmos, surface)

    expected_dim = (SKY_BANDS * H_BINS) + S_BINS + V_BINS + PROFILE_BINS + 3 + H_BINS + S_BINS + V_BINS + 3
    norm = sqrt(sum(v * v for v in vector))

    assert len(vector) == expected_dim
    assert meta["vector_dim"] == expected_dim
    assert norm > 0.0
    assert abs(norm - 1.0) < 1e-6


def test_perceptual_v1_detects_luma_gradient_changes() -> None:
    """Different sky luma gradients should change perceptual embedding."""
    n_el = 6
    n_az = 8
    # same hue-ish blue family, only horizon brightness differs
    sky_a = [
        [[0.15, 0.35, 0.70] for _ in range(n_az)] if i < 2 else [[0.25, 0.45, 0.80] for _ in range(n_az)]
        for i in range(n_el)
    ]
    sky_b = [
        [[0.05, 0.20, 0.45] for _ in range(n_az)] if i < 2 else [[0.25, 0.45, 0.80] for _ in range(n_az)]
        for i in range(n_el)
    ]
    ground_pixels = [[0.30, 0.28, 0.22] for _ in range(128)]

    vec_a, _ = compute_perceptual_v1_from_buffers(sky_a, ground_pixels)
    vec_b, _ = compute_perceptual_v1_from_buffers(sky_b, ground_pixels)

    assert cosine_distance(vec_a, vec_b) > 0.01
