"""Demo: build a local signature index on a grid and run similarity search."""

from __future__ import annotations

from datetime import datetime, timezone

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skycolor_locator.ingest.mock_providers import MockEarthStateProvider, MockSurfaceProvider  # noqa: E402
from skycolor_locator.orchestrate.batch import GridSpec, build_signature_index  # noqa: E402
from skycolor_locator.signature.core import compute_color_signature  # noqa: E402


def main() -> int:
    """Run local batch indexing and print top-k search results."""
    now_utc = datetime.now(timezone.utc)

    earth = MockEarthStateProvider()
    surface = MockSurfaceProvider()

    grid = GridSpec(
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=100.0,
        lon_max=120.0,
        step_deg=5.0,
    )
    config = {"bins": 24, "n_az": 24, "n_el": 12}

    index, _ = build_signature_index(
        dt=now_utc,
        spec=grid,
        earth_provider=earth,
        surface_provider=surface,
        config=config,
    )

    target_lat = 0.0
    target_lon = 110.0
    target_atmos = earth.get_atmosphere_state(now_utc, target_lat, target_lon)
    target_surface = surface.get_surface_state(target_lat, target_lon)
    target_sig = compute_color_signature(
        now_utc,
        target_lat,
        target_lon,
        target_atmos,
        target_surface,
        config=config,
    )

    top_k = 5
    results = index.query(target_sig.signature, top_k=top_k)

    print("=== Skycolor Locator Local Search Demo ===")
    print(f"time_utc: {now_utc.isoformat()}")
    print(f"target: lat={target_lat:.3f}, lon={target_lon:.3f}")
    print(f"top_k: {top_k}\n")
    print("rank | key                   | distance")
    print("-----+-----------------------+----------")
    for rank, (key, distance) in enumerate(results, start=1):
        print(f"{rank:>4} | {key:<21} | {distance:>8.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
