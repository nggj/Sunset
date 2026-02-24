"""Lightweight DEM-based terrain horizon helpers."""

from __future__ import annotations

from math import atan2, ceil, cos, degrees, hypot, radians, sin


def _clamp_idx(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def raycast_horizon_profile(
    dem_m: list[list[float]],
    observer_row: int,
    observer_col: int,
    cell_size_m: float,
    max_distance_m: float,
    az_step_deg: float = 5.0,
    observer_height_m: float = 1.7,
) -> dict[str, object]:
    """Compute horizon elevation profile by simple 2D ray casting over DEM grid.

    Returns dict with keys:
    - `az_step_deg`
    - `horizon_profile_deg`: list horizon elevation per azimuth bin [0, 360)
    """
    if not dem_m or not dem_m[0]:
        raise ValueError("dem_m must be a non-empty 2D grid")
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    if max_distance_m <= 0.0:
        raise ValueError("max_distance_m must be positive")
    if az_step_deg <= 0.0:
        raise ValueError("az_step_deg must be positive")

    n_rows = len(dem_m)
    n_cols = len(dem_m[0])
    for row in dem_m:
        if len(row) != n_cols:
            raise ValueError("dem_m rows must share equal column count")

    row0 = _clamp_idx(observer_row, 0, n_rows - 1)
    col0 = _clamp_idx(observer_col, 0, n_cols - 1)
    observer_elev = float(dem_m[row0][col0]) + observer_height_m

    az_count = int(round(360.0 / az_step_deg))
    az_step = 360.0 / az_count
    max_steps = int(ceil(max_distance_m / cell_size_m))

    profile: list[float] = []
    for i in range(az_count):
        az_deg = i * az_step
        az = radians(az_deg)
        dr = -cos(az)
        dc = sin(az)
        max_angle = -90.0

        for step in range(1, max_steps + 1):
            rr = int(round(row0 + step * dr))
            cc = int(round(col0 + step * dc))
            if rr < 0 or rr >= n_rows or cc < 0 or cc >= n_cols:
                break
            distance = hypot((rr - row0) * cell_size_m, (cc - col0) * cell_size_m)
            if distance <= 0.0:
                continue
            elev = float(dem_m[rr][cc])
            angle = degrees(atan2(elev - observer_elev, distance))
            if angle > max_angle:
                max_angle = angle

        profile.append(max_angle)

    return {"az_step_deg": az_step, "horizon_profile_deg": profile}
