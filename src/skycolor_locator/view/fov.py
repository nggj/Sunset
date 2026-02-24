"""Camera field-of-view ray sampling helpers."""

from __future__ import annotations

from math import asin, atan2, cos, radians, sin, sqrt, tan

from skycolor_locator.contracts import CameraProfile

Vec3 = tuple[float, float, float]


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec3) -> float:
    return sqrt(max(0.0, _dot(v, v)))


def _normalize(v: Vec3) -> Vec3:
    n = _norm(v)
    if n <= 0.0:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _rotate_about_axis(v: Vec3, axis: Vec3, angle_deg: float) -> Vec3:
    axis_n = _normalize(axis)
    theta = radians(angle_deg)
    c = cos(theta)
    s = sin(theta)
    term1 = (v[0] * c, v[1] * c, v[2] * c)
    cross = _cross(axis_n, v)
    term2 = (cross[0] * s, cross[1] * s, cross[2] * s)
    dot = _dot(axis_n, v)
    term3 = (
        axis_n[0] * dot * (1.0 - c),
        axis_n[1] * dot * (1.0 - c),
        axis_n[2] * dot * (1.0 - c),
    )
    return (term1[0] + term2[0] + term3[0], term1[1] + term2[1] + term3[1], term1[2] + term2[2] + term3[2])


def world_dir_from_az_el(az_deg: float, el_deg: float) -> Vec3:
    """Convert world azimuth/elevation (deg) to ENU direction unit vector."""
    az = radians(az_deg)
    el = radians(el_deg)
    x_east = cos(el) * sin(az)
    y_north = cos(el) * cos(az)
    z_up = sin(el)
    return _normalize((x_east, y_north, z_up))


def camera_basis(camera: CameraProfile) -> tuple[Vec3, Vec3, Vec3]:
    """Return camera basis vectors (right, up, forward) in world ENU frame."""
    forward = world_dir_from_az_el(camera.yaw_deg, camera.pitch_deg)
    world_up: Vec3 = (0.0, 0.0, 1.0)
    right = _cross(forward, world_up)
    if _norm(right) <= 1e-9:
        right = (1.0, 0.0, 0.0)
    right = _normalize(right)
    up = _normalize(_cross(right, forward))

    if abs(camera.roll_deg) > 1e-12:
        right = _normalize(_rotate_about_axis(right, forward, camera.roll_deg))
        up = _normalize(_rotate_about_axis(up, forward, camera.roll_deg))

    return right, up, _normalize(forward)


def ray_dir_for_pixel(i: int, j: int, width: int, height: int, camera: CameraProfile) -> Vec3:
    """Compute world ray direction for pixel (i,j) in camera image plane."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    u = ((i + 0.5) / width) * 2.0 - 1.0
    v = 1.0 - ((j + 0.5) / height) * 2.0

    x = tan(radians(camera.hfov_deg) / 2.0) * u
    y = tan(radians(camera.vfov_deg) / 2.0) * v

    right, up, forward = camera_basis(camera)
    direction = (
        forward[0] + x * right[0] + y * up[0],
        forward[1] + x * right[1] + y * up[1],
        forward[2] + x * right[2] + y * up[2],
    )
    return _normalize(direction)


def az_el_from_world_dir(direction: Vec3) -> tuple[float, float]:
    """Convert ENU world direction to azimuth [0,360) and elevation degrees."""
    d = _normalize(direction)
    az = (atan2(d[0], d[1]) * 180.0 / 3.141592653589793) % 360.0
    el = asin(max(-1.0, min(1.0, d[2]))) * 180.0 / 3.141592653589793
    return az, el


def sample_sky_in_camera_view(
    sky_rgb: list[list[list[float]]],
    n_az: int,
    n_el: int,
    camera: CameraProfile,
    horizon_profile: list[float] | None = None,
) -> tuple[list[list[list[float] | None]], list[list[float]], int]:
    """Sample sky buffer into camera image grid with ground masking."""
    width = int(camera.sample_width)
    height = int(camera.sample_height)
    view_grid: list[list[list[float] | None]] = []
    sky_pixels: list[list[float]] = []
    ground_pixel_count = 0

    for j in range(height):
        row: list[list[float] | None] = []
        for i in range(width):
            direction = ray_dir_for_pixel(i, j, width, height, camera)
            az_deg, el_deg = az_el_from_world_dir(direction)
            horizon_el = 0.0
            if horizon_profile is not None and len(horizon_profile) > 0:
                h_idx = min(int((az_deg / 360.0) * len(horizon_profile)), len(horizon_profile) - 1)
                horizon_el = float(horizon_profile[h_idx])
            if el_deg < horizon_el:
                row.append(None)
                ground_pixel_count += 1
                continue

            az_idx = min(int((az_deg / 360.0) * n_az), n_az - 1)
            el_clamped = max(0.0, min(90.0, el_deg))
            el_idx = min(int((el_clamped / 90.0) * (n_el - 1)), n_el - 1)
            pixel = list(sky_rgb[el_idx][az_idx])
            row.append(pixel)
            sky_pixels.append(pixel)
        view_grid.append(row)

    return view_grid, sky_pixels, ground_pixel_count
