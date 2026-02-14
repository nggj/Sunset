"""Deprecated experimental module.

Canonical implementation lives in `src/skycolor_locator/sky/analytic.py`.
This file is retained only as a research/prototyping reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import math
import numpy as np


# ---------------------------
# 0) 데이터 구조 (실서비스에서는 Earth Engine/기상 API로 채움)
# ---------------------------

@dataclass
class AtmosphereState:
    """
    time/lat/lon에서의 대기 상태(실시간/준실시간).
    - aod_550: 550nm 에어로졸 광학두께 (위성/재분석)
    - ozone_du: 총 오존량(Dobson Unit) (위성/재분석)
    - cloud_fraction, cloud_optical_depth: 구름 (위성)
    - visibility_km: 가시거리(있으면 turbidity 추정에 사용)
    """
    aod_550: float = 0.1
    ozone_du: float = 300.0
    cloud_fraction: float = 0.0
    cloud_optical_depth: float = 0.0
    visibility_km: Optional[float] = None


@dataclass
class SurfaceClass:
    """
    지표 팔레트(느리게 업데이트되는 상수 + 정적 상수로부터 구성).
    fraction: 해당 색/클래스가 이미지에서 차지할 비율(근사).
    rgb: sRGB (0..1)
    """
    rgb: np.ndarray
    fraction: float


@dataclass
class SurfaceState:
    classes: List[SurfaceClass]


# ---------------------------
# 1) 태양 위치 계산 (UTC + lat/lon -> solar zenith/azimuth)
# ---------------------------

def _julian_day(dt_utc: datetime) -> float:
    if dt_utc.tzinfo is None:
        raise ValueError("dt must be timezone-aware")
    dt_utc = dt_utc.astimezone(timezone.utc)

    year, month, day = dt_utc.year, dt_utc.month, dt_utc.day
    hour = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600 + dt_utc.microsecond / 3.6e9

    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + A // 4

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5 + hour / 24.0
    return jd


def solar_position(dt: datetime, lat_deg: float, lon_deg: float) -> Tuple[float, float, float]:
    """
    반환:
      - sza_deg: Solar Zenith Angle (0=머리 위, 90=수평선)
      - saz_deg: Solar Azimuth (북=0°, 동=90°)
      - sun_elev_deg: 태양 고도(=90-sza)
    """
    dt = dt.astimezone(timezone.utc)
    jd = _julian_day(dt)
    T = (jd - 2451545.0) / 36525.0  # Julian century

    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)

    Mrad = math.radians(M)
    C = ((1.914602 - T * (0.004817 + 0.000014 * T)) * math.sin(Mrad)
         + (0.019993 - 0.000101 * T) * math.sin(2 * Mrad)
         + 0.000289 * math.sin(3 * Mrad))
    true_long = L0 + C

    omega = 125.04 - 1934.136 * T
    lambda_app = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    epsilon0 = 23 + (26 + ((21.448 - T * (46.815 + T * (0.00059 - T * 0.001813)))) / 60) / 60
    epsilon = epsilon0 + 0.00256 * math.cos(math.radians(omega))

    decl = math.degrees(math.asin(math.sin(math.radians(epsilon)) * math.sin(math.radians(lambda_app))))

    y = math.tan(math.radians(epsilon / 2)) ** 2
    Etime = 4 * math.degrees(
        y * math.sin(2 * math.radians(L0))
        - 2 * e * math.sin(Mrad)
        + 4 * e * y * math.sin(Mrad) * math.cos(2 * math.radians(L0))
        - 0.5 * y * y * math.sin(4 * math.radians(L0))
        - 1.25 * e * e * math.sin(2 * Mrad)
    )

    minutes = dt.hour * 60 + dt.minute + dt.second / 60 + dt.microsecond / 6e7
    tst = (minutes + Etime + 4 * lon_deg) % 1440  # True solar time
    hour_angle = tst / 4 - 180
    if hour_angle < -180:
        hour_angle += 360

    lat_rad = math.radians(lat_deg)
    decl_rad = math.radians(decl)
    ha_rad = math.radians(hour_angle)

    cos_zenith = math.sin(lat_rad) * math.sin(decl_rad) + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad)
    cos_zenith = min(1.0, max(-1.0, cos_zenith))
    sza_deg = math.degrees(math.acos(cos_zenith))
    sun_elev_deg = 90 - sza_deg

    # azimuth (북=0°, 동=90°)
    sin_az = -(math.sin(ha_rad) * math.cos(decl_rad)) / (math.cos(math.radians(sza_deg)) + 1e-12)
    cos_az = (math.sin(decl_rad) - math.sin(lat_rad) * math.cos(math.radians(sza_deg))) / (
        (math.cos(lat_rad) * math.sin(math.radians(sza_deg))) + 1e-12
    )
    saz_deg = (math.degrees(math.atan2(sin_az, cos_az)) + 180) % 360

    return sza_deg, saz_deg, sun_elev_deg


# ---------------------------
# 2) Preetham/Perez 하늘 모델 (xyY -> sRGB)
# ---------------------------

def estimate_turbidity(aod_550: Optional[float] = None, visibility_km: Optional[float] = None) -> float:
    """
    실서비스에서는 더 정교한 추정(에어로졸 성분/습도/PM 등) 권장.
    여기서는 간단히 AOD 기반으로 turbidity T를 근사.
    """
    if aod_550 is not None:
        T = 2.0 + 10.0 * max(0.0, float(aod_550))  # heuristic
    elif visibility_km is not None:
        T = 2.0 + 20.0 * max(0.0, (10.0 - visibility_km) / 10.0)
    else:
        T = 3.0
    return float(np.clip(T, 1.8, 10.0))


def perez_coeffs(T: float) -> Dict[str, Tuple[float, float, float, float, float]]:
    # (A,B,C,D,E)
    return {
        "Y": (0.1787 * T - 1.4630,
              -0.3554 * T + 0.4275,
              -0.0227 * T + 5.3251,
              0.1206 * T - 2.5771,
              -0.0670 * T + 0.3703),
        "x": (-0.0193 * T - 0.2592,
              -0.0665 * T + 0.0008,
              -0.0004 * T + 0.2125,
              -0.0641 * T - 0.8989,
              -0.0033 * T + 0.0452),
        "y": (-0.0167 * T - 0.2608,
              -0.0950 * T + 0.0092,
              -0.0079 * T + 0.2102,
              -0.0441 * T - 1.6537,
              -0.0109 * T + 0.0529),
    }


# Zenith chromaticity polynomial matrices (Preetham 계열)
_Mx = np.array([
    [0.00166, -0.00375, 0.00209, 0.0],
    [-0.02903, 0.06377, -0.03202, 0.00394],
    [0.11693, -0.21196, 0.06052, 0.25886]
])
_My = np.array([
    [0.00275, -0.00610, 0.00317, 0.0],
    [-0.04214, 0.08970, -0.04153, 0.00516],
    [0.15346, -0.26756, 0.06670, 0.26688]
])


def zenith_luminance_Yz(T: float, theta_s_rad: float) -> float:
    chi = (4 / 9 - T / 120) * (math.pi - 2 * theta_s_rad)
    return (4.0453 * T - 4.9710) * math.tan(chi) - 0.2155 * T + 2.4192


def zenith_chromaticity_xy(T: float, theta_s_rad: float) -> Tuple[float, float]:
    t = np.array([T * T, T, 1.0])
    th = np.array([theta_s_rad ** 3, theta_s_rad ** 2, theta_s_rad, 1.0])
    xz = float(t @ _Mx @ th)
    yz = float(t @ _My @ th)
    return xz, yz


def perez_F(theta: np.ndarray, gamma: np.ndarray, coeffs: Tuple[float, float, float, float, float]) -> np.ndarray:
    A, B, C, D, E = coeffs
    cos_theta = np.clip(np.cos(theta), 0.01, None)
    return (1 + A * np.exp(B / cos_theta)) * (1 + C * np.exp(D * gamma) + E * (np.cos(gamma) ** 2))


_M_XYZ_to_sRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]
])


def xyz_to_srgb(XYZ: np.ndarray) -> np.ndarray:
    rgb_lin = XYZ @ _M_XYZ_to_sRGB.T
    rgb_lin = np.clip(rgb_lin, 0.0, None)
    a = 0.055
    rgb = np.where(rgb_lin <= 0.0031308, 12.92 * rgb_lin, (1 + a) * np.power(rgb_lin, 1 / 2.4) - a)
    return np.clip(rgb, 0.0, 1.0)


def ozone_rgb_correction(rgb: np.ndarray, ozone_du: Optional[float], sza_deg: float, aod_550: float) -> np.ndarray:
    """
    Lange et al. 결과(오존 영향이 SZA/TOC/VZA에 따라 증가, 에어로졸에 의해 복잡하게 변함)를
    '간단한 보정항'으로 흉내낸 버전.
    - 실서비스: 스펙트럴/RT 또는 ML 잔차로 대체 권장.
    """
    if ozone_du is None:
        return rgb
    s = np.clip((sza_deg - 50) / 40, 0.0, 1.0)  # 50° 이후 영향 증가, 90°에서 1
    k = 0.25 * (float(ozone_du) / 300.0) * float(s) * math.exp(-2.0 * max(float(aod_550), 0.0))
    corr = np.array([1 - 0.35 * k, 1 - 0.55 * k, 1.0])
    return np.clip(rgb * corr, 0.0, 1.0)


def render_sky_rgb(
    dt: datetime,
    lat_deg: float,
    lon_deg: float,
    atmos: AtmosphereState,
    n_az: int = 128,
    n_el: int = 64,
) -> Tuple[np.ndarray, Dict]:
    """
    하늘 돔(zenith->horizon) 색을 (n_el x n_az x 3) sRGB로 반환.
    """
    sza_deg, saz_deg, sun_elev_deg = solar_position(dt, lat_deg, lon_deg)

    theta_s = math.radians(sza_deg)
    phi_s = math.radians(saz_deg)
    T = estimate_turbidity(atmos.aod_550, atmos.visibility_km)

    coeffs = perez_coeffs(T)
    Yz = zenith_luminance_Yz(T, theta_s)
    xz, yz = zenith_chromaticity_xy(T, theta_s)

    theta = np.linspace(0, math.pi / 2, n_el)[:, None]  # (n_el,1)
    phi = np.linspace(0, 2 * math.pi, n_az, endpoint=False)[None, :]  # (1,n_az)

    cos_gamma = np.sin(theta) * math.sin(theta_s) * np.cos(phi - phi_s) + np.cos(theta) * math.cos(theta_s)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)

    F_Y = perez_F(theta, gamma, coeffs["Y"])
    F_x = perez_F(theta, gamma, coeffs["x"])
    F_y = perez_F(theta, gamma, coeffs["y"])

    F_Y0 = float(perez_F(np.array(0.0), np.array(theta_s), coeffs["Y"]))
    F_x0 = float(perez_F(np.array(0.0), np.array(theta_s), coeffs["x"]))
    F_y0 = float(perez_F(np.array(0.0), np.array(theta_s), coeffs["y"]))

    Y = Yz * F_Y / F_Y0
    x = xz * F_x / F_x0
    y = yz * F_y / F_y0

    # 히스토그램용이므로 절대 광도 대신 상대 광도 사용 (Y/Yz)
    Yn = Y / (Yz + 1e-9)

    y_safe = np.clip(y, 1e-6, None)
    X = x * (Yn / y_safe)
    Z = (1 - x - y) * (Yn / y_safe)
    XYZ = np.stack([X, Yn, Z], axis=-1)

    rgb = xyz_to_srgb(XYZ)

    # 오존 보정(간이)
    rgb = ozone_rgb_correction(rgb, atmos.ozone_du, sza_deg, atmos.aod_550)

    # 구름(간이): overcast 쪽으로 blend
    if atmos.cloud_fraction > 0:
        overcast = np.array([0.75, 0.75, 0.78])
        cf = np.clip(atmos.cloud_fraction * (1 - math.exp(-0.3 * atmos.cloud_optical_depth)), 0.0, 1.0)
        rgb = (1 - cf) * rgb + cf * overcast

    meta = {
        "sza_deg": sza_deg,
        "saz_deg": saz_deg,
        "sun_elev_deg": sun_elev_deg,
        "turbidity": T,
    }
    return rgb, meta


# ---------------------------
# 3) Ground(지표) 색 근사 + 합성
# ---------------------------

def render_ground_rgb(
    n_az: int,
    n_el_ground: int,
    surface: SurfaceState,
    sun_elev_deg: float,
    atmos: AtmosphereState,
    sky_horizon_rgb: np.ndarray,
) -> np.ndarray:
    """
    지표를 '팔레트 혼합 + 태양고도 기반 조도 + haze'로 근사한 ground 이미지.
    """
    fractions = np.array([c.fraction for c in surface.classes], dtype=float)
    fractions = fractions / (fractions.sum() + 1e-12)
    cum = np.cumsum(fractions)

    # (단순) azimuth 방향으로 클래스가 섞여 보이도록 결정적 배치
    idx = np.searchsorted(cum, (np.arange(n_az) + 0.5) / n_az)
    base = np.stack([surface.classes[k].rgb for k in idx], axis=0)  # (n_az,3)

    direct = max(0.0, math.sin(math.radians(sun_elev_deg)))  # flat ground normal
    ambient = 0.25 + 0.4 * atmos.cloud_fraction
    illum = ambient + 0.9 * direct * (1 - atmos.cloud_fraction)

    ground = np.clip(base * illum, 0.0, 1.0)

    # 간이 haze (AOD가 크면 지평선색으로 섞임)
    haze_strength = 1 - math.exp(-2.5 * max(atmos.aod_550, 0.0))
    ground = (1 - haze_strength) * ground + haze_strength * sky_horizon_rgb
    ground = np.clip(ground, 0.0, 1.0)

    return np.repeat(ground[None, :, :], n_el_ground, axis=0)


# ---------------------------
# 4) 색 분포 곡선(=Hue histogram curve) 계산
# ---------------------------

def srgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-6

    idx = (mx == r) & mask
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    idx = (mx == g) & mask
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2
    idx = (mx == b) & mask
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4

    h = (h / 6.0) % 1.0

    s = np.zeros_like(mx)
    s[mx > 1e-6] = diff[mx > 1e-6] / mx[mx > 1e-6]
    v = mx

    return np.stack([h, s, v], axis=-1)


def hue_histogram(rgb: np.ndarray, bins: int = 180, weight_mode: str = "sv") -> Tuple[np.ndarray, np.ndarray]:
    hsv = srgb_to_hsv(rgb)
    h = hsv[..., 0].ravel()
    s = hsv[..., 1].ravel()
    v = hsv[..., 2].ravel()

    if weight_mode == "sv":
        w = s * v
    elif weight_mode == "s":
        w = s
    elif weight_mode == "v":
        w = v
    else:
        w = np.ones_like(h)

    hist, edges = np.histogram(h, bins=bins, range=(0, 1), weights=w, density=False)
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-12
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist


def smooth_circular(hist: np.ndarray, sigma_bins: float = 2.0) -> np.ndarray:
    radius = int(3 * sigma_bins)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()

    padded = np.concatenate([hist[-radius:], hist, hist[:radius]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def compute_color_signature(
    dt: datetime,
    lat_deg: float,
    lon_deg: float,
    atmos: AtmosphereState,
    surface: SurfaceState,
    bins: int = 180,
    n_az: int = 128,
    n_el_sky: int = 64,
    n_el_ground: int = 32,
    smooth_sigma_bins: float = 2.0,
) -> Dict:
    """
    핵심 출력:
      - sky_hue_hist: 하늘 Hue 분포 곡선
      - ground_hue_hist: 지표 Hue 분포 곡선
      - signature: [sky_hist | ground_hist] concat (벡터검색용)
    """
    sky_rgb, meta = render_sky_rgb(dt, lat_deg, lon_deg, atmos, n_az=n_az, n_el=n_el_sky)
    sky_horizon_rgb = sky_rgb[-1, :, :].mean(axis=0)

    ground_rgb = render_ground_rgb(
        n_az=n_az,
        n_el_ground=n_el_ground,
        surface=surface,
        sun_elev_deg=meta["sun_elev_deg"],
        atmos=atmos,
        sky_horizon_rgb=sky_horizon_rgb,
    )

    hue_bins, sky_hist = hue_histogram(sky_rgb, bins=bins, weight_mode="sv")
    _, ground_hist = hue_histogram(ground_rgb, bins=bins, weight_mode="sv")

    if smooth_sigma_bins and smooth_sigma_bins > 0:
        sky_hist = smooth_circular(sky_hist, sigma_bins=smooth_sigma_bins)
        ground_hist = smooth_circular(ground_hist, sigma_bins=smooth_sigma_bins)

    signature = np.concatenate([sky_hist, ground_hist], axis=0)

    return {
        "meta": meta,
        "hue_bins": hue_bins,            # 0..1 (0~360°로 쓰려면 *360)
        "sky_hue_hist": sky_hist,
        "ground_hue_hist": ground_hist,
        "signature": signature,
    }


# ---------------------------
# 5) (옵션) 시그니처 비교 함수 (검색/랭킹에 사용)
# ---------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(1 - np.dot(a, b) / denom)


def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    """1D histogram Earth Mover's Distance 근사 (cumsum 차이 평균)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return float(np.mean(np.abs(np.cumsum(p) - np.cumsum(q))))


# ---------------------------
# 사용 예시
# ---------------------------

if __name__ == "__main__":
    # 예시 입력 (실서비스에서는 이 부분이 Earth Engine + 기상으로 채워짐)
    atmos = AtmosphereState(aod_550=0.15, ozone_du=320, cloud_fraction=0.2, cloud_optical_depth=5)

    surface = SurfaceState(classes=[
        SurfaceClass(rgb=np.array([0.12, 0.35, 0.12]), fraction=0.55),  # vegetation
        SurfaceClass(rgb=np.array([0.25, 0.25, 0.28]), fraction=0.30),  # urban
        SurfaceClass(rgb=np.array([0.00, 0.25, 0.45]), fraction=0.15),  # water
    ])

    sig = compute_color_signature(
        dt=datetime(2026, 2, 14, 6, 0, tzinfo=timezone.utc),
        lat_deg=37.5665,
        lon_deg=126.9780,
        atmos=atmos,
        surface=surface,
        bins=180
    )

    print(sig["meta"])
    print("signature dim:", sig["signature"].shape)

