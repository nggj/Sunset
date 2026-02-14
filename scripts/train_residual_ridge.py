"""Train a lightweight ridge residual model from JSONL data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from math import log
from pathlib import Path

from skycolor_locator.contracts import AtmosphereState, SurfaceClass, SurfaceState
from skycolor_locator.ml.features import featurize
from skycolor_locator.ml.residual_model import ResidualHistogramModel
from skycolor_locator.signature.core import compute_color_signature


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual histogram ridge model")
    parser.add_argument("--input", required=True, help="Input JSONL dataset path")
    parser.add_argument("--output", required=True, help="Output model JSON path")
    parser.add_argument("--bins", type=int, default=36, help="Hue histogram bin count")
    parser.add_argument("--ridge", type=float, default=1.0, help="Ridge lambda")
    return parser.parse_args()


def _load_numpy() -> object:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "numpy is required for training. Install optional deps: pip install -e '.[ml]'"
        ) from exc
    return np


def _load_sample(line: str) -> dict[str, object]:
    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("Each JSONL line must decode to an object")
    return data


def main() -> None:
    args = _parse_args()
    np = _load_numpy()

    rows_x: list[list[float]] = []
    rows_y: list[list[float]] = []
    feature_names_ref: list[str] | None = None
    eps = 1e-6

    for raw_line in Path(args.input).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sample = _load_sample(line)

        dt = datetime.fromisoformat(str(sample["time_utc"]))
        lat = float(sample["lat"])
        lon = float(sample["lon"])

        atmos = AtmosphereState(
            cloud_fraction=float(sample["cloud_fraction"]),
            aerosol_optical_depth=float(sample["aerosol_optical_depth"]),
            total_ozone_du=float(sample["total_ozone_du"]),
            humidity=float(sample.get("humidity")) if sample.get("humidity") is not None else None,
            visibility_km=(
                float(sample.get("visibility_km"))
                if sample.get("visibility_km") is not None
                else None
            ),
            pressure_hpa=(
                float(sample.get("pressure_hpa")) if sample.get("pressure_hpa") is not None else None
            ),
            cloud_optical_depth=(
                float(sample.get("cloud_optical_depth"))
                if sample.get("cloud_optical_depth") is not None
                else None
            ),
            missing_realtime=bool(sample.get("missing_realtime", False)),
        )
        surface = SurfaceState(
            surface_class=SurfaceClass(str(sample["surface_class"])),
            dominant_albedo=float(sample["dominant_albedo"]),
            landcover_mix={
                str(k): float(v)
                for k, v in dict(sample.get("landcover_mix", {})).items()  # type: ignore[arg-type]
            },
        )

        baseline = compute_color_signature(dt, lat, lon, atmos, surface, {"bins": args.bins})
        x, feature_names = featurize(dt, lat, lon, atmos, surface)
        if feature_names_ref is None:
            feature_names_ref = feature_names
        elif feature_names_ref != feature_names:
            raise ValueError("Feature name mismatch across samples")

        target_sky = [float(v) for v in sample["target_sky_hue_hist"]]  # type: ignore[index]
        target_ground = [float(v) for v in sample["target_ground_hue_hist"]]  # type: ignore[index]
        if len(target_sky) != args.bins or len(target_ground) != args.bins:
            raise ValueError("target hist lengths must match bins")

        z_phys = [log(v + eps) for v in baseline.signature]
        z_true = [log(v + eps) for v in (target_sky + target_ground)]
        delta = [zt - zp for zt, zp in zip(z_true, z_phys, strict=True)]

        rows_x.append(x)
        rows_y.append(delta)

    if not rows_x:
        raise ValueError("Dataset is empty")
    assert feature_names_ref is not None

    x_mat = np.asarray(rows_x, dtype=float)
    y_mat = np.asarray(rows_y, dtype=float)

    n_samples, n_features = x_mat.shape
    out_dim = y_mat.shape[1]

    x_aug = np.concatenate([x_mat, np.ones((n_samples, 1), dtype=float)], axis=1)
    eye = np.eye(n_features + 1, dtype=float)
    eye[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + float(args.ridge) * eye
    rhs = x_aug.T @ y_mat
    beta = np.linalg.solve(lhs, rhs)

    weights = beta[:-1, :]
    bias = beta[-1, :]

    pred = x_aug @ beta
    mse = float(np.mean((pred - y_mat) ** 2))

    model = ResidualHistogramModel(
        version="residual_hist_ridge_v1",
        bins=int(args.bins),
        feature_names=feature_names_ref,
        weights=weights.tolist(),
        bias=bias.tolist(),
        eps=eps,
    )
    model.save_json(args.output)

    print(f"samples={n_samples} features={n_features} out_dim={out_dim}")
    print(f"logit_mse={mse:.8f}")
    print(f"saved_model={args.output}")


if __name__ == "__main__":
    main()
