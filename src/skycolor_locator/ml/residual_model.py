"""Residual histogram correction model in pure Python."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp, log, sqrt
from pathlib import Path
from typing import Any

from skycolor_locator.contracts import ColorSignature


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [exp(value - max_logit) for value in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exps]


@dataclass(slots=True)
class ResidualHistogramModel:
    """Linear residual model that predicts delta logits for sky/ground histograms."""

    version: str
    bins: int
    feature_names: list[str]
    weights: list[list[float]]
    bias: list[float]
    eps: float = 1e-6

    def predict_delta_logits(self, x: list[float]) -> list[float]:
        """Predict additive histogram logits from a feature vector."""
        if len(x) != len(self.feature_names):
            raise ValueError("feature vector length does not match model feature names")
        if len(self.weights) != len(self.feature_names):
            raise ValueError("weight matrix row count mismatch")

        out_dim = 2 * self.bins
        if len(self.bias) != out_dim:
            raise ValueError("bias length mismatch")

        out = self.bias.copy()
        for i, value in enumerate(x):
            row = self.weights[i]
            if len(row) != out_dim:
                raise ValueError("weight row dimension mismatch")
            for j in range(out_dim):
                out[j] += value * row[j]
        return out

    def apply_to_signature(self, sig: ColorSignature, x: list[float]) -> ColorSignature:
        """Apply residual correction and return a corrected ColorSignature."""
        if len(sig.signature) != 2 * self.bins:
            raise ValueError("signature length must match model bins")

        delta = self.predict_delta_logits(x)
        sky_phys = sig.sky_hue_hist
        ground_phys = sig.ground_hue_hist

        sky_z = [log(max(value, 0.0) + self.eps) for value in sky_phys]
        ground_z = [log(max(value, 0.0) + self.eps) for value in ground_phys]

        sky_corr = _softmax([z + d for z, d in zip(sky_z, delta[: self.bins], strict=True)])
        ground_corr = _softmax(
            [z + d for z, d in zip(ground_z, delta[self.bins :], strict=True)]
        )
        corrected_signature = sky_corr + ground_corr

        delta_l2 = sqrt(sum(value * value for value in delta))
        meta: dict[str, Any] = dict(sig.meta)
        meta["residual_applied"] = True
        meta["residual_model_version"] = self.version
        meta["residual_delta_l2"] = delta_l2

        quality_flags = list(sig.quality_flags)
        if "residual_applied" not in quality_flags:
            quality_flags.append("residual_applied")

        return ColorSignature(
            hue_bins=list(sig.hue_bins),
            sky_hue_hist=sky_corr,
            ground_hue_hist=ground_corr,
            signature=corrected_signature,
            meta=meta,
            uncertainty_score=sig.uncertainty_score,
            quality_flags=quality_flags,
        )

    def save_json(self, path: str | Path) -> None:
        """Serialize model as JSON file."""
        payload = {
            "version": self.version,
            "bins": self.bins,
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "eps": self.eps,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @classmethod
    def load_json(cls, path: str | Path) -> "ResidualHistogramModel":
        """Load model from JSON file."""
        payload = json.loads(Path(path).read_text())
        return cls(
            version=str(payload["version"]),
            bins=int(payload["bins"]),
            feature_names=[str(name) for name in payload["feature_names"]],
            weights=[[float(v) for v in row] for row in payload["weights"]],
            bias=[float(v) for v in payload["bias"]],
            eps=float(payload.get("eps", 1e-6)),
        )
