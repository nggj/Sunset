"""Unit tests for data contracts."""

from __future__ import annotations

import json

import pytest

from skycolor_locator.contracts import ColorSignature


def _valid_signature() -> ColorSignature:
    return ColorSignature(
        hue_bins=[0.0, 0.5, 1.0],
        sky_hue_hist=[0.2, 0.3, 0.5],
        ground_hue_hist=[0.4, 0.1, 0.5],
        signature=[0.2, 0.3, 0.5, 0.4, 0.1, 0.5],
        meta={"sun_elev_deg": 21.5},
        uncertainty_score=0.1,
        quality_flags=["ok"],
    )


def test_color_signature_validates_and_serializes() -> None:
    """A valid signature is created and can be JSON serialized via to_dict."""
    signature = _valid_signature()
    payload = signature.to_dict()

    assert payload["signature"] == [0.2, 0.3, 0.5, 0.4, 0.1, 0.5]
    assert json.loads(json.dumps(payload))["meta"]["sun_elev_deg"] == 21.5


def test_color_signature_rejects_negative_hist_values() -> None:
    """Histogram validation fails when negative entries exist."""
    with pytest.raises(ValueError, match="must not contain negative"):
        ColorSignature(
            hue_bins=[0.0, 0.5],
            sky_hue_hist=[-0.1, 1.1],
            ground_hue_hist=[0.5, 0.5],
            signature=[-0.1, 1.1, 0.5, 0.5],
            meta={},
            uncertainty_score=0.2,
            quality_flags=[],
        )


def test_color_signature_rejects_invalid_signature_length() -> None:
    """Signature length must be exactly 2 * len(hue_bins)."""
    with pytest.raises(ValueError, match=r"2 \* len\(hue_bins\)"):
        ColorSignature(
            hue_bins=[0.0, 0.5, 1.0],
            sky_hue_hist=[0.2, 0.3, 0.5],
            ground_hue_hist=[0.4, 0.1, 0.5],
            signature=[0.2, 0.3, 0.5],
            meta={},
            uncertainty_score=0.3,
            quality_flags=["short_signature"],
        )
