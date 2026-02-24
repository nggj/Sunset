"""Command-line entrypoint for skycolor_locator."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime

from skycolor_locator.orchestrate.earthstate_ingest import IngestConfig, ingest_earthstate
from skycolor_locator.time.bucketing import to_utc


def _parse_iso_datetime(value: str) -> datetime:
    """Parse ISO datetime string and normalize to aware UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid datetime: {value}") from exc
    return to_utc(parsed)


def build_parser() -> argparse.ArgumentParser:
    """Create and return the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="skycolor_locator",
        description="Skycolor Locator command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command")
    ingest = subparsers.add_parser(
        "ingest-earthstate",
        help="Ingest EarthState(time_bucket, tile) records into local SQLite store.",
    )
    ingest.add_argument("--store", required=True)
    ingest.add_argument("--provider", choices=["mock", "gee"], default="mock")
    ingest.add_argument("--lat-min", type=float, required=True)
    ingest.add_argument("--lat-max", type=float, required=True)
    ingest.add_argument("--lon-min", type=float, required=True)
    ingest.add_argument("--lon-max", type=float, required=True)
    ingest.add_argument("--tile-step-deg", type=float, required=True)
    ingest.add_argument("--start-utc", type=_parse_iso_datetime, required=True)
    ingest.add_argument("--end-utc", type=_parse_iso_datetime, required=True)
    ingest.add_argument("--bucket-minutes", type=int, required=True)
    ingest.add_argument("--max-tiles", type=int, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI application."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-earthstate":
        cfg = IngestConfig(
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            tile_step_deg=args.tile_step_deg,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            bucket_minutes=args.bucket_minutes,
            provider_mode=args.provider,
            store_path=args.store,
            max_tiles=args.max_tiles,
        )
        stats = ingest_earthstate(cfg)
        print(
            "ingest-earthstate complete "
            f"tiles_written={stats['tiles_written']} "
            f"buckets_written={stats['buckets_written']} "
            f"missing_realtime_count={stats['missing_realtime_count']}"
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
