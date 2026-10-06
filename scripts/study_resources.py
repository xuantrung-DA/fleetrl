"""Wait for enough host memory and disk space before a study phase."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import psutil


def existing_parent(path: Path) -> Path:
    while not path.exists():
        if path.parent == path:
            raise OSError(f"No existing parent for {path}")
        path = path.parent
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-ram-gb", type=float, default=8.0)
    parser.add_argument("--min-disk-gb", type=float, default=10.0)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if min(args.min_ram_gb, args.min_disk_gb) <= 0 or args.interval < 1:
        parser.error("resource thresholds and interval must be positive")

    disk_path = existing_parent(args.output.resolve())
    while True:
        available_ram_gb = psutil.virtual_memory().available / 2**30
        free_disk_gb = psutil.disk_usage(str(disk_path)).free / 2**30
        print(
            f"[RESOURCES] Available RAM: {available_ram_gb:.1f} GB; "
            f"free disk: {free_disk_gb:.1f} GB",
            flush=True,
        )
        if free_disk_gb < args.min_disk_gb:
            print(
                f"[RESOURCES] Study paused: free disk is below {args.min_disk_gb:.1f} GB.",
                flush=True,
            )
            return 1
        if available_ram_gb >= args.min_ram_gb:
            return 0
        print(
            f"[RESOURCES] Waiting for at least {args.min_ram_gb:.1f} GB available RAM. "
            "Close other memory-heavy programs or press Ctrl+C to stop.",
            flush=True,
        )
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, psutil.Error) as exc:
        print(f"[RESOURCES] Cannot check host resources: {exc}", flush=True)
        raise SystemExit(1) from exc
