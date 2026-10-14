"""Generate or verify SHA-256 checksums for files prepared for Git."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the existing manifest")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = root / "SHA256SUMS.txt"
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    names = set(listing.stdout.decode("utf-8").split("\0")) - {"", manifest.name}
    export_root = root / "runs/publish_bugfix_20261014"
    export_manifest = export_root / "manifest.json"
    if export_manifest.is_file():
        for row in json.loads(export_manifest.read_text(encoding="utf-8")):
            exported = (root / row["export"]).resolve()
            if export_root.resolve() not in exported.parents:
                parser.exit(1, "Export manifest contains a path outside the publication folder.\n")
            names.add(exported.relative_to(root).as_posix())
        for filename in ("README.md", "manifest.json", "PORTABILITY.md", "portability.json"):
            if (export_root / filename).is_file():
                names.add((export_root / filename).relative_to(root).as_posix())
    names = sorted(names)
    rows = []
    for name in names:
        data = (root / name).read_bytes()
        # Git stores text as LF; historical reports explicitly disable text conversion.
        if not name.startswith("reports/") and b"\0" not in data[:8000]:
            data = data.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(data).hexdigest()
        rows.append(f"{digest}  {name}\n")
    expected = "".join(rows)
    if args.check:
        if not manifest.exists() or manifest.read_text(encoding="utf-8") != expected:
            parser.exit(1, "Checksum manifest differs; run scripts/update_checksums.py.\n")
        print(f"Verified {len(rows)} file checksums")
    else:
        manifest.write_text(expected, encoding="utf-8", newline="\n")
        print(f"Wrote {len(rows)} file checksums")


if __name__ == "__main__":
    main()
