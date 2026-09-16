"""Generate or verify SHA-256 checksums for files prepared for Git."""

import argparse
import hashlib
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
    names = sorted(set(listing.stdout.decode("utf-8").split("\0")) - {"", manifest.name})
    rows = []
    for name in names:
        with (root / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
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
