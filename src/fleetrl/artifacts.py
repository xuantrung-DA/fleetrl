"""Atomic local artifact replacement with bounded Windows sharing-lock retries."""

import time
from pathlib import Path


def atomic_replace(source, target):
    source = Path(source)
    target = Path(target)
    for attempt in range(10):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            # Antivirus/indexers may briefly hold a just-closed ZIP on Windows.
            # Preserve the previous checkpoint and never bypass permissions.
            time.sleep(min(1.0, 0.1 * 2**attempt))
