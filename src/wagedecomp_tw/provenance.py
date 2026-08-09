from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_results_manifest(results_dir: Path) -> Path:
    files = []
    for path in sorted(results_dir.rglob("*")):
        if path.is_file() and path.name != "results_manifest.json":
            files.append(
                {
                    "path": path.relative_to(results_dir.parent).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    output = results_dir / "results_manifest.json"
    output.write_text(
        json.dumps({"files": files}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output

