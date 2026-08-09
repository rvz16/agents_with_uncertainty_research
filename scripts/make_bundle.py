#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "runs",
    "build",
    "dist",
}


def should_include(path: Path, root: Path, output: Path) -> bool:
    relative = path.relative_to(root)
    return (
        not any(part in EXCLUDED_PARTS for part in relative.parts)
        and path != output
        and path.suffix not in {".pyc", ".pyo"}
        and not path.name.endswith(".tar.gz")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source-only toolkit archive")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and should_include(path.resolve(), root.resolve(), output):
                archive.add(path, arcname=Path(root.name) / path.relative_to(root))
    print(f"created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
