#!/usr/bin/env python3
"""Patch the installed SWE-Bench harness for rootless podman 3.4.4 compatibility.

Why this is needed
==================

The standard SWE-Bench harness (`swebench` PyPI package, tested against v4.1.0)
talks to the Docker daemon via the docker-py SDK. On a host where:

  * Docker is restricted (no docker group membership) so we use rootless podman
    via its docker-compat socket (DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock),
  * podman is v3.4.4 (Ubuntu 22.04 default),

two failure modes appear:

  1. `client.api.build()` against podman 3.4.4 ignores the pull=False kwarg and
     tries to pull the unqualified base image `sweb.base.py.x86_64:latest` from
     docker.io. docker.io has no such image (because SWE-Bench expects the base
     image to live in local podman storage as `localhost/sweb.base.py.x86_64`),
     so the build fails immediately. This breaks every env-image build, which
     in turn breaks every instance build with `--namespace none`.

  2. `client.containers.create(platform=...)` fails with
     `docker.errors.InvalidVersion: platform is not supported for API version
     < 1.41`. podman 3.4.4 reports API v1.40.

This script patches the installed `swebench/harness/docker_build.py` to:

  1. Replace `client.api.build(...)` with a `subprocess.run(["podman", "build",
     "--pull=false", ...])` call. The podman CLI's `--pull=false` is honored
     (unlike the SDK kwarg), so local images are used as the FROM base.

  2. Remove the `platform=test_spec.platform` kwarg from the `containers.create`
     call inside `build_container`. The platform value is still passed through
     to `build_image` (which builds with `--platform=linux/x86_64` via the
     Dockerfile FROM directive), so the container inherits the right arch.

Idempotent — running twice is a no-op.

The complementary script-level change (adding `--namespace none` to the
harness command in `spot_check_generators.run_swebench_eval()`) is in this
repo already.

Usage
-----

    python experiments/orchestration_hypothesis_testing/scripts/patch_swebench_harness.py

Also requires (set in your pipeline shell script before invoking the harness):

    mkdir -p $HOME/buildah-tmp   # or somewhere outside the quota'd partition
    export TMPDIR=$HOME/buildah-tmp
    export BUILDAH_TMPDIR=$HOME/buildah-tmp

Without TMPDIR redirect, buildah layer commits scribble into /var/tmp and may
blow through a tight root-partition disk quota mid-build.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path


PATCH_MARKER_BUILD = "# PATCH(podman-3.4.4): use podman CLI with --pull=false"
PATCH_MARKER_PLATFORM = "# PATCH(podman-3.4.4): platform kwarg disabled"


def locate_docker_build() -> Path:
    """Find docker_build.py inside the installed swebench package."""
    spec = importlib.util.find_spec("swebench")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "swebench not importable in this Python. Activate the env that runs "
            "the harness, then re-run."
        )
    root = Path(spec.origin).parent
    candidate = root / "harness" / "docker_build.py"
    if not candidate.exists():
        raise SystemExit(f"docker_build.py not found at {candidate}")
    return candidate


def patch_build_call(src: str) -> tuple[str, bool]:
    """Replace `response = client.api.build(...)` + its streaming for-loop with
    a `subprocess.run(["podman", "build", "--pull=false", ...])` block.

    Returns (new_src, changed_bool).
    """
    if PATCH_MARKER_BUILD in src:
        return src, False

    lines = src.splitlines(keepends=True)

    # Locate `response = client.api.build(`
    i_start = next(
        (i for i, ln in enumerate(lines) if "response = client.api.build(" in ln),
        None,
    )
    if i_start is None:
        raise SystemExit(
            "Could not locate 'response = client.api.build(' line. The swebench "
            "version may have changed — inspect docker_build.py manually."
        )
    indent = " " * (len(lines[i_start]) - len(lines[i_start].lstrip()))

    # Find matching close paren (bare `)` at same indent)
    i_end = None
    for j in range(i_start + 1, len(lines)):
        if lines[j].strip() == ")":
            i_end = j
            break
    if i_end is None:
        raise SystemExit("Could not find closing ')' for client.api.build call")

    # Find `for chunk in response:` and the end of its block
    i_for = next(
        (j for j in range(i_end + 1, len(lines)) if "for chunk in response:" in lines[j]),
        None,
    )
    if i_for is None:
        raise SystemExit("Could not find 'for chunk in response:' loop")

    # End of for-loop = first non-empty line that dedents below the loop body
    inner_indent = None
    i_for_end = None
    for j in range(i_for + 1, len(lines)):
        s = lines[j]
        if not s.strip():
            continue
        cur = len(s) - len(s.lstrip())
        if inner_indent is None:
            inner_indent = cur
            continue
        if cur < inner_indent:
            i_for_end = j
            break
    if i_for_end is None:
        i_for_end = len(lines)

    replacement = (
        f"{indent}{PATCH_MARKER_BUILD} so unqualified FROM image names\n"
        f"{indent}# (e.g. 'sweb.base.py.x86_64:latest') resolve against the local\n"
        f"{indent}# image store instead of docker.io. The Docker SDK's\n"
        f"{indent}# client.api.build() ignores pull=False on podman 3.4.4.\n"
        f"{indent}import subprocess\n"
        f"{indent}podman_cmd = [\n"
        f'{indent}    "podman", "build",\n'
        f'{indent}    "--pull=false",\n'
        f'{indent}    "-t", image_name,\n'
        f'{indent}    "-f", str(Path(build_dir) / "Dockerfile"),\n'
        f"{indent}    str(build_dir),\n"
        f"{indent}]\n"
        f"{indent}if nocache:\n"
        f'{indent}    podman_cmd.insert(2, "--no-cache")\n'
        f"{indent}logger.info(f\"build cmd: {{' '.join(podman_cmd)}}\")\n"
        f"{indent}proc = subprocess.run(podman_cmd, capture_output=True, text=True)\n"
        f"{indent}buildlog = proc.stdout + proc.stderr\n"
        f"{indent}for line in buildlog.splitlines():\n"
        f"{indent}    logger.info(ansi_escape(line))\n"
        f"{indent}if proc.returncode != 0:\n"
        f'{indent}    raise docker.errors.BuildError(reason=f"podman build exited {{proc.returncode}}", build_log=buildlog)\n'
    )

    new_lines = lines[:i_start] + [replacement] + lines[i_for_end:]
    new_src = "".join(new_lines)

    # Ensure `from pathlib import Path` is imported
    if "from pathlib import Path" not in new_src:
        new_src = new_src.replace(
            "import os\n", "import os\nfrom pathlib import Path\n", 1
        )

    return new_src, True


def patch_platform_kwarg(src: str) -> tuple[str, bool]:
    """Comment out `platform=test_spec.platform,` inside the build_container()
    call to containers.create(). Leave the SAME kwarg inside build_image()
    calls untouched (it's a positional arg there).

    Uses a paren-depth scanner rather than a regex because the call body
    contains nested parens (e.g. `name=test_spec.get_instance_container_name(run_id)`).
    """
    if PATCH_MARKER_PLATFORM in src:
        return src, False

    open_idx = src.find("client.containers.create(")
    if open_idx == -1:
        raise SystemExit(
            "Could not locate 'client.containers.create(' in docker_build.py. "
            "The swebench version may have changed — inspect it manually."
        )
    paren_idx = src.index("(", open_idx)
    depth = 1
    i = paren_idx + 1
    while i < len(src) and depth > 0:
        ch = src[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth != 0:
        raise SystemExit(
            "Could not find matching ')' for client.containers.create(. "
            "The swebench version may have changed — inspect it manually."
        )
    close_idx = i  # one past the matching ')'

    call_span = src[open_idx:close_idx]
    needle = "platform=test_spec.platform,"
    rel = call_span.find(needle)
    if rel == -1:
        raise SystemExit(
            "Could not find 'platform=test_spec.platform,' inside the matched "
            "containers.create(...) call. The swebench version may have changed."
        )

    line_start = src.rfind("\n", 0, open_idx + rel) + 1
    indent = src[line_start:open_idx + rel]
    if indent.strip():
        raise SystemExit(
            "Unexpected formatting: 'platform=...' is not at the start of its "
            "own indent block — inspect docker_build.py manually."
        )

    replacement = (
        f"{indent}{PATCH_MARKER_PLATFORM}\n"
        f"{indent}# platform=test_spec.platform,"
    )
    line_end = src.index("\n", open_idx + rel)
    new_src = src[:line_start] + replacement + src[line_end:]
    return new_src, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing the file.",
    )
    args = parser.parse_args()

    target = locate_docker_build()
    print(f"target: {target}")

    src = target.read_text()
    original = src
    n_changes = 0

    src, c1 = patch_build_call(src)
    if c1:
        n_changes += 1
        print("  applied: podman CLI build (patch 1/2)")
    else:
        print("  skipped: podman CLI build patch already present")

    src, c2 = patch_platform_kwarg(src)
    if c2:
        n_changes += 1
        print("  applied: platform kwarg disabled in containers.create (patch 2/2)")
    else:
        print("  skipped: platform kwarg patch already present")

    if n_changes == 0:
        print("Already patched. Nothing to do.")
        return 0

    # Syntax check before writing
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"ERROR: patched source has SyntaxError: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"--dry-run: would write {len(src)} chars to {target}")
        return 0

    backup = target.with_suffix(".py.preharnesspatch")
    if not backup.exists():
        backup.write_text(original)
        print(f"backup: {backup}")
    target.write_text(src)
    print(f"wrote: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
