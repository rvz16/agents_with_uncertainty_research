"""SWE-bench Lite harness via Docker exec.

Uses Princeton's pre-built images (pulled on demand) and runs critics
inside a long-lived container per instance. No LLM — Y=1 is produced by
applying the dataset's gold `patch`; Y=0 is the base_commit source unchanged.
`test_patch` (new/modified tests) is always applied, because the dataset's
FAIL_TO_PASS / PASS_TO_PASS test IDs live there.

Critics:
    critic_syntax  — ast.parse succeeds on every file touched by `patch`
    critic_lint    — pyflakes clean on every file touched by `patch`
    critic_early   — first FAIL_TO_PASS test passes
    critic_mid     — all FAIL_TO_PASS tests pass

`run_full_test` runs FAIL_TO_PASS ∪ PASS_TO_PASS — the dataset's own oracle.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Instance pool — curated small-dep instances across small repos to minimise
# image pull size (~2-3 GB per image) and test wall-clock time.
# ---------------------------------------------------------------------------

SWE_INSTANCE_POOL: list[str] = [
    # seaborn (small dep graph, ~3 GB images)
    "mwaskom__seaborn-3010",
    "mwaskom__seaborn-2848",
    "mwaskom__seaborn-3190",
    # requests (tiny, ~2 GB)
    "psf__requests-1963",
    "psf__requests-2317",
    "psf__requests-2674",
    "psf__requests-863",
    # flask (small, ~2 GB)
    "pallets__flask-5063",
    # pytest self-hosting (~2 GB)
    "pytest-dev__pytest-7168",
    "pytest-dev__pytest-5227",
    "pytest-dev__pytest-5692",
]

# Timeouts
EXEC_TIMEOUT_CRITIC = 180    # per critic / oracle; Docker emulation slows things
EXEC_TIMEOUT_SYNTAX = 30
CONTAINER_LIFETIME = 7200    # 2 hr; covers 5 variants under amd64 emulation


# ---------------------------------------------------------------------------
# Dataset — cached
# ---------------------------------------------------------------------------

# SWE-Bench dataset selection. Two upstream variants are supported:
#   - "SWE-bench_Lite"     (300 instances, the original lightweight subset)
#   - "SWE-bench_Verified" (~500 instances, the hand-verified set)
# Pick via env var SWE_BENCH_DATASET (default: Verified). The runner CLI
# (run_swebench_full.py --dataset {lite,verified}) sets this for the
# duration of the run.
_DATASET_NAME_MAP = {
    "lite":     "SWE-bench_Lite",
    "verified": "SWE-bench_Verified",
}


def _dataset_id(name: str | None = None) -> str:
    if name is None:
        name = os.environ.get("SWE_BENCH_DATASET", "verified")
    name = name.strip().lower()
    if name not in _DATASET_NAME_MAP:
        raise ValueError(
            f"Unknown SWE_BENCH_DATASET={name!r}; expected one of "
            f"{sorted(_DATASET_NAME_MAP)}"
        )
    return f"princeton-nlp/{_DATASET_NAME_MAP[name]}"


@lru_cache(maxsize=2)
def _dataset_for(dataset_id: str) -> list[dict]:
    ds = load_dataset(dataset_id, split="test")
    return list(ds)


def _dataset() -> list[dict]:
    return _dataset_for(_dataset_id())


@lru_cache(maxsize=2)
def _by_id_for(dataset_id: str) -> dict[str, dict]:
    return {ex["instance_id"]: ex for ex in _dataset_for(dataset_id)}


def _by_id() -> dict[str, dict]:
    return _by_id_for(_dataset_id())


def list_instance_ids() -> list[str]:
    """Instance IDs from SWE_INSTANCE_POOL that actually exist in the dataset."""
    present = set(_by_id().keys())
    return [i for i in SWE_INSTANCE_POOL if i in present]


def get_instance(instance_id: str) -> dict:
    return _by_id()[instance_id]


def _json_or_list(x) -> list[str]:
    return json.loads(x) if isinstance(x, str) else list(x)


def get_ftp(instance_id: str) -> list[str]:
    return _json_or_list(get_instance(instance_id)["FAIL_TO_PASS"])


def get_ptp(instance_id: str) -> list[str]:
    return _json_or_list(get_instance(instance_id)["PASS_TO_PASS"])


_FILE_RE = re.compile(r"^(?:\+\+\+|---) b?/(?P<path>[^\s]+)", re.MULTILINE)


def changed_files_from_patch(patch_text: str) -> list[str]:
    """Unique .py paths touched by a unified diff (drop /dev/null)."""
    paths = set()
    for m in _FILE_RE.finditer(patch_text):
        p = m.group("path")
        if p == "/dev/null":
            continue
        if not p.endswith(".py"):
            continue
        paths.add(p)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Docker primitives
# ---------------------------------------------------------------------------

_IMAGE_BY_INSTANCE: dict[str, str] = {}


def _arch() -> str:
    return os.environ.get("SWEBENCH_ARCH", "x86_64")


def _remote_image(instance_id: str) -> str:
    tag = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.{_arch()}.{tag}:latest"


def _local_image(instance_id: str) -> str:
    return f"sweb.eval.{_arch()}.{instance_id.lower()}:latest"


def _image(instance_id: str) -> str:
    return _IMAGE_BY_INSTANCE.get(instance_id, _remote_image(instance_id))


def _docker_platform() -> str:
    return os.environ.get("SWEBENCH_DOCKER_PLATFORM", "linux/amd64")


def _container_name(instance_id: str) -> str:
    return f"swe_abbo_{instance_id.replace('__', '_')}"


def _build_missing_images() -> bool:
    return os.environ.get("SWEBENCH_BUILD_MISSING_IMAGES", "1").lower() not in {
        "0", "false", "no", "off",
    }


def _patch_local_arm64_test_spec(spec) -> None:
    """Small compatibility fixes for locally built SWE images on aarch64."""
    if spec.arch != "arm64":
        return

    if spec.repo in {"scikit-learn/scikit-learn", "pydata/xarray"}:
        apt = (
            "apt-get update && apt-get install -y "
            "gfortran libopenblas-dev liblapack-dev pkg-config "
            "&& rm -rf /var/lib/apt/lists/*"
        )
        if apt not in spec.env_script_list:
            spec.env_script_list.insert(1, apt)

    if spec.repo == "pydata/xarray":
        for i, cmd in enumerate(spec.env_script_list):
            if cmd == "conda env update -f environment.yml":
                drop_cdms2 = "sed -i '/^[[:space:]]*- cdms2$/d' environment.yml"
                if drop_cdms2 not in spec.env_script_list:
                    spec.env_script_list.insert(i, drop_cdms2)
                break

    if spec.repo == "scikit-learn/scikit-learn":
        pip_pin = "python -m pip install 'pip<23.1'"
        if pip_pin not in spec.env_script_list:
            spec.env_script_list.append(pip_pin)

    if spec.repo == "sympy/sympy":
        reset_commit = None
        for cmd in spec.repo_script_list:
            m = re.match(r"git reset --hard ([0-9a-f]+)$", cmd)
            if m:
                reset_commit = m.group(1)
                break
        for i, cmd in enumerate(spec.repo_script_list):
            if cmd.startswith("git clone -o origin --branch ") and "https://github.com/sympy/sympy /testbed" in cmd:
                fallback = "git clone -o origin https://github.com/sympy/sympy /testbed"
                if reset_commit:
                    fallback = (
                        "git init /testbed && cd /testbed && "
                        "git remote add origin https://github.com/sympy/sympy && "
                        f"git fetch --depth 1 origin {reset_commit} && "
                        "git checkout FETCH_HEAD"
                    )
                spec.repo_script_list[i] = (
                    f"{cmd} || "
                    f"(rm -rf /testbed && {fallback})"
                )
                break

    for i, cmd in enumerate(spec.env_script_list):
        if "conda create -n testbed python=3.5" in cmd:
            spec.env_script_list[i] = cmd.replace("python=3.5", "python=3.6")

    if any("lxml" in cmd or "pikepdf" in cmd for cmd in spec.env_script_list):
        apt = (
            "apt-get update && apt-get install -y "
            "libxml2-dev libxslt1-dev zlib1g-dev "
            "&& rm -rf /var/lib/apt/lists/*"
        )
        if apt not in spec.env_script_list:
            spec.env_script_list.insert(1, apt)

    if spec.repo == "astropy/astropy":
        for i, cmd in enumerate(spec.env_script_list):
            if "conda create -n testbed python=3.6 setuptools==38.2.4 -y" in cmd:
                spec.env_script_list[i] = cmd.replace(" setuptools==38.2.4", "")
                setuptools_pin = "python -m pip install 'setuptools==38.2.4'"
                if setuptools_pin not in spec.env_script_list:
                    spec.env_script_list.insert(i + 2, setuptools_pin)
                break
        for i, cmd in enumerate(spec.env_script_list):
            if cmd.startswith("python -m pip install ") and "--no-clean" not in cmd:
                spec.env_script_list[i] = cmd.replace(
                    "python -m pip install ",
                    "python -m pip install --no-clean ",
                    1,
                )
        for i, cmd in enumerate(spec.repo_script_list):
            if cmd == "python -m pip install -e .[test] --verbose":
                spec.repo_script_list[i:i + 1] = [
                    "python -m pip install jinja2",
                    "python -m pip install 'Cython<3'",
                    "python -m pip install extension-helpers",
                    "python -m pip install --no-build-isolation -e .[test] --verbose",
                ]
                break

    if spec.repo == "matplotlib/matplotlib":
        for i, cmd in enumerate(spec.repo_script_list):
            if cmd.startswith("tar -xvzf ") and "--no-same-owner" not in cmd:
                spec.repo_script_list[i] = cmd.replace(
                    "tar -xvzf ",
                    "tar --no-same-owner -xvzf ",
                    1,
                )


def _looks_like_missing_remote_image(stderr: str) -> bool:
    s = stderr.lower()
    return any(
        marker in s
        for marker in (
            "requested access to the resource is denied",
            "manifest unknown",
            "not found",
            "pull access denied",
            "repository does not exist",
            "toomanyrequests",
            "too many requests",
            "pull rate limit",
        )
    )


def _looks_like_rootless_storage_flake(exc: BaseException | str | None) -> bool:
    s = str(exc or "").lower()
    return (
        ("lgetxattr" in s or "lsetxattr" in s or "containers-storage" in s)
        and ("permission denied" in s or "operation not supported" in s)
    )


def _cleanup_after_rootless_storage_flake(image_key: str, verbose: bool = True) -> None:
    """Clean partial Podman build state after rootless overlay metadata flakes."""
    if verbose:
        print("  rootless Podman storage flake; cleaning partial build state before retry", flush=True)
    _sh(["docker", "container", "prune", "-f"], timeout=60)
    _sh(["docker", "image", "rm", "-f", image_key], timeout=60)


# Harness stability knobs. Tuned so transient Docker/containerd hiccups
# (image-pull stalls, containerd metadata races during concurrent runs,
# Hub rate limits) get retried instead of marked as instance failures.
PULL_TIMEOUT_S        = 3600   # was 1800. Sympy/Django images can exceed 30 min
                               # on cold cache + slow link.
START_TIMEOUT_S       = 120    # was 60. Startup is bursty under concurrent runs.
PULL_RETRY_ATTEMPTS   = 3      # 1 attempt + 2 retries
START_RETRY_ATTEMPTS  = 3
RETRY_BACKOFF_S       = (10, 30)  # backoff sequence between attempts


def _sh(cmd: list[str] | str, timeout: int = 600, check: bool = False):
    """Run a shell command. Converts subprocess.TimeoutExpired into a fake
    completed result with returncode=-1 so callers can treat it the same way
    as a non-zero exit (retryable). Previously, a TimeoutExpired bubbled all
    the way up and aborted the whole sweep on one slow image pull."""
    try:
        return subprocess.run(
            cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
            timeout=timeout, check=check,
        )
    except subprocess.TimeoutExpired as e:
        # Synthesise a CompletedProcess-shaped object that returncode-based
        # logic can recognise as failure.
        return subprocess.CompletedProcess(
            args=cmd, returncode=-1,
            stdout=(e.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or ""),
            stderr=f"timeout after {timeout}s: " + (
                (e.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")
            ),
        )


def _retry(
    attempt_fn,
    attempts: int,
    backoff_s: tuple[int, ...] = RETRY_BACKOFF_S,
    label: str = "op",
    verbose: bool = True,
):
    """Run attempt_fn() up to `attempts` times. attempt_fn returns the
    subprocess.CompletedProcess; we retry while returncode != 0. Returns the
    final result (success or last failure). Sleeps backoff_s[i-1] between
    attempts i and i+1 (clamped to the last entry once we run out)."""
    last = None
    for i in range(attempts):
        last = attempt_fn()
        if last.returncode == 0:
            return last
        if i + 1 < attempts:
            wait = backoff_s[min(i, len(backoff_s) - 1)]
            if verbose:
                tail = (last.stderr or "")[-200:].replace("\n", " | ")
                print(f"  {label} attempt {i+1}/{attempts} failed (rc={last.returncode}): {tail}",
                      flush=True)
                print(f"  retrying in {wait}s ...", flush=True)
            time.sleep(wait)
    return last


def _build_local_image(instance_id: str, verbose: bool = True) -> str:
    """Build a namespace-less SWE-bench image for the local architecture."""
    import docker
    import swebench.harness.docker_build as docker_build
    from swebench.harness.test_spec.test_spec import make_test_spec

    build_root = Path(
        os.environ.get(
            "SWEBENCH_BUILD_DIR",
            "/capstor/store/cscs/swissai/a0142/agents_uq/swebench_build_images",
        )
    )
    build_max_workers = int(os.environ.get("SWEBENCH_BUILD_MAX_WORKERS", "1"))
    docker_build.BASE_IMAGE_BUILD_DIR = build_root / "base"
    docker_build.ENV_IMAGE_BUILD_DIR = build_root / "env"
    docker_build.INSTANCE_IMAGE_BUILD_DIR = build_root / "instances"
    lock_dir = build_root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    lock_path = lock_dir / "swebench_local_build.lock"
    with open(lock_path, "w") as lock_file:
        if verbose:
            print(f"  waiting for SWE image build lock {lock_path} ...", flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        local_img = _local_image(instance_id)
        r = _sh(["docker", "image", "inspect", local_img], timeout=30)
        if r.returncode == 0:
            return local_img
        if verbose:
            print(f"  building local SWE image {local_img} ...", flush=True)
        client = docker.from_env()
        spec = make_test_spec(
            get_instance(instance_id),
            namespace=None,
            instance_image_tag="latest",
            env_image_tag="latest",
            arch=_arch(),
        )
        _patch_local_arm64_test_spec(spec)
        last_failed = []
        env_build_attempts = max(int(os.environ.get("SWEBENCH_ENV_BUILD_ATTEMPTS", "4")), 1)
        force_rebuilds = [(attempt % 2) == 0 for attempt in range(1, env_build_attempts + 1)]
        last_env_exc = None
        for attempt, force_rebuild in enumerate(force_rebuilds, start=1):
            last_env_exc = None
            try:
                _, last_failed = docker_build.build_env_images(
                    client,
                    [spec],
                    force_rebuild=force_rebuild,
                    max_workers=build_max_workers,
                    namespace=None,
                    instance_image_tag="latest",
                    env_image_tag="latest",
                )
            except Exception as exc:
                last_env_exc = exc
                last_failed = [spec.env_image_key]
                if _looks_like_rootless_storage_flake(exc):
                    _cleanup_after_rootless_storage_flake(spec.env_image_key, verbose=verbose)
            env_exists = _sh(["docker", "image", "inspect", spec.env_image_key], timeout=30)
            if env_exists.returncode == 0 and not last_failed:
                last_env_exc = None
                break
            if env_exists.returncode == 0 and last_env_exc is not None:
                last_failed = []
                last_env_exc = None
                break
            if not last_failed:
                last_failed = [spec.env_image_key]
            if attempt < len(force_rebuilds) and verbose:
                next_force = force_rebuilds[attempt]
                print(f"  SWE env image build failed/missing; retrying with force_rebuild={next_force}", flush=True)
        if last_failed:
            if last_env_exc is not None:
                raise RuntimeError(f"SWE env image build failed for {instance_id}: {last_failed}") from last_env_exc
            raise RuntimeError(f"SWE env image build failed for {instance_id}: {last_failed}")

        instance_build_attempts = int(os.environ.get("SWEBENCH_INSTANCE_BUILD_ATTEMPTS", "2"))
        last_exc = None
        for attempt in range(1, instance_build_attempts + 1):
            try:
                docker_build.build_instance_image(spec, client, None, nocache=False)
            except Exception as exc:
                last_exc = exc
                if _looks_like_rootless_storage_flake(exc):
                    _cleanup_after_rootless_storage_flake(spec.instance_image_key, verbose=verbose)
                if attempt < instance_build_attempts and verbose:
                    print("  SWE instance image build failed; retrying", flush=True)
                continue
            img_exists = _sh(["docker", "image", "inspect", spec.instance_image_key], timeout=30)
            if img_exists.returncode == 0:
                return spec.instance_image_key
            if attempt < instance_build_attempts and verbose:
                print("  SWE instance image missing after build; retrying", flush=True)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"SWE instance image build failed for {instance_id}: {spec.instance_image_key}")


def pull_image(instance_id: str, verbose: bool = True) -> None:
    local_img = _local_image(instance_id)
    r = _sh(["docker", "image", "inspect", local_img], timeout=30)
    if r.returncode == 0:
        _IMAGE_BY_INSTANCE[instance_id] = local_img
        return

    remote_img = _remote_image(instance_id)
    r = _sh(["docker", "image", "inspect", remote_img], timeout=30)
    if r.returncode == 0:
        _IMAGE_BY_INSTANCE[instance_id] = remote_img
        return
    if verbose:
        print(f"  pulling {remote_img} ...", flush=True)

    last = None
    for i in range(PULL_RETRY_ATTEMPTS):
        last = _sh(
            ["docker", "pull", "--platform", _docker_platform(), remote_img],
            timeout=PULL_TIMEOUT_S,
        )
        if last.returncode == 0:
            _IMAGE_BY_INSTANCE[instance_id] = remote_img
            return
        if _build_missing_images() and _looks_like_missing_remote_image(last.stderr or ""):
            if verbose:
                print("  remote SWE image unavailable; falling back to local build", flush=True)
            _IMAGE_BY_INSTANCE[instance_id] = _build_local_image(instance_id, verbose=verbose)
            return
        if i + 1 < PULL_RETRY_ATTEMPTS:
            wait = RETRY_BACKOFF_S[min(i, len(RETRY_BACKOFF_S) - 1)]
            if verbose:
                tail = (last.stderr or "")[-200:].replace("\n", " | ")
                print(f"  pull {remote_img} attempt {i+1}/{PULL_RETRY_ATTEMPTS} failed (rc={last.returncode}): {tail}",
                      flush=True)
                print(f"  retrying in {wait}s ...", flush=True)
            time.sleep(wait)
    if last is None or last.returncode != 0:
        raise RuntimeError(f"pull failed after {PULL_RETRY_ATTEMPTS} attempts: "
                           f"{((last.stderr if last else '') or '')[-400:]}")


def start_container(instance_id: str, verbose: bool = True) -> str:
    cname = _container_name(instance_id)
    _sh(["docker", "rm", "-f", cname], timeout=30)
    r = _retry(
        lambda: _sh(
            [
                "docker", "run", "-d", "--platform", _docker_platform(),
                "--name", cname,
                "--entrypoint", "/bin/bash",
                _image(instance_id),
                "-c", f"sleep {CONTAINER_LIFETIME}",
            ],
            timeout=START_TIMEOUT_S,
        ),
        attempts=START_RETRY_ATTEMPTS,
        label=f"start {cname}",
        verbose=verbose,
    )
    if r.returncode != 0:
        raise RuntimeError(f"start failed after {START_RETRY_ATTEMPTS} attempts: "
                           f"{(r.stderr or '')[-400:]}")
    return cname


def stop_container(cname: str) -> None:
    _sh(["docker", "rm", "-f", cname], timeout=30)


def _exec(cname: str, inner_cmd: str, timeout: int = EXEC_TIMEOUT_CRITIC):
    return subprocess.run(
        ["docker", "exec", cname, "bash", "-lc", inner_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _exec_stdin(cname: str, inner_cmd: str, stdin: str, timeout: int = 60):
    return subprocess.run(
        ["docker", "exec", "-i", cname, "bash", "-lc", inner_cmd],
        input=stdin, capture_output=True, text=True, timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Repo preparation
# ---------------------------------------------------------------------------

def prepare_repo(
    cname: str, instance: dict, apply_fix: bool,
) -> None:
    """Reset /testbed to base_commit, apply test_patch, optionally apply fix."""
    # Hard reset and drop any leftover changes
    base = instance["base_commit"]
    _exec(cname, f"cd /testbed && git reset --hard {base} && git clean -fdx >/dev/null 2>&1", timeout=60)
    # Apply test_patch (mandatory — defines the FAIL_TO_PASS / PASS_TO_PASS tests)
    r = _exec_stdin(cname, "cat > /tmp/test.patch && cd /testbed && git apply /tmp/test.patch",
                    instance["test_patch"], timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"test_patch apply failed: {r.stderr[-400:]}")
    if apply_fix:
        r = _exec_stdin(cname, "cat > /tmp/fix.patch && cd /testbed && git apply /tmp/fix.patch",
                        instance["patch"], timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"gold patch apply failed: {r.stderr[-400:]}")


# ---------------------------------------------------------------------------
# Critics + oracle
# ---------------------------------------------------------------------------

PYTEST_ENV_PREFIX = (
    "source /opt/miniconda3/bin/activate && conda activate testbed && cd /testbed && "
)


def _run_pytest_subset(cname: str, tests: list[str], timeout: int = EXEC_TIMEOUT_CRITIC) -> bool:
    if not tests:
        return True
    args = " ".join(shlex.quote(t) for t in tests)
    cmd = PYTEST_ENV_PREFIX + f"python -m pytest --no-header -rA -q {args}"
    r = _exec(cname, cmd, timeout=timeout)
    return r.returncode == 0


def run_critic_syntax(cname: str, instance: dict) -> tuple[bool, str]:
    files = changed_files_from_patch(instance["patch"])
    if not files:
        return True, "no py files"
    listing = " ".join(shlex.quote(f) for f in files)
    cmd = f"cd /testbed && for f in {listing}; do /opt/miniconda3/envs/testbed/bin/python -c \"import ast; ast.parse(open('$f').read())\" || exit 1; done"
    r = _exec(cname, cmd, timeout=EXEC_TIMEOUT_SYNTAX)
    return r.returncode == 0, r.stderr.strip()[:200]


def run_critic_lint(cname: str, instance: dict) -> tuple[bool, str]:
    files = changed_files_from_patch(instance["patch"])
    if not files:
        return True, "no py files"
    listing = " ".join(shlex.quote(f) for f in files)
    cmd = PYTEST_ENV_PREFIX + f"python -m pyflakes {listing}"
    r = _exec(cname, cmd, timeout=EXEC_TIMEOUT_SYNTAX)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[:200]


def run_critic_early(cname: str, instance: dict) -> tuple[bool, str]:
    ftp = get_ftp(instance["instance_id"])
    first = ftp[:1]
    ok = _run_pytest_subset(cname, first)
    return ok, f"first FTP={first[0][:60] if first else 'none'} ok={ok}"


def run_critic_mid(cname: str, instance: dict) -> tuple[bool, str]:
    ftp = get_ftp(instance["instance_id"])
    ok = _run_pytest_subset(cname, ftp)
    return ok, f"all FTP (n={len(ftp)}) ok={ok}"


def run_full_test(cname: str, instance: dict, max_ptp: int = 30,
                  timeout: int = 1200) -> tuple[bool, str]:
    """Verify Y. FAIL_TO_PASS must all pass; PASS_TO_PASS subset must not regress.

    Caps PASS_TO_PASS to first `max_ptp` tests for speed: psf/requests has
    ~140 PTP tests that easily exceed a 5-minute Docker exec. Conservative
    Y proxy: still requires all FTP + first 30 PTP to pass.
    """
    ftp = get_ftp(instance["instance_id"])
    ptp = get_ptp(instance["instance_id"])[:max_ptp]
    tests = ftp + ptp
    ok = _run_pytest_subset(cname, tests, timeout=timeout)
    return ok, f"full (FTP={len(ftp)}, PTP_subset={len(ptp)}) ok={ok}"


# ---------------------------------------------------------------------------
# Critic registry + hand-tuned likelihoods
# ---------------------------------------------------------------------------

SWE_CRITIC_NAMES: list[str] = [
    "critic_syntax",
    "critic_lint",
    "critic_early",
    "critic_mid",
]


def run_critic(cname: str, critic_name: str, instance: dict) -> tuple[bool, str]:
    if critic_name == "critic_syntax":
        return run_critic_syntax(cname, instance)
    if critic_name == "critic_lint":
        return run_critic_lint(cname, instance)
    if critic_name == "critic_early":
        return run_critic_early(cname, instance)
    if critic_name == "critic_mid":
        return run_critic_mid(cname, instance)
    raise ValueError(f"unknown critic: {critic_name}")


# Hand-tuned priors. Real SWE-bench patches are small, well-written fixes
# of real bugs, so static critics should pass for both Y=1 and Y=0 most of
# the time. critic_early and critic_mid are the semantic signal.
SWE_CRITIC_LIKELIHOODS: dict[str, dict[str, float]] = {
    "critic_syntax": {"p_pass_y1": 0.99, "p_pass_y0": 0.98},
    "critic_lint":   {"p_pass_y1": 0.85, "p_pass_y0": 0.80},
    "critic_early":  {"p_pass_y1": 0.97, "p_pass_y0": 0.08},
    "critic_mid":    {"p_pass_y1": 0.96, "p_pass_y0": 0.05},
}


# ---------------------------------------------------------------------------
# Calibration collection
# ---------------------------------------------------------------------------

@dataclass
class SWEPerInstanceTiming:
    instance_id: str
    pull_s: float
    y0_s: float
    y1_s: float


def collect_calibration_samples_from_pairs(
    instance_ids: list[str],
    verbose: bool = True,
    keep_containers: bool = False,
):
    """For each instance: pull image, start container, produce one Y=1 sample
    (gold patch applied) and one Y=0 sample (base_commit unchanged), running
    all critics on each. Returns (samples, kept_ids, timings).
    """
    from abbo.realworld.agents.calibration import CalibrationSample

    samples = []
    kept = []
    timings: list[SWEPerInstanceTiming] = []

    for i, iid in enumerate(instance_ids, 1):
        t_all = time.perf_counter()
        try:
            t_pull = time.perf_counter()
            pull_image(iid, verbose=verbose)
            pull_s = time.perf_counter() - t_pull

            cname = start_container(iid)
            inst = get_instance(iid)

            side_timings = {}
            try:
                for apply_fix, y, arm in ((False, False, "no_fix"),
                                          (True, True, "gold_fix")):
                    t_side = time.perf_counter()
                    prepare_repo(cname, inst, apply_fix=apply_fix)
                    for critic in SWE_CRITIC_NAMES:
                        passed, detail = run_critic(cname, critic, inst)
                        samples.append(CalibrationSample(
                            bug_id=iid,
                            arm=arm,
                            critic=critic,
                            critic_passed=passed,
                            patch_correct=y,
                        ))
                        if verbose:
                            s = "PASS" if passed else "FAIL"
                            print(
                                f"    [{iid}] arm={arm:8s} {critic:14s} {s}  {detail[:80]}",
                                flush=True,
                            )
                    side_timings[arm] = time.perf_counter() - t_side
                kept.append(iid)
            finally:
                if not keep_containers:
                    stop_container(cname)

            timings.append(SWEPerInstanceTiming(
                instance_id=iid, pull_s=pull_s,
                y0_s=side_timings.get("no_fix", 0.0),
                y1_s=side_timings.get("gold_fix", 0.0),
            ))
            if verbose:
                print(
                    f"  [{i}/{len(instance_ids)}] {iid} done in "
                    f"{time.perf_counter() - t_all:.1f}s  "
                    f"(pull={pull_s:.1f}s, Y=0={side_timings.get('no_fix', 0):.1f}s, "
                    f"Y=1={side_timings.get('gold_fix', 0):.1f}s)",
                    flush=True,
                )
        except Exception as e:
            if verbose:
                print(f"  [skip] {iid}: {type(e).__name__}: {e}", flush=True)
            continue

    return samples, kept, timings
