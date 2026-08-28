"""Рабочая версия critic_L1_lint.

В репозитории она сломана тремя способами сразу: `subprocess` не импортирован,
в вызов ruff подставляется несуществующая переменная `path` вместо `tmp`, и
проверяется `result.timed_out`, которого у CompletedProcess нет. Поэтому L1
возвращал None на каждом кандидате и в веру не входил вообще.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import os
import subprocess
import tempfile

RUFF_RULES = "F821,F811"
RUFF_TIMEOUT_S = 15
_RUFF_CLEAN, _RUFF_VIOLATIONS = 0, 1
RUFF = "/Users/victor/Documents/vs_files/research/.venv/bin/ruff"


def lint_ok(code: str) -> bool | None:
    """True — чисто, False — есть нарушения, None — вердикт не получен."""
    if not code or not code.strip():
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run(
            [RUFF, "check", "--quiet", "--no-cache", "--select", RUFF_RULES, tmp],
            capture_output=True, text=True, timeout=RUFF_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False          # неполученный вердикт не должен выглядеть как чистый
    except FileNotFoundError:
        return None
    finally:
        os.unlink(tmp)
    if r.returncode not in (_RUFF_CLEAN, _RUFF_VIOLATIONS):
        return None
    return r.returncode == _RUFF_CLEAN
