"""CUDA kernel compilation — supports both exec() and tempfile+importlib approaches.

The tempfile approach (used by original KernelBench) is needed for Triton/TileLang
decorators that don't work with direct exec().
"""

import tempfile
import importlib
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType


def load_model_via_exec(
    code: str,
    build_directory: str | None = None,
) -> tuple[dict | None, str | None]:
    """Load model by executing code string directly (works for CUDA inline).

    Args:
        code: Python source defining Model/ModelNew.
        build_directory: Optional directory for CUDA build artifacts.

    Returns:
        (namespace_dict, None) on success, (None, error_message) on failure.
    """
    if build_directory:
        os.environ["TORCH_EXTENSIONS_DIR"] = build_directory

    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return namespace, None


def load_model_via_tempfile(
    code: str,
    module_name: str = "kernel_module",
) -> tuple[dict | None, str | None]:
    """Load model by writing to temp file and importing via importlib.

    This approach is required for code using Triton decorators (@triton.jit)
    or other constructs that need proper module context.
    Mirrors KernelBench's load_custom_model_with_tempfile().

    Args:
        code: Python source defining ModelNew.
        module_name: Name for the temporary module.

    Returns:
        (namespace_dict, None) on success, (None, error_message) on failure.
    """
    tmp_dir = tempfile.mkdtemp(prefix="kernel_module_")
    tmp_path = os.path.join(tmp_dir, f"{module_name}.py")

    try:
        with open(tmp_path, "w") as f:
            f.write(code)

        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        if spec is None or spec.loader is None:
            return None, f"Could not create module spec from {tmp_path}"

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        namespace = {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}
        return namespace, None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        # Clean up sys.modules to avoid conflicts between iterations
        sys.modules.pop(module_name, None)


def extract_and_compile(
    candidate_code: str,
    backend: str = "cuda",
    build_directory: str | None = None,
) -> tuple[dict | None, str | None]:
    """Extract ModelNew from candidate code, handling compilation.

    Uses tempfile approach for Triton/TileLang backends,
    exec() approach for CUDA inline.

    Args:
        candidate_code: Full Python source defining ModelNew.
        backend: "cuda", "triton", or "tilelang".
        build_directory: Optional build dir for CUDA extensions.

    Returns:
        (namespace_dict, None) on success, (None, error_message) on failure.
    """
    if backend in ("triton", "tilelang"):
        ns, err = load_model_via_tempfile(candidate_code)
    else:
        build_dir = build_directory or tempfile.mkdtemp(prefix="kernel_build_")
        ns, err = load_model_via_exec(candidate_code, build_directory=build_dir)

    if err:
        return None, err

    if ns is None or "ModelNew" not in ns:
        return None, "ModelNew class not found in candidate code"

    return ns, None
