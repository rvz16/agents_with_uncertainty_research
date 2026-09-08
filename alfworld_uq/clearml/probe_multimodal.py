#!/usr/bin/env python3
"""Can we run ALFWorld with pictures at all? Two blockers, one task.

Our runs use AlfredTWEnv, whose observations are strings; frames do not exist
there, so images are not switched off but absent. The visual version needs
AlfredThorEnv, i.e. the ai2thor simulator: a Unity build that renders on the
GPU and normally expects an X server. Whether that works headless in our
container has never been tested.

The model side turned out to need nothing new -- Qwen3.6-35B-A3B, the
checkpoint we already serve, is image-text-to-text and its chat template
carries <|vision_start|> -- but "accepts images" and "our vLLM serves images
and still returns log-probabilities" are different claims, and the second is
what the experiment depends on.

This probe answers both and stops. It runs nothing of the experiment itself.
"""

import base64
import io
import os
import subprocess
import sys

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def verdict(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAILED"
    print(f"[probe] VERDICT {name}: {mark} {detail}".rstrip(), flush=True)
    return ok


def probe_thor() -> bool:
    """A Unity build that renders without a display, or nothing else matters."""
    print("[probe] installing ai2thor", flush=True)
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", "ai2thor"])
    if rc != 0:
        return verdict("ai2thor install", False, f"pip rc={rc}")
    try:
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering
    except Exception as exc:  # noqa: BLE001
        return verdict("ai2thor import", False, f"{type(exc).__name__}: {exc}")

    try:
        controller = Controller(platform=CloudRendering, width=300, height=300)
        event = controller.step(action="Pass")
        frame = getattr(event, "frame", None)
        controller.stop()
    except Exception as exc:  # noqa: BLE001
        return verdict("ai2thor headless render", False, f"{type(exc).__name__}: {str(exc)[:200]}")
    if frame is None:
        return verdict("ai2thor headless render", False, "no frame on the event")
    return verdict("ai2thor headless render", True, f"frame {frame.shape}")


def probe_alfworld_thor() -> bool:
    """The ALFWorld layer on top: same games, rendered instead of described."""
    try:
        from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return verdict("AlfredThorEnv import", False, f"{type(exc).__name__}: {exc}")
    return verdict("AlfredThorEnv import", True, "the visual env class is available")


def _red_square_png() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def probe_vision_endpoint(base_url: str, model: str) -> bool:
    """Does our own server take an image and still return log-probabilities?

    The UQ work needs both at once: an image-capable endpoint that drops
    logprobs would be useless to us, and that is exactly the kind of thing
    that only shows up when asked.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="local", timeout=120.0, max_retries=0)
    content = [
        {"type": "text", "text": "Name the dominant colour in one word."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_red_square_png()}"},
        },
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=16,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )
    except Exception as exc:  # noqa: BLE001
        return verdict("image request", False, f"{type(exc).__name__}: {str(exc)[:220]}")

    text = (response.choices[0].message.content or "").strip()
    logprobs = getattr(response.choices[0], "logprobs", None)
    tokens = getattr(logprobs, "content", None) or []
    has_top = bool(tokens and getattr(tokens[0], "top_logprobs", None))
    verdict("image request", True, f"answer={text!r}")
    verdict("logprobs with an image", bool(tokens), f"{len(tokens)} tokens")
    return verdict("top_logprobs with an image", has_top, "needed for token entropy")


def main() -> int:
    task = Task.current_task() or Task.init(
        project_name="agentic-uq", task_name="alfworld multimodal probe"
    )
    task.output_uri = FILE_STORE
    params = task.get_parameters_as_dict().get("Args", {}) or {}
    for key, value in params.items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    base_url = os.environ.get("LLM_BASE_URI", "")
    model = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")

    results = {
        "ai2thor": probe_thor(),
        "alfworld_thor": probe_alfworld_thor(),
    }
    if base_url:
        results["vision_endpoint"] = probe_vision_endpoint(base_url, model)
    else:
        print("[probe] no LLM_BASE_URI: the endpoint half is skipped", flush=True)

    print(f"[probe] summary: {results}", flush=True)
    # The probe reports; it does not fail the task on a negative answer, because
    # a negative answer is the result we are paying for.
    return 0


if __name__ == "__main__":
    sys.exit(main())
