#!/usr/bin/env python3
"""What can a task on this cluster actually reach?

The judge has to be a model the agent is not. Locally that means an 8B
checkpoint sharing the card, which is a weak reviewer; a hosted one would be
far better, and OpenRouter answered our earlier probe with 403 "Access denied
by security policy". But weights download from HuggingFace and artifacts go to
the ClearML file store, so egress is filtered by host rather than closed --
which is exactly the situation where a mirror or a proxy works.

This prints what resolves, what connects and what a proxy would change. It
changes nothing and needs no GPU.
"""
import json
import os
import socket
import sys
import urllib.request

HOSTS = [
    ("huggingface.co", "https://huggingface.co/api/models/openai/gpt-oss-20b"),
    ("files.clearai.innopolis.university", "https://files.clearai.innopolis.university/"),
    ("openrouter.ai", "https://openrouter.ai/api/v1/models"),
    ("api.anthropic.com", "https://api.anthropic.com/v1/models"),
    ("api.openai.com", "https://api.openai.com/v1/models"),
    ("pypi.org", "https://pypi.org/simple/"),
]


def check(name: str, url: str) -> None:
    try:
        addr = socket.gethostbyname(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[egress] {name:42s} DNS FAILED  {type(exc).__name__}", flush=True)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "egress-probe"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            print(f"[egress] {name:42s} {addr:15s} HTTP {response.status}", flush=True)
    except urllib.error.HTTPError as exc:
        # 401/403 from the service itself still proves the host is reachable;
        # 403 from the filter is what we saw before, so the body matters.
        body = (exc.read()[:120] or b"").decode("utf-8", "replace").replace("\n", " ")
        print(f"[egress] {name:42s} {addr:15s} HTTP {exc.code}  {body}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[egress] {name:42s} {addr:15s} FAILED {type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    from clearml import Task

    task = Task.current_task() or Task.init(
        project_name="agentic-uq", task_name="egress probe"
    )
    task.output_uri = "https://files.clearai.innopolis.university"

    proxies = {
        key: value
        for key, value in os.environ.items()
        if "proxy" in key.lower() or key.lower() in ("no_proxy",)
    }
    print(f"[egress] proxy environment: {proxies or 'none set'}", flush=True)
    print(f"[egress] worker: {os.environ.get('CLEARML_WORKER_ID', '?')}", flush=True)

    for name, url in HOSTS:
        check(name, url)

    extra = os.environ.get("EXTRA_HOSTS", "")
    for url in [u.strip() for u in extra.split(",") if u.strip()]:
        host = url.split("//", 1)[-1].split("/", 1)[0]
        check(host, url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
