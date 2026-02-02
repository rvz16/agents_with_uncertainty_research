import json
import urllib.request

from sage_agent import LLMClient


class OllamaClient(LLMClient):
    def __init__(
        self, model: str, base_url: str = "http://localhost:11434", verbose: bool = False
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.verbose = verbose

    def complete(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if self.verbose:
            print(f"\n--- Ollama prompt ({self.model}) ---\n{prompt}\n")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        response = data.get("response", "")
        if self.verbose:
            print(f"--- Ollama response ({self.model}) ---\n{response}\n")
        return response
