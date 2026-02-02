import os

from openai import OpenAI

from sage_agent import LLMClient


class OpenRouterClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.verbose = verbose
        api_key = api_key or os.getenv("SAGE_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not set. "
                "Set SAGE_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
            )
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def complete(self, prompt: str) -> str:
        if self.verbose:
            print(f"\n--- OpenRouter prompt ({self.model}) ---\n{prompt}\n")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        if self.verbose:
            print(f"--- OpenRouter response ({self.model}) ---\n{content}\n")
        return content
