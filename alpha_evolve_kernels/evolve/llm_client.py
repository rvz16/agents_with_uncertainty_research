"""OpenAI-compatible LLM client for generating code diffs."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class EvolveLLMClient:
    """Thin wrapper around OpenAI-compatible API for diff generation."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        )

    def generate_diff(self, messages: list[dict[str, str]]) -> str:
        """Generate a single diff response from the LLM.

        Args:
            messages: OpenAI-format message list.

        Returns:
            Raw LLM output text containing SEARCH/REPLACE blocks.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_diffs_batch(
        self,
        messages: list[dict[str, str]],
        n: int = 4,
        temperatures: list[float] | None = None,
    ) -> list[str]:
        """Generate N candidate diffs in parallel.

        Uses different temperatures for diversity (AlphaEvolve insight:
        mix of exploration and exploitation).

        Args:
            messages: Shared message list.
            n: Number of candidates to generate.
            temperatures: Per-candidate temperatures. Defaults to
                [0.4, 0.7, 1.0, 1.2] pattern for exploration diversity.

        Returns:
            List of raw LLM outputs.
        """
        if temperatures is None:
            base_temps = [0.4, 0.7, 1.0, 1.2]
            temperatures = [base_temps[i % len(base_temps)] for i in range(n)]

        results = [""] * n

        def _generate_one(idx: int, temp: float) -> tuple[int, str]:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens,
            )
            return idx, response.choices[0].message.content or ""

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = {
                executor.submit(_generate_one, i, temperatures[i]): i
                for i in range(n)
            }
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    results[idx] = text
                except Exception as e:
                    idx = futures[future]
                    results[idx] = f"# LLM Error: {e}"

        return results

    def generate_diffs_batch_with_logprobs(
        self,
        messages: list[dict[str, str]],
        n: int = 4,
        temperatures: list[float] | None = None,
        top_logprobs: int = 5,
    ) -> list[tuple[str, Any]]:
        """Generate N candidate diffs with logprobs for UQ filtering.

        Same as generate_diffs_batch but requests logprobs from the API.
        The raw logprobs object is returned for parsing by uq_filter.py.

        Args:
            messages: Shared message list.
            n: Number of candidates to generate.
            temperatures: Per-candidate temperatures.
            top_logprobs: Number of top token alternatives to return.

        Returns:
            List of (raw_text, logprobs_object) tuples.
            logprobs_object is the raw API response logprobs field
            (parsed by uq_filter.parse_openai_logprobs).
        """
        if temperatures is None:
            base_temps = [0.4, 0.7, 1.0, 1.2]
            temperatures = [base_temps[i % len(base_temps)] for i in range(n)]

        results: list[tuple[str, Any]] = [("", None)] * n

        def _generate_one(idx: int, temp: float) -> tuple[int, str, Any]:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            lp = choice.logprobs
            return idx, text, lp

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = {
                executor.submit(_generate_one, i, temperatures[i]): i
                for i in range(n)
            }
            for future in as_completed(futures):
                try:
                    idx, text, lp = future.result()
                    results[idx] = (text, lp)
                except Exception as e:
                    idx = futures[future]
                    results[idx] = (f"# LLM Error: {e}", None)

        return results
