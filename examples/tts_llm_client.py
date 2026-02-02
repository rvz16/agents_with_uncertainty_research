from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from llm_tts.integration import ChatTTS
from sage_agent import LLMClient


@dataclass
class TTSLLMClient(LLMClient):
    base_url: str
    model: str
    tts_strategy: str = "self_consistency"
    tts_budget: int = 8
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0
    system_prompt: str = (
        "You are a precise assistant. "
        "Respond with one line only in this exact format: Answer: <final answer>."
    )

    last_metadata: dict = None

    def __post_init__(self) -> None:
        self._llm = ChatTTS(
            base_url=self.base_url,
            model=self.model,
            tts_strategy=self.tts_strategy,
            tts_budget=self.tts_budget,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        self.last_metadata = {}

    def complete(self, prompt: str) -> str:
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        response = self._llm.invoke(messages)
        self.last_metadata = response.response_metadata.get("tts_metadata", {})
        return response.content

    @property
    def last_uncertainty(self) -> Optional[float]:
        if not self.last_metadata:
            return None
        return self.last_metadata.get("uncertainty_score")

    @property
    def last_confidence(self) -> Optional[float]:
        if not self.last_metadata:
            return None
        return self.last_metadata.get("consensus_score")
