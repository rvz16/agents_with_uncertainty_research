"""Prompt construction for the AlphaEvolve evolutionary loop.

Builds rich prompts from program database contents, following AlphaEvolve's
prompt sampling approach: parent program + inspirations + evaluation feedback.
"""

from ..evaluator.models import KernelTask, EvalResult
from .models import ProgramScore


SYSTEM_PROMPT = """\
Act as an expert GPU kernel engineer. Your task is to iteratively \
improve the provided codebase to produce faster, correct CUDA kernels.

You will receive:
1. A reference PyTorch model (class Model) that defines the target computation
2. The current best implementation (class ModelNew) with its evaluation results
3. Optionally, other high-performing implementations for inspiration
4. Evaluation feedback (compilation errors, correctness issues, timing data)

Your goal: propose changes to ModelNew that make it faster while maintaining \
correctness. Use techniques like tiling, shared memory, operator fusion, \
vectorized loads, warp-level primitives, etc.

Describe each change with a SEARCH/REPLACE block:

<<<<<<< SEARCH
# Original code block to be found and replaced
=======
# New code block to replace the original
>>>>>>> REPLACE

Make sure the changes you propose are consistent with each other. \
If you add a new variable, also update code that uses it.
"""


class PromptSampler:
    """Builds OpenAI-format message lists for the LLM."""

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def build_prompt(
        self,
        task: KernelTask,
        parent: ProgramScore,
        inspirations: list[ProgramScore] | None = None,
        feedback: EvalResult | None = None,
        hp_suggestions: dict | None = None,
        strategy_hint: str | None = None,
        strategy_description: str | None = None,
    ) -> list[dict[str, str]]:
        """Build a complete prompt for the LLM.

        Args:
            task: The kernel optimization task.
            parent: Current best program to mutate.
            inspirations: Other high-performing programs for cross-pollination.
            feedback: Previous evaluation result (errors, timing).
            hp_suggestions: Suggested hyperparameters from Bayesian optimizer.

        Returns:
            List of message dicts in OpenAI format.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        user_parts = []

        # Reference model
        user_parts.append("## Reference PyTorch Model\n")
        user_parts.append(f"```python\n{task.reference_code}\n```\n")

        # Current program (parent)
        user_parts.append("## Current Implementation\n")
        user_parts.append(f"Score: speedup={parent.eval_result.speedup:.4f}x, "
                         f"correct={parent.eval_result.correctness}\n")
        user_parts.append(f"```python\n{parent.program.code}\n```\n")

        # Evaluation feedback
        if feedback:
            user_parts.append("## Previous Evaluation Feedback\n")
            user_parts.append(self._format_feedback(feedback))
            user_parts.append("\n")

        # Inspirations
        if inspirations:
            user_parts.append("## Other High-Performing Implementations\n")
            user_parts.append(
                "The following programs also performed well. "
                "You may draw ideas from them:\n"
            )
            for i, insp in enumerate(inspirations):
                user_parts.append(
                    f"\n### Inspiration {i+1} "
                    f"(speedup={insp.eval_result.speedup:.4f}x, "
                    f"correct={insp.eval_result.correctness})\n"
                )
                user_parts.append(f"```python\n{insp.program.code}\n```\n")

        # Hyperparameter suggestions from Bayesian optimizer
        if hp_suggestions:
            user_parts.append("## Suggested Hyperparameters\n")
            user_parts.append(
                "The Bayesian optimizer suggests trying these values "
                "(based on previous experiments):\n"
            )
            for name, value in hp_suggestions.items():
                user_parts.append(f"- {name} = {value}\n")
            user_parts.append("\n")

        # Strategy hint from POMDP selector
        if strategy_hint:
            user_parts.append("## Suggested Optimization Strategy\n")
            user_parts.append(f"Focus on: **{strategy_hint}**\n")
            if strategy_description:
                user_parts.append(f"{strategy_description}\n")
            user_parts.append(
                "This strategy has been identified as most promising "
                "by the optimization controller. Prioritize changes "
                "aligned with this approach.\n\n"
            )

        # Task instruction
        user_parts.append("## Task\n")
        user_parts.append(
            "Suggest improvements to the current implementation. "
            "Focus on making it faster while maintaining correctness. "
            "Describe each change with a SEARCH/REPLACE block.\n"
        )

        messages.append({"role": "user", "content": "".join(user_parts)})

        return messages

    def _format_feedback(self, result: EvalResult) -> str:
        """Format evaluation result as human-readable feedback."""
        lines = []
        if result.compile_error:
            lines.append(f"**COMPILATION FAILED:**\n```\n{result.compile_error}\n```")
            lines.append("Fix the compilation error before attempting optimizations.")
        elif result.runtime_error:
            lines.append(f"**RUNTIME ERROR:**\n```\n{result.runtime_error}\n```")
        elif result.correctness_error:
            lines.append(f"**INCORRECT OUTPUT:**\n{result.correctness_error}")
            lines.append(f"\nPerformance (despite incorrectness):\n{result.profiler_output}")
            lines.append("Fix correctness first, then optimize for speed.")
        else:
            lines.append(f"**Status:** {'CORRECT' if result.correctness else 'INCORRECT'}")
            lines.append(f"**Speedup:** {result.speedup:.4f}x")
            if result.correctness and result.speedup > 1.0:
                lines.append("\nThe kernel is already faster than baseline. Try to improve further.")
            elif result.correctness:
                lines.append("\nThe kernel is correct but slower than baseline. Optimize for speed.")
        return "\n".join(lines)
