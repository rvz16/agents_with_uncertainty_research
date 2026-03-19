"""Main AlphaEvolve-style evolutionary loop for kernel optimization."""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..evaluator.models import KernelTask, EvalResult
from ..evaluator.evaluator import KernelBenchEvaluator
from .models import Program, ProgramScore
from .database import ProgramDatabase
from .prompt_sampler import PromptSampler
from .diff_engine import apply_or_fallback
from .llm_client import EvolveLLMClient
from .strategy_selector import MutationStrategySelector
from .uq_filter import UQPreCompilationFilter, parse_openai_logprobs

if TYPE_CHECKING:
    from ..bayesian_opt.optimizer import BayesianKernelOptimizer

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary loop."""
    max_iterations: int = 50
    candidates_per_iteration: int = 4
    n_inspirations: int = 2
    seed: int = 42
    log_interval: int = 5  # Print stats every N iterations


@dataclass
class EvolutionResult:
    """Result of a completed evolutionary run."""
    best: ProgramScore | None = None
    history: list[dict] = field(default_factory=list)
    total_evaluations: int = 0
    total_time_s: float = 0.0
    db_stats: dict = field(default_factory=dict)


class EvolutionaryLoop:
    """AlphaEvolve-style evolutionary loop for kernel optimization.

    Core loop (per iteration):
    1. Sample parent + inspirations from database
    2. Optionally get HP suggestions from Bayesian optimizer
    3. Build prompt via PromptSampler
    4. Generate N candidate diffs via LLM
    5. Apply diffs to produce child programs
    6. Evaluate children via KernelBenchEvaluator
    7. Store results in database
    8. Feed observations back to Bayesian optimizer
    """

    def __init__(
        self,
        task: KernelTask,
        evaluator: KernelBenchEvaluator,
        llm_client: EvolveLLMClient,
        database: ProgramDatabase | None = None,
        prompt_sampler: PromptSampler | None = None,
        config: EvolutionConfig | None = None,
        hp_optimizer: "BayesianKernelOptimizer | None" = None,
        strategy_selector: MutationStrategySelector | None = None,
        uq_filter: UQPreCompilationFilter | None = None,
    ):
        self.task = task
        self.evaluator = evaluator
        self.llm = llm_client
        self.db = database or ProgramDatabase()
        self.prompt_sampler = prompt_sampler or PromptSampler()
        self.config = config or EvolutionConfig()
        self.hp_optimizer = hp_optimizer
        self.strategy_selector = strategy_selector
        self.uq_filter = uq_filter
        self._rng = np.random.default_rng(self.config.seed)

    def _seed_database(self, initial_code: str):
        """Evaluate initial program and add to database."""
        logger.info("Evaluating initial program...")
        result = self.evaluator.evaluate(self.task, initial_code)

        program = Program(code=initial_code, generation=0)
        ps = ProgramScore(program=program, eval_result=result)
        self.db.add(ps)

        logger.info(f"Initial: {ps.summary()}")
        return ps

    def run(self, initial_code: str) -> EvolutionResult:
        """Run the full evolutionary loop.

        Args:
            initial_code: Starting ModelNew implementation.

        Returns:
            EvolutionResult with best program and history.
        """
        start_time = time.time()
        history = []
        total_evals = 0

        # Seed the database
        initial_ps = self._seed_database(initial_code)
        total_evals += 1

        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()

            # 1. Sample parent and inspirations
            parent = self.db.sample_parent(self._rng)
            inspirations = self.db.sample_inspirations(
                self.config.n_inspirations,
                self._rng,
                exclude_id=parent.program.id,
            )

            # 2. Get HP suggestions (optional)
            hp_suggestions = None
            if self.hp_optimizer:
                try:
                    suggested_config = self.hp_optimizer.suggest()
                    hp_suggestions = suggested_config.to_dict()
                except Exception as e:
                    logger.warning(f"HP optimizer suggest failed: {e}")

            # 3. Get strategy hint from POMDP selector (Variant 1)
            strategy_hint = None
            strategy_description = None
            if self.strategy_selector:
                strategy_hint = self.strategy_selector.select_strategy()
                strategy_description = self.strategy_selector.get_strategy_description(strategy_hint)

            # 4. Build prompt (with optional strategy hint)
            feedback = parent.eval_result if iteration > 1 else None
            messages = self.prompt_sampler.build_prompt(
                task=self.task,
                parent=parent,
                inspirations=inspirations,
                feedback=feedback,
                hp_suggestions=hp_suggestions,
                strategy_hint=strategy_hint,
                strategy_description=strategy_description,
            )

            # 5. Generate candidate diffs (with logprobs if UQ filter active)
            uq_scores: list[float | None] = []
            if self.uq_filter:
                try:
                    llm_outputs_with_lp = self.llm.generate_diffs_batch_with_logprobs(
                        messages, n=self.config.candidates_per_iteration
                    )
                    llm_outputs = []
                    raw_logprobs = []
                    for text, lp in llm_outputs_with_lp:
                        llm_outputs.append(text)
                        raw_logprobs.append(lp)
                except Exception as e:
                    logger.error(f"Iteration {iteration}: LLM generation failed: {e}")
                    continue
            else:
                try:
                    llm_outputs = self.llm.generate_diffs_batch(
                        messages, n=self.config.candidates_per_iteration
                    )
                    raw_logprobs = [None] * len(llm_outputs)
                except Exception as e:
                    logger.error(f"Iteration {iteration}: LLM generation failed: {e}")
                    continue

            # 6. Apply diffs, filter, evaluate, store
            iter_best = None
            n_filtered = 0
            for j, raw_output in enumerate(llm_outputs):
                # 6a. UQ filter: skip low-confidence generations (Variant 3)
                uq_score = None
                if self.uq_filter and raw_logprobs[j] is not None:
                    gen_logprobs = parse_openai_logprobs(raw_logprobs[j])
                    if gen_logprobs and len(gen_logprobs) > 0:
                        should_eval, uq_score = self.uq_filter.should_evaluate(gen_logprobs)
                        if not should_eval:
                            n_filtered += 1
                            continue

                # 6b. Apply diff or use as full replacement
                child_code, method = apply_or_fallback(parent.program.code, raw_output)

                if method == "no_change":
                    continue

                # 6c. Evaluate
                eval_result = self.evaluator.evaluate(self.task, child_code)
                total_evals += 1

                child_program = Program(
                    code=child_code,
                    parent_id=parent.program.id,
                    generation=iteration,
                    metadata={
                        "diff_method": method,
                        "candidate_idx": j,
                        "strategy": strategy_hint,
                        "uq_score": uq_score,
                    },
                )
                child_ps = ProgramScore(program=child_program, eval_result=eval_result)

                # 6d. Store in database
                is_new_elite = self.db.add(child_ps)

                if iter_best is None or child_ps.fitness > iter_best.fitness:
                    iter_best = child_ps

                # 6e. Feed outcome to POMDP selector (Variant 1)
                if self.strategy_selector and strategy_hint:
                    improved = child_ps.fitness > parent.fitness
                    self.strategy_selector.observe(strategy_hint, improved)

                # 6f. Feed outcome to UQ filter for calibration (Variant 3)
                if self.uq_filter and uq_score is not None:
                    self.uq_filter.observe(uq_score, eval_result.correctness)

                # Feed back to Bayesian optimizer
                if self.hp_optimizer and eval_result.correctness:
                    try:
                        self.hp_optimizer.observe_from_code(
                            child_code, eval_result.speedup
                        )
                    except Exception:
                        pass  # HP extraction may fail

            # Log progress
            iter_time = time.time() - iter_start
            iter_record = {
                "iteration": iteration,
                "iter_time_s": iter_time,
                "best_fitness": self.db.best.fitness if self.db.best else 0,
                "iter_best_fitness": iter_best.fitness if iter_best else 0,
                "total_evals": total_evals,
                "strategy": strategy_hint,
                "n_filtered": n_filtered,
            }
            history.append(iter_record)

            if iteration % self.config.log_interval == 0 or iteration == 1:
                stats = self.db.get_stats()
                log_parts = [
                    f"Iter {iteration}/{self.config.max_iterations}",
                    f"best={stats['best_fitness']:.3f}",
                    f"correct={stats['n_correct']}/{stats['size']}",
                    f"evals={total_evals}",
                    f"time={iter_time:.1f}s",
                ]
                if strategy_hint:
                    log_parts.append(f"strategy={strategy_hint}")
                if n_filtered > 0:
                    log_parts.append(f"uq_filtered={n_filtered}")
                logger.info(" | ".join(log_parts))

        total_time = time.time() - start_time
        return EvolutionResult(
            best=self.db.best,
            history=history,
            total_evaluations=total_evals,
            total_time_s=total_time,
            db_stats=self.db.get_stats(),
        )
