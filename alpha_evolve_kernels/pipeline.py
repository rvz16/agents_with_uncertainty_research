"""End-to-end pipeline wiring all 3 prototypes together.

Usage:
    from alpha_evolve_kernels.pipeline import AlphaEvolveKernelPipeline

    pipeline = AlphaEvolveKernelPipeline(
        task=task,
        llm_config={"base_url": "...", "model": "...", "api_key": "..."},
    )
    result = pipeline.run(initial_code)
"""

import logging
from dataclasses import dataclass

from .evaluator.models import KernelTask
from .evaluator.evaluator import KernelBenchEvaluator
from .evolve.models import ProgramScore, IslandConfig, Program
from .evolve.database import ProgramDatabase
from .evolve.prompt_sampler import PromptSampler
from .evolve.llm_client import EvolveLLMClient
from .evolve.loop import EvolutionaryLoop, EvolutionConfig, EvolutionResult
from .evolve.strategy_selector import MutationStrategySelector, StrategyConfig
from .evolve.uq_filter import UQPreCompilationFilter, UQFilterConfig
from .bayesian_opt.models import ParameterDomain, COMMON_DOMAINS
from .bayesian_opt.optimizer import BayesianKernelOptimizer

logger = logging.getLogger(__name__)


def code_length_feature(program: Program, eval_result) -> float:
    """Feature function: code length (for MAP-Elites diversity)."""
    return len(program.code)


def uses_shared_memory_feature(program: Program, eval_result) -> float:
    """Feature function: whether code uses shared memory."""
    return 1.0 if "__shared__" in program.code or "shared_mem" in program.code else 0.0


DEFAULT_ISLANDS = [
    IslandConfig(
        name="code_length",
        feature_fn=code_length_feature,
        bins=[500, 1000, 2000, 4000],
    ),
    IslandConfig(
        name="uses_shared_memory",
        feature_fn=uses_shared_memory_feature,
        bins=[0.5],  # 0 or 1
    ),
]


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Evolution
    max_iterations: int = 20
    candidates_per_iteration: int = 4
    n_inspirations: int = 2
    seed: int = 42

    # Bayesian optimization
    use_bayesian_opt: bool = True
    hp_domains: list[ParameterDomain] | None = None
    acquisition: str = "ei"
    n_initial_random: int = 5

    # Evaluator
    device: str | None = None
    n_correctness_trials: int = 5
    n_timing_trials: int = 100

    # MAP-Elites
    use_islands: bool = True

    # POMDP strategy selector (Variant 1)
    use_strategy_selector: bool = False
    strategy_budget_depth: int = 2

    # UQ pre-compilation filter (Variant 3)
    use_uq_filter: bool = False
    uq_threshold: float = 5.0
    uq_estimator: str = "perplexity"


class AlphaEvolveKernelPipeline:
    """End-to-end pipeline: AlphaEvolve + KernelBench Evaluator + Bayesian HP Opt."""

    def __init__(
        self,
        task: KernelTask,
        llm_config: dict,
        config: PipelineConfig | None = None,
    ):
        self.task = task
        self.config = config or PipelineConfig()

        # 1. Evaluator (Prototype 1)
        self.evaluator = KernelBenchEvaluator(
            device=self.config.device,
            n_correctness_trials=self.config.n_correctness_trials,
            n_timing_trials=self.config.n_timing_trials,
        )

        # 2. LLM Client
        self.llm_client = EvolveLLMClient(
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            model=llm_config.get("model", "gpt-4o"),
            api_key=llm_config.get("api_key"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 4096),
        )

        # 3. Program Database with MAP-Elites
        islands = DEFAULT_ISLANDS if self.config.use_islands else []
        self.database = ProgramDatabase(islands=islands)

        # 4. Prompt Sampler
        self.prompt_sampler = PromptSampler()

        # 5. Bayesian HP Optimizer (Prototype 3)
        self.hp_optimizer = None
        if self.config.use_bayesian_opt:
            domains = self.config.hp_domains or [
                COMMON_DOMAINS["block_size_x"],
                COMMON_DOMAINS["tile_size"],
                COMMON_DOMAINS["num_warps"],
            ]
            self.hp_optimizer = BayesianKernelOptimizer(
                domains=domains,
                acquisition=self.config.acquisition,
                n_initial=self.config.n_initial_random,
                seed=self.config.seed,
            )

        # 6. POMDP Strategy Selector (Variant 1)
        self.strategy_selector = None
        if self.config.use_strategy_selector:
            self.strategy_selector = MutationStrategySelector(
                StrategyConfig(budget_depth=self.config.strategy_budget_depth)
            )

        # 7. UQ Pre-Compilation Filter (Variant 3)
        self.uq_filter = None
        if self.config.use_uq_filter:
            self.uq_filter = UQPreCompilationFilter(
                UQFilterConfig(
                    threshold=self.config.uq_threshold,
                    estimator_type=self.config.uq_estimator,
                )
            )

        # 8. Evolution Config
        self.evolution_config = EvolutionConfig(
            max_iterations=self.config.max_iterations,
            candidates_per_iteration=self.config.candidates_per_iteration,
            n_inspirations=self.config.n_inspirations,
            seed=self.config.seed,
        )

    def run(self, initial_code: str) -> EvolutionResult:
        """Run the full pipeline.

        Args:
            initial_code: Starting ModelNew implementation.

        Returns:
            EvolutionResult with best program, history, and stats.
        """
        logger.info(f"Starting AlphaEvolve pipeline for task '{self.task.task_id}'")
        logger.info(f"Config: {self.config.max_iterations} iterations, "
                    f"{self.config.candidates_per_iteration} candidates/iter")

        loop = EvolutionaryLoop(
            task=self.task,
            evaluator=self.evaluator,
            llm_client=self.llm_client,
            database=self.database,
            prompt_sampler=self.prompt_sampler,
            config=self.evolution_config,
            hp_optimizer=self.hp_optimizer,
            strategy_selector=self.strategy_selector,
            uq_filter=self.uq_filter,
        )

        result = loop.run(initial_code)

        logger.info(f"Pipeline complete: {result.total_evaluations} evaluations "
                    f"in {result.total_time_s:.1f}s")
        if result.best:
            logger.info(f"Best: {result.best.summary()}")

        return result
