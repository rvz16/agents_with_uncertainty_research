"""MAP-Elites inspired program database for evolutionary search."""

import numpy as np
from typing import Optional

from .models import Program, ProgramScore, IslandConfig


class ProgramDatabase:
    """Stores programs with scores, inspired by MAP-Elites + island models.

    Key behaviors from AlphaEvolve:
    - Elite grid: best program per feature bin (diversity preservation)
    - Tournament selection: sample parent from top programs
    - Inspiration sampling: random programs for cross-pollination
    """

    def __init__(
        self,
        islands: list[IslandConfig] | None = None,
        max_programs: int = 1000,
    ):
        self._all: list[ProgramScore] = []
        self._elite_grid: dict[tuple[int, ...], ProgramScore] = {}
        self._islands = islands or []
        self._max_programs = max_programs
        self._best: ProgramScore | None = None

    @property
    def size(self) -> int:
        return len(self._all)

    @property
    def best(self) -> Optional[ProgramScore]:
        return self._best

    def add(self, ps: ProgramScore) -> bool:
        """Add a program to the database.

        Returns True if the program is a new elite (best in its bin).
        """
        self._all.append(ps)

        # Update global best
        if self._best is None or ps.fitness > self._best.fitness:
            self._best = ps

        # Update elite grid
        is_new_elite = False
        if self._islands:
            bin_key = tuple(
                island.get_bin(ps.program, ps.eval_result)
                for island in self._islands
            )
            existing = self._elite_grid.get(bin_key)
            if existing is None or ps.fitness > existing.fitness:
                self._elite_grid[bin_key] = ps
                is_new_elite = True

        # Enforce max size (remove worst non-elite programs)
        if len(self._all) > self._max_programs:
            elite_ids = {ps.program.id for ps in self._elite_grid.values()}
            non_elite = [p for p in self._all if p.program.id not in elite_ids]
            if non_elite:
                worst = min(non_elite, key=lambda p: p.fitness)
                self._all.remove(worst)

        return is_new_elite

    def sample_parent(self, rng: np.random.Generator) -> ProgramScore:
        """Sample a parent program using tournament selection.

        Picks 3 random programs, returns the one with highest fitness.
        Falls back to best program if database is very small.
        """
        if not self._all:
            raise ValueError("Database is empty")

        if len(self._all) <= 3:
            return max(self._all, key=lambda p: p.fitness)

        # Tournament selection (k=3)
        indices = rng.choice(len(self._all), size=min(3, len(self._all)), replace=False)
        tournament = [self._all[i] for i in indices]
        return max(tournament, key=lambda p: p.fitness)

    def sample_inspirations(self, n: int, rng: np.random.Generator, exclude_id: str | None = None) -> list[ProgramScore]:
        """Sample N random programs for cross-pollination.

        Prioritizes diverse selection from elite grid when available.
        """
        if not self._all:
            return []

        candidates = self._all
        if exclude_id:
            candidates = [p for p in candidates if p.program.id != exclude_id]

        if not candidates:
            return []

        n = min(n, len(candidates))

        # Mix: half from elites, half random
        if self._elite_grid:
            elites = list(self._elite_grid.values())
            if exclude_id:
                elites = [p for p in elites if p.program.id != exclude_id]

            n_elite = min(n // 2 + 1, len(elites))
            n_random = n - n_elite

            elite_idx = rng.choice(len(elites), size=n_elite, replace=False)
            selected = [elites[i] for i in elite_idx]

            if n_random > 0 and len(candidates) > n_elite:
                remaining = [p for p in candidates if p not in selected]
                if remaining:
                    rand_idx = rng.choice(len(remaining), size=min(n_random, len(remaining)), replace=False)
                    selected.extend(remaining[i] for i in rand_idx)

            return selected[:n]

        indices = rng.choice(len(candidates), size=n, replace=False)
        return [candidates[i] for i in indices]

    def get_top(self, n: int = 5) -> list[ProgramScore]:
        """Get top N programs by fitness."""
        sorted_all = sorted(self._all, key=lambda p: p.fitness, reverse=True)
        return sorted_all[:n]

    def get_stats(self) -> dict:
        """Summary statistics of the database."""
        if not self._all:
            return {"size": 0, "best_fitness": 0.0}

        fitnesses = [p.fitness for p in self._all]
        valid = [p for p in self._all if p.is_valid]
        correct = [p for p in self._all if p.eval_result.correctness]

        return {
            "size": len(self._all),
            "n_elites": len(self._elite_grid),
            "n_valid": len(valid),
            "n_correct": len(correct),
            "best_fitness": max(fitnesses),
            "mean_fitness": float(np.mean(fitnesses)),
            "best_speedup": max((p.eval_result.speedup for p in self._all), default=0.0),
            "generations": max((p.program.generation for p in self._all), default=0),
        }
