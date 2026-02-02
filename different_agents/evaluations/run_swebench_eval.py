#!/usr/bin/env python3
"""
SWE-bench Evaluation for Uncertainty-Guided LLM Agents.

SWE-bench presents real GitHub issues and requires generating patches to fix them.
This is more realistic than HumanEval but also more challenging.

Evaluation modes:
1. LITE - Patch similarity check (no Docker, fast but approximate)
2. FULL - Docker-based test execution (accurate but slow, requires setup)

Available datasets:
- SWE-bench_Lite: 300 instances, curated for faster evaluation
- SWE-bench_Verified: 500 expert-verified solvable problems
- SWE-bench: Full 2,294 instances

Usage:
    python run_swebench_eval.py --limit 10 --print-each
    python run_swebench_eval.py --dataset verified --limit 20
    python run_swebench_eval.py --use-ollama --limit 5
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Project root is 2 levels up from different_agents/evaluations/
ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = ROOT / "different_agents" / "shared"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SHARED_DIR))


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class PatchResult:
    """Result of evaluating a single patch."""
    instance_id: str
    repo: str
    
    # Generation
    generated_patch: str
    expected_patch: str
    
    # Metrics
    exact_match: bool
    similarity_score: float  # 0-1, higher is better
    file_match: bool  # Did we modify the right file?
    
    # Uncertainty
    uncertainty: float
    questions_asked: int = 0
    
    # Additional info
    error: Optional[str] = None


@dataclass 
class SWEBenchResult:
    """Aggregated results for SWE-bench evaluation."""
    dataset_name: str
    total_instances: int
    
    # Accuracy metrics
    exact_match_rate: float
    avg_similarity: float
    file_match_rate: float
    
    # Uncertainty metrics
    avg_uncertainty: float
    confident_similarity: float  # Similarity when confident
    
    # By repo breakdown
    per_repo_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


# =============================================================================
# LLM Client
# =============================================================================

def create_llm_client(args: argparse.Namespace):
    """Create LLM client based on args."""
    if args.use_ollama:
        from ollama_client import OllamaClient
        return OllamaClient(model=args.ollama_model, verbose=False)
    else:
        from tts_llm_client import TTSLLMClient
        return TTSLLMClient(
            base_url=args.service_url,
            model=args.model,
            tts_budget=args.tts_budget,
        )


def get_uncertainty(llm) -> float:
    """Get last uncertainty from LLM client."""
    uncertainty = getattr(llm, "last_uncertainty", None)
    return uncertainty if uncertainty is not None else 0.5


# =============================================================================
# Patch Generation and Evaluation
# =============================================================================

def generate_patch(
    llm,
    problem_statement: str,
    repo: str,
    hints: str = "",
) -> Tuple[str, float]:
    """Generate a patch for a GitHub issue.
    
    Returns:
        (generated_patch, uncertainty)
    """
    prompt = f"""You are a software engineer fixing a bug in the {repo} repository.

## Issue Description
{problem_statement}

{f"## Hints{chr(10)}{hints}" if hints else ""}

## Task
Generate a git diff patch that fixes this issue. 
Output ONLY the patch in unified diff format, starting with:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
```

Your patch:
"""
    
    response = llm.complete(prompt)
    uncertainty = get_uncertainty(llm)
    
    # Extract patch from response
    patch = _extract_patch(response)
    
    return patch, uncertainty


def _extract_patch(response: str) -> str:
    """Extract git diff patch from LLM response."""
    # Try to find diff block
    diff_match = re.search(r'```(?:diff)?\n([\s\S]*?)```', response)
    if diff_match:
        return diff_match.group(1).strip()
    
    # Look for unified diff pattern
    diff_pattern = re.search(r'(---\s+a/.*?\n\+\+\+\s+b/.*?\n[\s\S]*?)(?:\n\n|$)', response)
    if diff_pattern:
        return diff_pattern.group(1).strip()
    
    # Return cleaned response as fallback
    return response.strip()


def evaluate_patch(
    generated: str,
    expected: str,
) -> Tuple[bool, float, bool]:
    """Evaluate generated patch against expected.
    
    Returns:
        (exact_match, similarity_score, file_match)
    """
    # Normalize patches
    gen_lines = generated.strip().split('\n')
    exp_lines = expected.strip().split('\n')
    
    # Exact match
    exact = generated.strip() == expected.strip()
    
    # Similarity score using SequenceMatcher
    similarity = difflib.SequenceMatcher(None, generated, expected).ratio()
    
    # File match - check if same files are being modified
    gen_files = set(re.findall(r'[+-]{3}\s+[ab]/(.+)', generated))
    exp_files = set(re.findall(r'[+-]{3}\s+[ab]/(.+)', expected))
    file_match = bool(gen_files & exp_files) if exp_files else False
    
    return exact, similarity, file_match


def compute_line_level_metrics(generated: str, expected: str) -> Dict[str, float]:
    """Compute detailed line-level metrics."""
    gen_lines = set(generated.strip().split('\n'))
    exp_lines = set(expected.strip().split('\n'))
    
    # Lines that should be added (starting with +)
    gen_adds = {l for l in gen_lines if l.startswith('+')}
    exp_adds = {l for l in exp_lines if l.startswith('+')}
    
    # Lines that should be removed (starting with -)
    gen_dels = {l for l in gen_lines if l.startswith('-')}
    exp_dels = {l for l in exp_lines if l.startswith('-')}
    
    # Precision/Recall for additions
    add_precision = len(gen_adds & exp_adds) / len(gen_adds) if gen_adds else 0
    add_recall = len(gen_adds & exp_adds) / len(exp_adds) if exp_adds else 0
    
    # Precision/Recall for deletions
    del_precision = len(gen_dels & exp_dels) / len(gen_dels) if gen_dels else 0
    del_recall = len(gen_dels & exp_dels) / len(exp_dels) if exp_dels else 0
    
    return {
        "add_precision": add_precision,
        "add_recall": add_recall,
        "del_precision": del_precision,
        "del_recall": del_recall,
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_swebench(
    args: argparse.Namespace,
    llm,
) -> SWEBenchResult:
    """Run SWE-bench evaluation."""
    from datasets import load_dataset
    
    # Select dataset
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified", 
        "full": "princeton-nlp/SWE-bench",
    }
    dataset_name = dataset_map.get(args.dataset, dataset_map["lite"])
    
    print("\n" + "=" * 70)
    print(f"SWE-bench Evaluation ({args.dataset.upper()})")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {args.ollama_model if args.use_ollama else args.model}")
    print(f"Limit: {args.limit}")
    print()
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying 'dev' split...")
        dataset = load_dataset(dataset_name, split="dev")
    
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    # Evaluate each instance
    results: List[PatchResult] = []
    repo_stats: Dict[str, List[float]] = {}
    
    for i, instance in enumerate(dataset):
        instance_id = instance.get("instance_id", f"instance_{i}")
        repo = instance.get("repo", "unknown")
        problem = instance.get("problem_statement", "")
        hints = instance.get("hints_text", "")
        expected_patch = instance.get("patch", "")
        
        # Generate patch
        try:
            generated_patch, uncertainty = generate_patch(
                llm, problem, repo, hints
            )
            
            # Evaluate
            exact, similarity, file_match = evaluate_patch(
                generated_patch, expected_patch
            )
            
            result = PatchResult(
                instance_id=instance_id,
                repo=repo,
                generated_patch=generated_patch,
                expected_patch=expected_patch,
                exact_match=exact,
                similarity_score=similarity,
                file_match=file_match,
                uncertainty=uncertainty,
            )
            
        except Exception as e:
            result = PatchResult(
                instance_id=instance_id,
                repo=repo,
                generated_patch="",
                expected_patch=expected_patch,
                exact_match=False,
                similarity_score=0.0,
                file_match=False,
                uncertainty=1.0,
                error=str(e),
            )
        
        results.append(result)
        
        # Track per-repo stats
        if repo not in repo_stats:
            repo_stats[repo] = []
        repo_stats[repo].append(result.similarity_score)
        
        # Print progress
        if args.print_each:
            status = "✓" if result.exact_match else ("~" if result.similarity_score > 0.5 else "✗")
            file_status = "F✓" if result.file_match else "F✗"
            print(f"[{i+1}/{len(dataset)}] {instance_id[:40]:40} {status} "
                  f"sim={result.similarity_score:.2f} {file_status} "
                  f"unc={result.uncertainty:.2f}")
            if result.error:
                print(f"    Error: {result.error[:60]}")
    
    # Compute aggregated metrics
    exact_matches = sum(1 for r in results if r.exact_match)
    file_matches = sum(1 for r in results if r.file_match)
    avg_sim = sum(r.similarity_score for r in results) / len(results) if results else 0
    avg_unc = sum(r.uncertainty for r in results) / len(results) if results else 0
    
    # Confident predictions (low uncertainty)
    confident = [r for r in results if r.uncertainty < args.uncertainty_threshold]
    confident_sim = sum(r.similarity_score for r in confident) / len(confident) if confident else 0
    
    # Per-repo breakdown
    per_repo = {
        repo: {
            "count": len(sims),
            "avg_similarity": sum(sims) / len(sims) if sims else 0,
        }
        for repo, sims in repo_stats.items()
    }
    
    # Print summary
    print("\n" + "-" * 70)
    print("Results Summary")
    print("-" * 70)
    print(f"Total instances:       {len(results)}")
    print(f"Exact match rate:      {exact_matches/len(results):.4f} ({exact_matches}/{len(results)})")
    print(f"File match rate:       {file_matches/len(results):.4f} ({file_matches}/{len(results)})")
    print(f"Avg similarity:        {avg_sim:.4f}")
    print(f"Avg uncertainty:       {avg_unc:.4f}")
    print(f"Confident similarity:  {confident_sim:.4f} (n={len(confident)})")
    
    if len(per_repo) <= 10:
        print("\nPer-repository breakdown:")
        for repo, stats in sorted(per_repo.items(), key=lambda x: -x[1]["avg_similarity"]):
            print(f"  {repo[:30]:30} sim={stats['avg_similarity']:.2f} (n={stats['count']})")
    
    return SWEBenchResult(
        dataset_name=args.dataset,
        total_instances=len(results),
        exact_match_rate=exact_matches / len(results) if results else 0,
        avg_similarity=avg_sim,
        file_match_rate=file_matches / len(results) if results else 0,
        avg_uncertainty=avg_unc,
        confident_similarity=confident_sim,
        per_repo_stats=per_repo,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench evaluation with uncertainty quantification."
    )
    parser.add_argument(
        "--dataset",
        choices=["lite", "verified", "full"],
        default="lite",
        help="SWE-bench dataset variant.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of instances to evaluate (0 = all).",
    )
    parser.add_argument(
        "--model",
        default="mistralai/devstral-2512:free",
        help="Model for TTS service.",
    )
    parser.add_argument(
        "--service-url",
        default="http://localhost:8001/v1",
        help="TTS service URL.",
    )
    parser.add_argument(
        "--tts-budget",
        type=int,
        default=4,
        help="TTS reasoning budget.",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama instead of TTS.",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3:4b-instruct-2507-q8_0",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print per-instance results.",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.5,
        help="Threshold for confident predictions.",
    )
    
    args = parser.parse_args()
    
    llm = create_llm_client(args)
    result = evaluate_swebench(args, llm)
    
    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

