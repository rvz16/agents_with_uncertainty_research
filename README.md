# Uncertainty-Aware AI Agents Research

Research repository exploring **uncertainty quantification**, **structured output**, and **constraint decoding** techniques for LLM-powered tool-calling agents.

## Research Focus

This repository implements and evaluates methods for building more reliable AI agents that:

1. **Quantify uncertainty** in tool call decisions
2. **Use structured outputs** with schema-guided reasoning (SGR)
3. **Apply constraint decoding** to ensure valid tool calls
4. **Ask clarifying questions** when uncertainty is high instead of guessing

### Papers & Techniques Implemented

| Technique | Paper | Description |
|-----------|-------|-------------|
| **SAGE-Agent** | [arXiv:2511.08798](https://arxiv.org/abs/2511.08798) | POMDP-based belief tracking with EVPI question selection |
| **Self-Consistency UQ** | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) | Multiple sampling for uncertainty estimation |
| **Schema-Guided Reasoning** | SGR | Pydantic schemas for structured tool calls |
| **GRPO Training** | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | Certainty-weighted reinforcement learning |
| **Reflexion** | [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) | Self-reflection for error recovery |

### Evaluation Benchmarks

| Benchmark | Type | Description |
|-----------|------|-------------|
| **When2Call** | Tool calling | NVIDIA benchmark for tool call disambiguation |
| **ClarifyBench** | Clarification | Simulated clarification scenarios |
| **BFCL** | Function calling | Berkeley Function Calling Leaderboard |
| **HumanEval** | Code generation | OpenAI code generation benchmark |
| **GSM8K** | Math reasoning | Grade school math problems |

---

## SAGE-Agent: Core Implementation

**Structured Uncertainty Guided Clarification for LLM Agents**

The primary implementation is SAGE-Agent from [arXiv:2511.08798](https://arxiv.org/abs/2511.08798), which models tool calling as a POMDP and uses belief state tracking to decide when to ask clarifying questions.

### Key Insight

Traditional LLM agents often make tool calls with incomplete information, leading to errors. SAGE-Agent quantifies uncertainty and asks targeted questions when confidence is low:

```
User: "Book me a flight to NYC"

Traditional Agent: book_flight(origin=???, dest="NYC", date=???)  <- Guesses or fails

SAGE-Agent: "Which city will you be departing from?"
User: "Boston"
SAGE-Agent: "When would you like to travel?"
User: "March 15th"
SAGE-Agent: book_flight(origin="BOS", dest="NYC", date="2024-03-15")  <- Confident execution
```

### Algorithm 1: SAGE Decision Loop

```
1: Initialize belief B(0), aspect counts n(0), t <- 0
2: while t < T_max do
3:     Generate candidates C, compute pi_c(t)
4:     if max_c pi_c(t) >= tau_exec then Execute and return
5:     Generate questions Q
6:     Score: q* = argmax[EVPI(q) - Cost(q)]
7:     if Score(q*) < alpha * max_prob then Execute
8:     Ask q*, update domains, t++
9: end while
10: Final execution or escalation
```

### Paper Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `tau_exec` | τ | 0.85 | Execution confidence threshold |
| `alpha` | α | 0.1 | Termination threshold factor |
| `lambda_redundancy` | λ | 0.5 | Redundancy weight |
| `epsilon` | ε | 1e-4 | Small prob for infinite domains |
| `max_questions` | T_max | 6 | Maximum clarification questions |

---

## Implementation Variants

This repository contains multiple implementation variants for experimentation:

| Implementation | File | Features |
|---------------|------|----------|
| **Pure SAGE** | `different_agents/pure_sage/langgraph_sage_agent.py` | Clean Algorithm 1, paper hyperparameters only |
| **Experimental** | `different_agents/experimental/langgraph_sage_agent_experimental.py` | + SAUP uncertainty propagation, reflexion |
| **V3 (Full)** | `different_agents/v3/langgraph_sage_agent_v3.py` | + SGR, dynamic resampling, epistemic/aleatoric decomposition |
| **V4** | `different_agents/v4/langgraph_sage_agent_v4.py` | + Advanced configurations |
| **SGR-SAGE-UQ** | `different_agents/sgr_sage_uq/sgr_sage_uq_agent.py` | Schema-Guided Reasoning + Self-Consistency UQ + SAGE |

### Feature Comparison

| Feature | Pure SAGE | Experimental | V3 | SGR-SAGE-UQ |
|---------|-----------|--------------|-----|-------------|
| Belief State Tracking | ✓ | ✓ | ✓ | ✓ |
| EVPI Question Selection | ✓ | ✓ | ✓ | ✓ |
| SAUP Propagation | - | ✓ | ✓ | - |
| Reflexion | - | ✓ | - | - |
| Schema-Guided Reasoning | - | - | ✓ | ✓ |
| Self-Consistency UQ | - | - | ✓ | ✓ |
| Dynamic Resampling | - | - | ✓ | - |
| Epistemic/Aleatoric Split | - | - | ✓ | - |

---

## Installation

### Basic Installation

```bash
cd agents_with_uncertainty_research
pip install -e .
```

### With Optional Dependencies

```bash
# LangGraph support
pip install -e ".[langgraph]"

# OpenRouter LLM client
pip install -e ".[openrouter]"

# GRPO training (requires torch)
pip install -e ".[training]"

# All dependencies
pip install -e ".[all]"

# Development (includes pytest)
pip install -e ".[dev]"
```

### External Requirements

```bash
pip install datasets  # For When2Call, HumanEval, GSM8K benchmarks
```

---

## Quick Start

### 1. Pure SAGE-Agent (Paper Implementation)

```python
from sage_agent import (
    SAGEAgent,
    SAGEConfig,
    ToolSchema,
    ParameterDomain,
    SimpleConstraintExtractor,
)

# Define a tool with parameter domains
flight_tool = ToolSchema(
    name="book_flight",
    parameters={
        "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
        "destination": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
        "date": ParameterDomain.continuous(),
    },
    required=frozenset({"origin", "destination", "date"}),
)

# Create agent with paper hyperparameters
agent = SAGEAgent(
    tool_schemas=[flight_tool],
    candidate_generator=my_candidate_gen,
    question_generator=my_question_gen,
    question_asker=my_asker,
    tool_executor=my_executor,
    constraint_extractor=SimpleConstraintExtractor(),
    config=SAGEConfig(
        tau_exec=0.85,
        alpha=0.1,
        lambda_redundancy=0.5,
        max_questions=6,
    ),
)

result = agent.run("Book me a flight to NYC")
print(f"Tool call: {result.tool_call}")
print(f"Questions asked: {result.total_questions}")
print(f"Confidence: {result.final_probability:.2%}")
```

### 2. LangGraph Implementation

```python
from examples.langgraph_sage_agent import (
    GraphDeps, SAGEConfig, build_graph, create_initial_state,
)
from sage_agent import (
    LLMBackedCandidateGenerator,
    LLMBackedQuestionGenerator,
    SimpleConstraintExtractor,
    ToolRegistryExecutor,
)
from examples.openrouter_client import OpenRouterClient

llm = OpenRouterClient(model="openai/gpt-4o-mini")

deps = GraphDeps(
    tool_schemas={tool.name: tool},
    candidate_generator=LLMBackedCandidateGenerator(llm),
    question_generator=LLMBackedQuestionGenerator(llm),
    question_asker=ConsoleAsker(),
    tool_executor=ToolRegistryExecutor({"book_flight": my_function}),
    constraint_extractor=SimpleConstraintExtractor(),
    config=SAGEConfig(),
)

graph = build_graph(deps).compile()
result = graph.invoke(create_initial_state("Book flight to LAX", deps.tool_schemas))
```

### 3. SGR Tools (Schema-Guided Reasoning)

```python
from sage_agent import GeneratePlanTool, ReasoningTool

# Structured plan generation
plan = GeneratePlanTool(
    reasoning="User needs flight booking with missing parameters",
    research_goal="Book optimal flight based on user preferences",
    planned_steps=[
        "Clarify departure city",
        "Clarify travel date",
        "Search available flights",
        "Present options to user"
    ],
    search_strategies=["Ask clarifying questions", "Use flight search API"]
)

# Step-by-step reasoning
reasoning = ReasoningTool(
    reasoning_steps=["Origin is missing", "Need to ask clarifying question"],
    current_situation="User requested flight to NYC, departure unknown",
    plan_status="Step 1 of 3: Gathering required parameters",
    enough_data=False,
    remaining_steps=["Ask for origin", "Ask for date", "Execute search"],
    task_completed=False
)
```

---

## Running Evaluations

### When2Call Benchmark

```bash
# Pure SAGE evaluation
python different_agents/pure_sage/run_sage_eval.py --dataset when2call --limit 20 --print-each

# With OpenRouter
python different_agents/pure_sage/run_sage_eval.py --dataset when2call --use-openrouter --model openai/gpt-4o-mini

# With Ollama (local)
python different_agents/pure_sage/run_sage_eval.py --dataset when2call --use-ollama --ollama-model qwen2.5:7b-instruct

# Custom hyperparameters
python different_agents/pure_sage/run_sage_eval.py --dataset when2call --tau 0.9 --alpha 0.05 --max-questions 4
```

### ClarifyBench Benchmark

ClarifyBench evaluates agents on tasks requiring clarification. Available splits:

| Split | Description | Examples |
|-------|-------------|----------|
| `sample` | Sample examples | 10 |
| `A` | Ambiguous queries (need clarification) | ~200 |
| `E` | Explicit queries | ~240 |
| `I` | Implicit queries | ~150 |

```bash
# Test on sample data
python different_agents/evaluations/run_clarifybench_eval.py --split sample --limit 5 --print-each

# Ambiguous split (most relevant for SAGE - requires clarification)
python different_agents/evaluations/run_clarifybench_eval.py --split A --limit 50 --print-each

# With OpenRouter
python different_agents/evaluations/run_clarifybench_eval.py --split A --use-openrouter --model openai/gpt-4o-mini --limit 20

# With Ollama (local)
python different_agents/evaluations/run_clarifybench_eval.py --split sample --use-ollama --limit 10

# Custom hyperparameters
python different_agents/evaluations/run_clarifybench_eval.py --split A --tau 0.9 --alpha 0.05 --max-questions 4 --limit 20
```

### Experimental Variants (V3)

```bash
# V3 with full features
python different_agents/evaluations/run_when2call_eval.py --use-v3 --v3-config balanced --limit 10 --print-each

# V3 configurations: conservative, balanced, aggressive
python different_agents/evaluations/run_when2call_eval.py --use-v3 --v3-config aggressive --limit 50
```

### SGR-SAGE-UQ Benchmarks

```bash
# BFCL benchmark
python different_agents/sgr_sage_uq/run_sgr_sage_benchmarks.py --benchmark bfcl --limit 20

# GSM8K math reasoning
python different_agents/sgr_sage_uq/run_sgr_sage_benchmarks.py --benchmark gsm8k --limit 50

# HumanEval code generation
python different_agents/sgr_sage_uq/run_sgr_sage_benchmarks.py --benchmark humaneval --limit 20

# All benchmarks
python different_agents/sgr_sage_uq/run_sgr_sage_benchmarks.py --benchmark all --limit 10
```

### Multi-Benchmark Evaluation

```bash
python different_agents/evaluations/run_multi_benchmark_eval.py --benchmark humaneval --limit 10
python different_agents/evaluations/run_multi_benchmark_eval.py --benchmark gsm8k --limit 20
python different_agents/evaluations/run_multi_benchmark_eval.py --benchmark all --limit 5
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Tool Match Rate** | Fraction of correct tool names |
| **Parameter Match Rate** | Fraction of correct parameter values |
| **Coverage Rate** | Fraction of non-escalated predictions |
| **Avg Questions** | Average clarification questions asked |
| **ECE** | Expected Calibration Error |
| **Confident Accuracy** | Accuracy on low-uncertainty predictions |

---

## Project Structure

```
agents_with_uncertainty_research/
├── sage_agent/                    # Main package
│   ├── core/                      # Core algorithms
│   │   ├── sage_algorithm.py      # SAGEAgent (Algorithm 1)
│   │   ├── belief.py              # BeliefState
│   │   ├── evpi.py                # EVPI computation
│   │   ├── domains.py             # ParameterDomain
│   │   ├── types.py               # Type definitions
│   │   ├── pomdp.py               # POMDP formulation
│   │   ├── constraints.py         # Constraint extractors
│   │   ├── uncertainty_propagation.py  # SAUP
│   │   └── advanced_reasoning.py  # Decomposition, CoT, Reflexion
│   ├── langgraph/                 # LangGraph integration
│   │   └── sage_graph.py          # Canonical LangGraph impl
│   ├── tools/                     # SGR tools
│   │   └── sgr_tools.py           # GeneratePlanTool, ReasoningTool
│   ├── training/                  # GRPO training
│   │   └── grpo.py                # Certainty-weighted GRPO
│   ├── llm/                       # LLM clients
│   │   ├── openrouter.py
│   │   └── tts_service.py
│   ├── wiring/                    # LLM-backed generators
│   │   └── wiring.py
│   ├── metrics/                   # Evaluation metrics
│   │   └── metrics.py
│   └── sim/                       # Simulation
│       └── clarifybench.py
│
├── different_agents/              # Agent implementations & evaluations
│   ├── pure_sage/                 # Pure SAGE (Algorithm 1 from paper)
│   │   ├── langgraph_sage_agent.py
│   │   └── run_sage_eval.py
│   ├── experimental/              # SAUP + Reflexion
│   │   └── langgraph_sage_agent_experimental.py
│   ├── v3/                        # SGR + Resampling + SAUP
│   │   ├── langgraph_sage_agent_v3.py
│   │   ├── v3_configs.py
│   │   └── run_multi_benchmark_sage_v3.py
│   ├── v4/                        # Advanced configurations
│   │   ├── langgraph_sage_agent_v4.py
│   │   ├── langgraph_sage_agent_v4_swe.py
│   │   ├── v4_configs.py
│   │   └── run_*.py
│   ├── sgr_sage_uq/               # SGR + SAGE + Self-Consistency UQ
│   │   ├── sgr_sage_uq_agent.py
│   │   ├── sgr_plan_sage_tts_agent.py
│   │   ├── run_sgr_sage_benchmarks.py
│   │   └── test_*.py
│   ├── shared/                    # Shared LLM clients
│   │   ├── openrouter_client.py
│   │   ├── ollama_client.py
│   │   └── tts_llm_client.py
│   ├── evaluations/               # Benchmark evaluation scripts
│   │   ├── run_multi_benchmark_eval.py
│   │   ├── run_when2call_eval.py
│   │   ├── run_clarifybench_eval.py
│   │   └── run_swebench_eval.py
│   └── misc/                      # Examples, old versions, tests
│       ├── basic_usage.py
│       ├── grpo_training.py
│       └── langgraph_sage_agent_v2.py
│
├── tests/                         # Test suite
│   └── test_sage_agent_v1.py      # Pure SAGE tests
│
└── pyproject.toml                 # Package configuration
```

---

## LLM Backends

### OpenRouter

```python
from examples.openrouter_client import OpenRouterClient

llm = OpenRouterClient(
    model="openai/gpt-4o-mini",
    api_key="your-key",  # or set OPENROUTER_API_KEY
)
```

### Ollama (Local)

```python
from examples.ollama_client import OllamaClient

llm = OllamaClient(model="qwen2.5:7b-instruct")
```

### TTS Service (with Self-Consistency UQ)

```python
from examples.tts_llm_client import TTSLLMClient

llm = TTSLLMClient(
    base_url="http://localhost:8001/v1",
    model="openai/gpt-4o-mini",
    tts_budget=8,  # Number of samples for uncertainty estimation
)

response = llm.complete("...")
print(f"Uncertainty: {llm.last_uncertainty}")
```

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
export OPENROUTER_API_KEY="your-key"
export SAGE_MODEL="openai/gpt-4o-mini"

# TTS Service (for self-consistency UQ)
export SAGE_USE_TTS=1
export SAGE_TTS_URL="http://localhost:8001/v1"

# Debug flags
export SAGE_DISABLE_LLM_UNCERTAINTY=1
export SAGE_DISABLE_PROPAGATION=1
```

---

## GRPO Training (Certainty-Weighted RL)

Train agents with certainty-weighted rewards from Section 6.2 of the SAGE paper:

```python
from sage_agent import (
    SAGEGRPOTrainer,
    SAGEGRPOConfig,
    ActionType,
    CertaintyWeightedReward,
)

def sage_reward_fn(prompt, response):
    if "book_flight" in response:
        return (1.0, ActionType.TOOL_CALL, 0.85)
    elif "?" in response:
        return (0.5, ActionType.CLARIFICATION, 0.4)
    return (0.0, ActionType.OTHER, 0.5)

trainer = SAGEGRPOTrainer(
    policy=policy_model,
    reference=reference_model,
    reward_fn=sage_reward_fn,
    config=SAGEGRPOConfig(use_certainty_weighting=True),
    optimizer=optimizer,
)

results = trainer.train_epoch(prompts)
```

The certainty-weighted reward encourages:
- Execute tool calls when confident (`Cert = max_c pi_c`)
- Ask questions when uncertain (`Cert = 1 - max_c pi_c`)

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_sage_agent_v1.py -v

# Run with coverage
pytest tests/ --cov=sage_agent --cov-report=html
```

---

## References

```bibtex
@article{sage2024,
  title={Structured Uncertainty Guided Clarification for LLM Agents},
  author={...},
  journal={arXiv preprint arXiv:2511.08798},
  year={2024}
}

@article{selfconsistency2022,
  title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author={Wang et al.},
  journal={arXiv preprint arXiv:2203.11171},
  year={2022}
}

@article{reflexion2023,
  title={Reflexion: Language Agents with Verbal Reinforcement Learning},
  author={Shinn et al.},
  journal={arXiv preprint arXiv:2303.11366},
  year={2023}
}

@article{grpo2024,
  title={Group Relative Policy Optimization},
  author={...},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```

---

## License

MIT License
