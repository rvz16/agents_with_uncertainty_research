# SAGE-Agent

**Structured Uncertainty Guided Clarification for LLM Agents**

A Python implementation of the SAGE-Agent method from the paper [arXiv:2511.08798](https://arxiv.org/abs/2511.08798).

SAGE-Agent is a framework for building LLM-powered tool-calling agents that know when to ask clarifying questions instead of making uncertain tool calls.

## Key Insight

Traditional LLM agents often make tool calls with incomplete or ambiguous information, leading to errors. SAGE-Agent models tool calling as a POMDP (Partially Observable Markov Decision Process) and uses **belief state tracking** to quantify uncertainty, asking targeted clarification questions when uncertainty is high.

```
User: "Book me a flight to NYC"

Traditional Agent: book_flight(origin=???, dest="NYC", date=???)  <- Guesses or fails

SAGE-Agent: "Which city will you be departing from?"
User: "Boston"
SAGE-Agent: "When would you like to travel?"
User: "March 15th"
SAGE-Agent: book_flight(origin="BOS", dest="NYC", date="2024-03-15")  <- Confident execution
```

## Features

- **Belief State Tracking**: Maintain probability distributions over tool call candidates
- **EVPI-based Question Selection**: Choose questions that maximize expected information gain
- **Structured Uncertainty**: Quantify uncertainty from parameter domain constraints
- **Paper-Accurate Implementation**: Exact Algorithm 1 with hyperparameters tau=0.85, alpha=0.1, lambda=0.5
- **LangGraph Integration**: Graph-based execution with state management
- **Multiple LLM Backends**: OpenRouter, Ollama, TTS service support
- **Benchmark Evaluation**: When2Call, ClarifyBench, HumanEval, GSM8K

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

# GRPO training
pip install -e ".[training]"

# All dependencies
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

### External Requirements

For running evaluations, you'll need:

```bash
pip install datasets  # For When2Call, HumanEval, etc.
```

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
        "date": ParameterDomain.continuous(),  # Any date string
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
        tau_exec=0.85,      # Execute when 85% confident
        alpha=0.1,          # Stop asking when score < 10% of max_prob
        lambda_redundancy=0.5,
        max_questions=6,
    ),
)

# Run the agent
result = agent.run("Book me a flight to NYC")

print(f"Tool call: {result.tool_call}")
print(f"Questions asked: {result.total_questions}")
print(f"Confidence: {result.final_probability:.2%}")
```

### 2. LangGraph Implementation

```python
from examples.langgraph_sage_agent import (
    GraphDeps,
    SAGEConfig,
    build_graph,
    create_initial_state,
)
from sage_agent import (
    LLMBackedCandidateGenerator,
    LLMBackedQuestionGenerator,
    SimpleConstraintExtractor,
    ToolRegistryExecutor,
)

# Setup LLM client
from examples.openrouter_client import OpenRouterClient
llm = OpenRouterClient(model="openai/gpt-4o-mini")

# Build graph
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

## Algorithm 1: SAGE-Agent Decision Loop

```
1: Initialize belief B(0), aspect counts n(0), t <- 0
2: while t < T_max do
3:     Generate candidates C, compute pi_c(t) proportional to Pi_c(t)
4:     if max_c pi_c(t) >= tau_exec then
5:         Execute argmax_c pi_c(t) and return
6:     end if
7:     Generate questions Q
8:     For each q in Q: Score(q) = EVPI(q) - Cost(q)
9:     Select q* = argmax_q Score(q)
10:    if Score(q*) < alpha * max_c pi_c(t) then
11:        Execute (cost exceeds benefit)
12:    end if
13:    Ask q*, update domains, t++
14: end while
15: Final execution or escalation
```

## Core Concepts

### Belief State (Section 3.2)

The belief state tracks probability distributions over tool call candidates:

```
Pi_i(t) proportional to product_j p(theta_{i,j} | observations_t)
```

Where parameter certainty is:
- **1.0** if value is specified
- **1/|D|** if unspecified with finite domain D
- **epsilon** (1e-4) if domain is infinite/continuous

### EVPI - Expected Value of Perfect Information

```
EVPI(q) = E_r[max_c pi_c(t|q,r)] - max_c pi_c(t)
```

Questions are scored by information gain minus redundancy cost:
```
Score(q) = EVPI(q) - lambda * sum_{a in A(q)} n_a(t)
```

### Termination Conditions

1. **Confident**: `max_c pi_c(t) >= tau_exec` (default 0.85)
2. **Cost exceeds benefit**: `Score(q*) < alpha * max_c pi_c(t)`
3. **Max questions**: `t >= T_max` (default 6)

## Project Structure

```
agents_with_uncertainty_research/
|-- sage_agent/                    # Main package
|   |-- core/                      # Core algorithms
|   |   |-- sage_algorithm.py      # SAGEAgent (Algorithm 1)
|   |   |-- belief.py              # BeliefState
|   |   |-- evpi.py                # EVPI computation
|   |   |-- domains.py             # ParameterDomain
|   |   |-- types.py               # Type definitions
|   |   |-- pomdp.py               # POMDP formulation
|   |   |-- constraints.py         # Constraint extractors
|   |   |-- uncertainty_propagation.py  # SAUP
|   |   +-- advanced_reasoning.py  # Decomposition, CoT, Reflexion
|   |-- langgraph/                 # LangGraph integration
|   |   +-- sage_graph.py          # Canonical LangGraph impl
|   |-- training/                  # GRPO training
|   |   +-- grpo.py                # Certainty-weighted GRPO
|   |-- llm/                       # LLM clients
|   |   |-- openrouter.py
|   |   +-- tts_service.py
|   |-- wiring/                    # LLM-backed generators
|   |   +-- wiring.py
|   |-- metrics/                   # Evaluation metrics
|   |   +-- metrics.py
|   +-- sim/                       # Simulation
|       +-- clarifybench.py
|
|-- examples/                      # Example scripts
|   |-- langgraph_sage_agent.py    # Pure SAGE (clean)
|   |-- langgraph_sage_agent_experimental.py  # With SAUP, reflexion
|   |-- langgraph_sage_agent_v3.py # SGR, resampling, SAUP
|   |-- run_sage_eval.py           # Benchmark evaluation
|   |-- run_when2call_eval.py      # When2Call evaluation
|   |-- run_multi_benchmark_eval.py # Multi-benchmark
|   |-- basic_usage.py             # Simple example
|   |-- grpo_training.py           # GRPO training example
|   +-- openrouter_client.py       # OpenRouter LLM client
|
+-- pyproject.toml                 # Package configuration
```

## Available Implementations

| Implementation | File | Features |
|---------------|------|----------|
| **Pure SAGE** | `examples/langgraph_sage_agent.py` | Clean Algorithm 1, paper hyperparameters |
| **Experimental** | `examples/langgraph_sage_agent_experimental.py` | + SAUP, reflexion, LLM uncertainty |
| **V3 (Full)** | `examples/langgraph_sage_agent_v3.py` | + SGR, resampling, epistemic/aleatoric decomposition |
| **Canonical** | `sage_agent/langgraph/sage_graph.py` | Package-level LangGraph implementation |
| **Core** | `sage_agent/core/sage_algorithm.py` | Non-LangGraph SAGEAgent class |

## Running Evaluations

### When2Call Benchmark

```bash
# Pure SAGE evaluation
python examples/run_sage_eval.py --dataset when2call --limit 20 --print-each

# With OpenRouter
python examples/run_sage_eval.py --dataset when2call --use-openrouter --model openai/gpt-4o-mini

# With Ollama
python examples/run_sage_eval.py --dataset when2call --use-ollama --ollama-model qwen2.5:7b-instruct

# Custom hyperparameters
python examples/run_sage_eval.py --dataset when2call --tau 0.9 --alpha 0.05 --max-questions 4
```

### Experimental Features (V3)

```bash
# V3 with full features
python examples/run_when2call_eval.py --use-v3 --v3-config balanced --limit 10 --print-each

# V3 configurations: conservative, balanced, aggressive
python examples/run_when2call_eval.py --use-v3 --v3-config aggressive --limit 50
```

### Multi-Benchmark

```bash
# HumanEval
python examples/run_multi_benchmark_eval.py --benchmark humaneval --limit 10

# GSM8K
python examples/run_multi_benchmark_eval.py --benchmark gsm8k --limit 20

# All benchmarks
python examples/run_multi_benchmark_eval.py --benchmark all --limit 5
```

## Configuration

### Paper Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `tau_exec` | tau | 0.85 | Execution confidence threshold |
| `alpha` | alpha | 0.1 | Termination threshold factor |
| `lambda_redundancy` | lambda | 0.5 | Redundancy weight |
| `epsilon` | epsilon | 1e-4 | Small prob for infinite domains |
| `max_questions` | T_max | 6 | Maximum clarification questions |

### Environment Variables

```bash
# LLM Configuration
export OPENROUTER_API_KEY="your-key"
export SAGE_MODEL="openai/gpt-4o-mini"

# TTS Service
export SAGE_USE_TTS=1
export SAGE_TTS_URL="http://localhost:8001/v1"

# Debug
export SAGE_DISABLE_LLM_UNCERTAINTY=1
export SAGE_DISABLE_PROPAGATION=1
```

## LLM Backends

### OpenRouter

```python
from examples.openrouter_client import OpenRouterClient

llm = OpenRouterClient(
    model="openai/gpt-4o-mini",
    api_key="your-key",  # or set OPENROUTER_API_KEY
)
```

### Ollama

```python
from examples.ollama_client import OllamaClient

llm = OllamaClient(model="qwen2.5:7b-instruct")
```

### TTS Service (with uncertainty)

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

## Metrics

The evaluation scripts compute:

| Metric | Description |
|--------|-------------|
| **Tool Match Rate** | Fraction of correct tool names |
| **Parameter Match Rate** | Fraction of correct parameter values |
| **Coverage Rate** | Fraction of non-escalated predictions |
| **Avg Questions** | Average clarification questions asked |
| **ECE** | Expected Calibration Error |
| **Confident Accuracy** | Accuracy on low-uncertainty predictions |

## GRPO Training (Section 6.2)

Train agents with certainty-weighted rewards:

```python
from sage_agent import (
    SAGEGRPOTrainer,
    SAGEGRPOConfig,
    ActionType,
    CertaintyWeightedReward,
)

# Reward function returns (base_reward, action_type, max_prob)
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

## Citation

```bibtex
@article{sage2024,
  title={Structured Uncertainty Guided Clarification for LLM Agents},
  author={...},
  journal={arXiv preprint arXiv:2511.08798},
  year={2024}
}
```

## License

MIT License
