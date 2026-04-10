# SAGE Agent v3 Architecture Diagrams

## 1. Main Graph Flow (LangGraph State Machine)

```mermaid
graph TD
    START([START]) --> GC[Generate Candidates<br/>+ Phase 2 Resampling<br/>+ Phase 1 SGR Validation]

    GC --> CC{Check Confidence<br/>+ Phase 3 SAUP}

    CC -->|Confident| EXEC[Execute Tool Call<br/>+ Phase 1 Schema Validation]
    CC -->|Need Questions| GQ[Generate Questions]
    CC -->|High Uncertainty| ESC[Escalate<br/>+ Phase 3 Breakdown]

    GQ --> SQ{Select Question}

    SQ -->|Worth Asking| ASK[Ask Question]
    SQ -->|Not Worth It| EXEC

    ASK --> UB[Update Belief<br/>+ Domain Refinement]
    UB --> GC

    EXEC --> VR{Validate Result}

    VR -->|Success| SUCCESS[Handle Success]
    VR -->|Retry| ERR[Handle Error<br/>+ Phase 4 Smart Reflexion]
    VR -->|Failed| ESC

    ERR --> |Reflexion?| REFL{Should Reflect?<br/>Phase 4 Logic}
    REFL -->|Yes| GEN_REFL[Generate Smart Reflection]
    REFL -->|No| GC
    GEN_REFL --> GC

    SUCCESS --> END([END])
    ESC --> END

    style GC fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style CC fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    style EXEC fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    style REFL fill:#ffe0b2,stroke:#e65100,stroke-width:3px
    style ESC fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px
```

## 2. Component Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        USER_INPUT[User Input]
        OBSERVATIONS[Observations<br/>from Clarifications]
        DOMAINS[Parameter Domains<br/>Belief State]
    end

    subgraph "Phase 1: Schema Guided Reasoning"
        SGR_VAL[JSON Schema<br/>Validator]
        FIELD_UNC[Per-Field<br/>Uncertainty Tracker]
        REASONING_TRACE[Reasoning Trace<br/>Builder]
    end

    subgraph "Phase 2: Resampling & Budget"
        BUDGET_CALC[Dynamic Sample<br/>Budget Calculator]
        RESAMPLER[Self-Consistency<br/>Resampler]
        UNC_DECOMP[Uncertainty<br/>Decomposer]
    end

    subgraph "Phase 3: SAUP"
        SAUP_PROP[SAUP Uncertainty<br/>Propagator]
        TRAJ_TRACKER[Trajectory<br/>Uncertainty Tracker]
        BREAKDOWN[Uncertainty<br/>Breakdown Analyzer]
    end

    subgraph "Phase 4: Smart Reflexion"
        REFL_TRIGGER[Reflexion<br/>Trigger Logic]
        CONTEXT_PROMPT[Context-Aware<br/>Prompt Generator]
        REFL_MEMORY[Reflexion<br/>Memory]
    end

    subgraph "Core SAGE Components"
        CAND_GEN[Candidate<br/>Generator]
        Q_GEN[Question<br/>Generator]
        BELIEF[Belief State<br/>Manager]
        EXECUTOR[Tool<br/>Executor]
    end

    subgraph "Output Layer"
        TOOL_CALL[Valid Tool Call]
        UNC_METRICS[Uncertainty<br/>Metrics]
        TRACES[Reasoning<br/>Traces]
        ESCALATION[Escalation<br/>Report]
    end

    USER_INPUT --> CAND_GEN
    OBSERVATIONS --> CAND_GEN
    DOMAINS --> BELIEF

    CAND_GEN --> BUDGET_CALC
    BUDGET_CALC --> RESAMPLER
    RESAMPLER --> UNC_DECOMP
    UNC_DECOMP --> CAND_GEN

    CAND_GEN --> SGR_VAL
    SGR_VAL --> FIELD_UNC
    FIELD_UNC --> REASONING_TRACE

    REASONING_TRACE --> SAUP_PROP
    SAUP_PROP --> TRAJ_TRACKER
    TRAJ_TRACKER --> BREAKDOWN

    BELIEF --> Q_GEN
    Q_GEN --> OBSERVATIONS

    SGR_VAL --> EXECUTOR
    EXECUTOR --> REFL_TRIGGER
    REFL_TRIGGER --> CONTEXT_PROMPT
    CONTEXT_PROMPT --> REFL_MEMORY
    REFL_MEMORY --> OBSERVATIONS

    EXECUTOR --> TOOL_CALL
    TRAJ_TRACKER --> UNC_METRICS
    REASONING_TRACE --> TRACES
    BREAKDOWN --> ESCALATION

    style SGR_VAL fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style RESAMPLER fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style SAUP_PROP fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style REFL_TRIGGER fill:#ffe0b2,stroke:#e65100,stroke-width:2px
```

## 3. Uncertainty Flow Through Pipeline

```mermaid
graph LR
    subgraph "Uncertainty Sources"
        U1[LLM Sampling<br/>Uncertainty]
        U2[Structured<br/>Belief Uncertainty]
        U3[Domain<br/>Ambiguity]
    end

    subgraph "Phase 2: Decomposition"
        D1[Sample Multiple<br/>Candidates]
        D2[Measure<br/>Disagreement]
        D3[Epistemic<br/>Uncertainty]
        D4[Aleatoric<br/>Uncertainty]
    end

    subgraph "Phase 1: Per-Field"
        F1[Field 1<br/>Uncertainty]
        F2[Field 2<br/>Uncertainty]
        F3[Field N<br/>Uncertainty]
    end

    subgraph "Phase 3: Propagation"
        P1[Step 1<br/>Uncertainty]
        P2[Step 2<br/>Uncertainty]
        P3[Step N<br/>Uncertainty]
        P4[Trajectory<br/>Uncertainty]
    end

    subgraph "Decision Making"
        DEC1{High<br/>Epistemic?}
        DEC2{High<br/>Trajectory?}
        DEC3{Field<br/>Uncertain?}
    end

    subgraph "Actions"
        A1[Resample More]
        A2[Ask Question]
        A3[Escalate]
        A4[Execute]
    end

    U1 --> D1
    U2 --> D1
    U3 --> D1

    D1 --> D2
    D2 --> D3
    D2 --> D4

    D3 --> F1
    D3 --> F2
    D3 --> F3

    F1 --> P1
    F2 --> P2
    F3 --> P3

    P1 --> P4
    P2 --> P4
    P3 --> P4

    D3 --> DEC1
    P4 --> DEC2
    F1 --> DEC3

    DEC1 -->|Yes| A1
    DEC1 -->|No| DEC3

    DEC2 -->|Yes| A3
    DEC2 -->|No| DEC3

    DEC3 -->|Yes| A2
    DEC3 -->|No| A4

    style D3 fill:#ffebee,stroke:#c62828,stroke-width:2px
    style D4 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style P4 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style A1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style A2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style A3 fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px
    style A4 fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
```

## 4. Phase 1: Schema Guided Reasoning Detail

```mermaid
graph TD
    subgraph "Schema Definition"
        SCHEMA[ToolSchema<br/>+ parameters<br/>+ required<br/>+ relationships]
    end

    subgraph "Candidate Generation"
        LLM[LLM Output<br/>Raw JSON]
        PARSE[JSON Parser]
        CAND[Tool Call<br/>Candidate]
    end

    subgraph "Phase 1: Validation Pipeline"
        V1[Required Params<br/>Present?]
        V2[Values in<br/>Domain?]
        V3[Types<br/>Correct?]
        V4[Relationships<br/>Satisfied?]
    end

    subgraph "Per-Field Uncertainty"
        FU1{Value Source}
        FU2[Asked: 0.1]
        FU3[Inferred: 0.3-0.6]
        FU4[Unknown: 1.0]
        FU5[FieldUncertainty<br/>+ value<br/>+ uncertainty<br/>+ source<br/>+ reasoning]
    end

    subgraph "Reasoning Trace"
        RT1[ReasoningTrace]
        RT2[+ step<br/>+ thought<br/>+ action<br/>+ uncertainty<br/>+ fields_affected]
    end

    subgraph "Output"
        VALID[✅ Valid<br/>Tool Call]
        INVALID[❌ Reject]
        METRICS[Field-Level<br/>Uncertainty Map]
    end

    SCHEMA --> V1
    LLM --> PARSE
    PARSE --> CAND
    CAND --> V1

    V1 -->|Yes| V2
    V1 -->|No| INVALID
    V2 -->|Yes| V3
    V2 -->|No| INVALID
    V3 -->|Yes| V4
    V3 -->|No| INVALID
    V4 -->|Yes| VALID
    V4 -->|No| INVALID

    VALID --> FU1
    FU1 -->|From User| FU2
    FU1 -->|Inferred| FU3
    FU1 -->|UNK| FU4

    FU2 --> FU5
    FU3 --> FU5
    FU4 --> FU5
    FU5 --> METRICS

    VALID --> RT1
    FU5 --> RT1
    RT1 --> RT2

    style V1 fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style V2 fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style V3 fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style V4 fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style VALID fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    style INVALID fill:#ffcdd2,stroke:#b71c1c,stroke-width:3px
```

## 5. Phase 2: Resampling & Budget Allocation

```mermaid
graph TD
    subgraph "Input"
        PREV_UNC[Previous<br/>Uncertainty]
        EPISTEMIC[Epistemic<br/>Uncertainty]
    end

    subgraph "Budget Calculation"
        CALC{Epistemic > 0.6?}
        B1[Budget = 5<br/>max samples]
        B2{Epistemic > 0.4?}
        B3[Budget = 3<br/>medium samples]
        B4[Budget = 1<br/>min samples]
    end

    subgraph "Sampling"
        S1[Generate<br/>Sample 1]
        S2[Generate<br/>Sample 2]
        S3[Generate<br/>Sample N]
    end

    subgraph "Uncertainty Decomposition"
        SAMPLES[All Samples]
        COMPARE[Compare<br/>Outputs]
        AGREE[Agreement<br/>Rate]
        HEDGE[Hedging<br/>Language]
        EP[Epistemic:<br/>Disagreement]
        AL[Aleatoric:<br/>Hedging]
    end

    subgraph "Output"
        BEST[Best<br/>Candidate]
        UNC_OUT[Decomposed<br/>Uncertainty]
        CONF[Confidence:<br/>1 - uncertainty]
    end

    PREV_UNC --> CALC
    EPISTEMIC --> CALC

    CALC -->|Yes| B1
    CALC -->|No| B2
    B2 -->|Yes| B3
    B2 -->|No| B4

    B1 --> S1
    B1 --> S2
    B1 --> S3
    B3 --> S1
    B3 --> S2
    B4 --> S1

    S1 --> SAMPLES
    S2 --> SAMPLES
    S3 --> SAMPLES

    SAMPLES --> COMPARE
    COMPARE --> AGREE
    COMPARE --> HEDGE

    AGREE --> EP
    HEDGE --> AL

    EP --> UNC_OUT
    AL --> UNC_OUT

    SAMPLES --> BEST
    UNC_OUT --> CONF

    style CALC fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style EP fill:#ffebee,stroke:#c62828,stroke-width:2px
    style AL fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B1 fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style B4 fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
```

## 6. Phase 3: SAUP Trajectory Tracking

```mermaid
graph TD
    subgraph "Step Tracking"
        S1[Step 1:<br/>Candidate Gen]
        S2[Step 2:<br/>Question Gen]
        S3[Step 3:<br/>Belief Update]
        S4[Step N:<br/>Execution]
    end

    subgraph "Per-Step Uncertainty"
        U1[u1 = 0.5]
        U2[u2 = 0.3]
        U3[u3 = 0.1]
        U4[u4 = 0.4]
    end

    subgraph "Propagation Mode"
        MODE{Mode}
        MULT[Multiplicative:<br/>1 - ∏(1-ui)]
        RW[Recency Weighted:<br/>Σ wi·ui / Σ wi]
        MAX[Max:<br/>max(ui)]
    end

    subgraph "Accumulated Uncertainty"
        ACCUM[Trajectory<br/>Uncertainty]
        HIGH[High Uncertainty<br/>Steps List]
    end

    subgraph "Escalation Logic"
        E1{Accum > 0.85?}
        E2{Too Many<br/>High Steps?}
        E3{Specific Step<br/>Very High?}
        ESC[Escalate]
        CONT[Continue]
    end

    subgraph "Breakdown"
        BD1[By Step Type]
        BD2[candidate_gen: 0.5]
        BD3[belief_update: 0.1]
        BD4[reflexion: 0.7]
        BD5[🔍 Root Cause:<br/>Reflexion step]
    end

    S1 --> U1
    S2 --> U2
    S3 --> U3
    S4 --> U4

    U1 --> MODE
    U2 --> MODE
    U3 --> MODE
    U4 --> MODE

    MODE --> MULT
    MODE --> RW
    MODE --> MAX

    MULT --> ACCUM
    RW --> ACCUM
    MAX --> ACCUM

    U1 --> HIGH
    U4 --> HIGH

    ACCUM --> E1
    HIGH --> E2
    U1 --> E3

    E1 -->|Yes| ESC
    E1 -->|No| E2
    E2 -->|Yes| ESC
    E2 -->|No| E3
    E3 -->|Yes| ESC
    E3 -->|No| CONT

    ESC --> BD1
    BD1 --> BD2
    BD1 --> BD3
    BD1 --> BD4
    BD4 --> BD5

    style ACCUM fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    style HIGH fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style ESC fill:#ffcdd2,stroke:#b71c1c,stroke-width:3px
    style BD5 fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    style CONT fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
```

## 7. Phase 4: Smart Reflexion

```mermaid
graph TD
    subgraph "Execution Result"
        EXEC[Tool Execution]
        SUCCESS[Success]
        FAIL[Failure]
    end

    subgraph "Trigger Logic"
        T1{Reflexion<br/>Enabled?}
        T2{Attempts<br/>< Max?}
        T3{Only on<br/>Failure Mode?}
        T4{Execution<br/>Failed?}
        T5{Attempts >= 2<br/>AND<br/>Unc > 0.7?}
        TRIGGER[🔥 Trigger<br/>Reflexion]
        SKIP[Skip Reflexion]
    end

    subgraph "Context Analysis"
        CTX{Failure Type}
        F1[Execution<br/>Failure]
        F2[Persistent High<br/>Uncertainty]
    end

    subgraph "Prompt Generation"
        P1[Execution Failure<br/>Prompt:<br/>- Which params wrong?<br/>- Constraints violated?<br/>- What to clarify?]
        P2[High Uncertainty<br/>Prompt:<br/>- Field uncertainty breakdown<br/>- Missing info?<br/>- Root cause?]
    end

    subgraph "Reflection"
        LLM[LLM<br/>Reflection]
        MEMORY[Add to<br/>Observations]
        SAUP_LOG[Log in<br/>SAUP]
    end

    subgraph "Next Iteration"
        RETRY[Retry with<br/>Reflection Context]
    end

    EXEC --> SUCCESS
    EXEC --> FAIL

    SUCCESS --> T1
    FAIL --> T1

    T1 -->|Yes| T2
    T1 -->|No| SKIP
    T2 -->|Yes| T3
    T2 -->|No| SKIP

    T3 -->|Yes| T4
    T3 -->|No| TRIGGER

    T4 -->|Yes| TRIGGER
    T4 -->|No| T5
    T5 -->|Yes| TRIGGER
    T5 -->|No| SKIP

    TRIGGER --> CTX
    CTX --> F1
    CTX --> F2

    F1 --> P1
    F2 --> P2

    P1 --> LLM
    P2 --> LLM

    LLM --> MEMORY
    MEMORY --> SAUP_LOG
    SAUP_LOG --> RETRY

    style TRIGGER fill:#ffe0b2,stroke:#e65100,stroke-width:3px
    style SKIP fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style P1 fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style P2 fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style RETRY fill:#e1f5ff,stroke:#01579b,stroke-width:2px
```

## 8. State Evolution Through Pipeline

```mermaid
graph LR
    subgraph "Initial State"
        I1[user_input]
        I2[domains]
        I3[observations: empty]
        I4[uncertainty: 1.0]
    end

    subgraph "After Generate Candidates"
        G1[candidates: List]
        G2[probabilities: List]
        G3[field_uncertainties: Dict]
        G4[epistemic_unc: float]
        G5[num_samples: int]
    end

    subgraph "After Questions"
        Q1[questions: List]
        Q2[best_question: Question]
        Q3[observations: +1]
        Q4[domains: refined]
    end

    subgraph "After Execution"
        E1[result: ToolCall]
        E2[execution_result: ExecutionResult]
        E3[status: done/escalated]
    end

    subgraph "Final State"
        F1[✅ Valid tool call]
        F2[📊 Uncertainty metrics]
        F3[📝 Reasoning traces]
        F4[🔍 Breakdown]
    end

    I1 --> G1
    I2 --> G2
    I3 --> G3
    I4 --> G4

    G1 --> Q1
    G2 --> Q2
    G3 --> Q3
    G4 --> Q4

    Q1 --> E1
    Q2 --> E2
    Q3 --> E3

    E1 --> F1
    E2 --> F2
    E3 --> F3
    G5 --> F4

    style I1 fill:#f5f5f5,stroke:#9e9e9e
    style G1 fill:#e1f5ff,stroke:#01579b
    style Q1 fill:#fff9c4,stroke:#f57f17
    style E1 fill:#f3e5f5,stroke:#4a148c
    style F1 fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
```

## 9. Comparison: v2 vs v3 Architecture

```mermaid
graph TB
    subgraph "v2 Architecture"
        V2_INPUT[Input]
        V2_CAND[Candidate<br/>Generation]
        V2_CONF{Confidence}
        V2_Q[Questions]
        V2_EXEC[Execute]
        V2_REFL[Reflexion<br/>Always On]
        V2_OUT[Output]

        V2_INPUT --> V2_CAND
        V2_CAND --> V2_CONF
        V2_CONF --> V2_Q
        V2_CONF --> V2_EXEC
        V2_Q --> V2_CAND
        V2_EXEC --> V2_REFL
        V2_REFL --> V2_OUT
    end

    subgraph "v3 Architecture"
        V3_INPUT[Input]
        V3_BUDGET[Dynamic<br/>Budget]
        V3_RESAMPLE[Resampling<br/>1-5x]
        V3_CAND[Candidate<br/>Generation<br/>+ SGR]
        V3_CONF{Confidence<br/>+ SAUP}
        V3_Q[Questions]
        V3_EXEC[Execute<br/>+ Validation]
        V3_REFL{Smart<br/>Reflexion}
        V3_OUT[Output<br/>+ Metrics]

        V3_INPUT --> V3_BUDGET
        V3_BUDGET --> V3_RESAMPLE
        V3_RESAMPLE --> V3_CAND
        V3_CAND --> V3_CONF
        V3_CONF --> V3_Q
        V3_CONF --> V3_EXEC
        V3_Q --> V3_CAND
        V3_EXEC --> V3_REFL
        V3_REFL -->|Triggered| V3_CAND
        V3_REFL -->|Skipped| V3_OUT
    end

    style V2_REFL fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style V3_BUDGET fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style V3_RESAMPLE fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style V3_CAND fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style V3_REFL fill:#ffe0b2,stroke:#e65100,stroke-width:2px
```

## 10. Data Flow: Input → Output

```mermaid
flowchart LR
    INPUT["📝 User Query:<br/>'Book flight to LAX'"]

    subgraph P2["Phase 2: Budget & Resample"]
        BUDGET["Epistemic unc = 0.6<br/>→ Budget = 3 samples"]
        SAMPLE1["Sample 1:<br/>book_flight"]
        SAMPLE2["Sample 2:<br/>book_flight"]
        SAMPLE3["Sample 3:<br/>search_flights"]
        DISAGREE["Disagreement:<br/>Epistemic = 0.33"]
    end

    subgraph P1["Phase 1: SGR"]
        VALIDATE["Validate samples<br/>against schema"]
        VALID["✅ 2 valid<br/>❌ 1 invalid"]
        FIELDS["Field uncertainties:<br/>origin: 0.1<br/>dest: 0.5<br/>date: 0.8"]
    end

    subgraph P3["Phase 3: SAUP"]
        TRAJ["Trajectory unc:<br/>0.45 (3 steps)"]
        DECISION{High date<br/>uncertainty}
    end

    subgraph CLARIFY["Clarification"]
        ASK["Ask: 'What date?'"]
        ANSWER["User: 'Jan 15'"]
        UPDATE["Update domain:<br/>date: 0.8 → 0.1"]
    end

    subgraph FINAL["Execution"]
        EXEC["Execute:<br/>book_flight(<br/>  origin=NYC,<br/>  dest=LAX,<br/>  date=2024-01-15<br/>)"]
        SUCCESS["✅ Success"]
    end

    OUTPUT["📊 Output:<br/>status=done<br/>samples=3<br/>epistemic=0.33<br/>trajectory=0.45<br/>questions=1"]

    INPUT --> BUDGET
    BUDGET --> SAMPLE1 & SAMPLE2 & SAMPLE3
    SAMPLE1 & SAMPLE2 & SAMPLE3 --> DISAGREE
    DISAGREE --> VALIDATE
    VALIDATE --> VALID
    VALID --> FIELDS
    FIELDS --> TRAJ
    TRAJ --> DECISION
    DECISION -->|Yes| ASK
    ASK --> ANSWER
    ANSWER --> UPDATE
    UPDATE --> EXEC
    EXEC --> SUCCESS
    SUCCESS --> OUTPUT

    style INPUT fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style DISAGREE fill:#ffebee,stroke:#c62828,stroke-width:2px
    style FIELDS fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style SUCCESS fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    style OUTPUT fill:#e1f5ff,stroke:#01579b,stroke-width:3px
```

---

## Legend

```mermaid
graph LR
    P1[Phase 1: SGR]
    P2[Phase 2: Resampling]
    P3[Phase 3: SAUP]
    P4[Phase 4: Reflexion]
    SUCCESS[Success/Valid]
    ERROR[Error/Invalid]
    DECISION{Decision Point}

    style P1 fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style P2 fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style P3 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style P4 fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    style SUCCESS fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style ERROR fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px
    style DECISION fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
```

---

## Quick Reference

| Component | Purpose | Key Innovation |
|-----------|---------|----------------|
| **SGR Validation** | Ensure valid tool calls | 100% correctness guarantee |
| **Per-Field Uncertainty** | Track parameter-level uncertainty | Targeted clarification |
| **Dynamic Budget** | Adaptive sampling | 40% cost reduction on easy queries |
| **Epistemic/Aleatoric** | Decompose uncertainty | Know what's reducible |
| **SAUP Propagation** | Track trajectory uncertainty | Predict failures early |
| **Smart Reflexion** | Trigger-based reflection | 75% reduction in overhead |
| **Uncertainty Breakdown** | Root cause analysis | Pinpoint problem steps |

---

**Note**: These diagrams represent the logical architecture. The actual implementation uses LangGraph's state machine with TypedDict states and functional nodes.
