# SAGE-Agent Architecture Diagrams

## 1. Main Agent Flow (Complete)

```mermaid
flowchart TD
    subgraph INPUT
        A[👤 User Request]
    end

    subgraph ANALYZE["1️⃣ ANALYZE PROBLEM"]
        B[Parse Request]
        B --> C[Generate Multiple<br/>Interpretations]
        C --> D[Calculate Structured<br/>Uncertainty]
        D --> E{Uncertainty > 0.3?}
    end

    subgraph CLARIFY["2️⃣ CLARIFY (if uncertain)"]
        F[Generate Clarifying<br/>Question using EVPI]
        F --> G[Ask User]
        G --> H[Get User Response]
        H --> I[Update Belief State]
        I --> J[Refine Domain<br/>Constraints]
        J --> K{Still Uncertain?}
    end

    subgraph GENERATE["3️⃣ GENERATE CODE"]
        L[Select Best<br/>Interpretation]
        L --> M[Generate Code<br/>with LLM]
        M --> N[Get LLM Uncertainty<br/>from TTS Service]
    end

    subgraph VERIFY["4️⃣ VERIFY (Enhanced)"]
        O[Chain-of-Thought<br/>Verification]
        O --> P{CoT Errors<br/>Found?}
        P -->|Yes| Q[Mark Issues]
        P -->|No| R[Continue]
        Q --> R
        R --> S[SAUP Uncertainty<br/>Decomposition]
        S --> T[Get Multiple<br/>Verification Samples]
        T --> U[Compute Epistemic<br/>& Aleatoric]
        U --> V[LLM Code Review]
        V --> W{All Checks<br/>Passed?}
    end

    subgraph REFINE["5️⃣ REFINE (Reflexion)"]
        X[Add to Reflexion<br/>History]
        X --> Y[Build Prompt with<br/>Previous Failures]
        Y --> Z[Regenerate Code<br/>with Lessons]
        Z --> AA{Max Retries<br/>Reached?}
    end

    subgraph OUTPUT
        BB[✅ Return Solution]
        CC[⚠️ Escalate to Human]
    end

    A --> B
    E -->|Yes| F
    E -->|No| L
    K -->|Yes, max not reached| F
    K -->|No or max reached| L
    N --> O
    W -->|Yes| BB
    W -->|No| X
    AA -->|No| O
    AA -->|Yes| BB

    subgraph PROPAGATOR["📊 Uncertainty Chain"]
        direction LR
        PA[Step 1<br/>unc=0.2]
        PB[Step 2<br/>unc=0.3]
        PC[Step 3<br/>unc=0.2]
        PD[Accumulated<br/>unc=0.55]
        PA --> PB --> PC --> PD
    end
```

## 2. Uncertainty Calculation Flow

```mermaid
flowchart LR
    subgraph SOURCES["Uncertainty Sources"]
        A[🔄 LLM-TTS Service<br/>Multiple Traces]
        B[📊 Structured<br/>Interpretation Weights]
        C[🧠 SAUP<br/>Sample Decomposition]
    end

    subgraph CALCULATION["Calculation"]
        D[LLM Uncertainty<br/>= Disagreement Rate]
        E[Structured Uncertainty<br/>= 1 - max weight]
        F[Epistemic<br/>= Sample Variance]
        G[Aleatoric<br/>= Hedging Language]
    end

    subgraph COMBINE["Combined"]
        H[Combined =<br/>0.7×Structured + 0.3×LLM]
        I{Combined > 0.5?}
    end

    subgraph ACTION["Action"]
        J[✅ Execute]
        K[❓ Ask Question]
    end

    A --> D
    B --> E
    C --> F
    C --> G
    D --> H
    E --> H
    H --> I
    I -->|No| J
    I -->|Yes| K
```

## 3. Verification Pipeline (Enhanced)

```mermaid
flowchart TD
    A[Generated Code] --> B[Chain-of-Thought Verifier]
    
    subgraph COT["CoT Verification"]
        B --> C[Extract Reasoning Steps]
        C --> D[Check Each Step]
        D --> E{Arithmetic<br/>Errors?}
        E -->|15+27=41 ❌| F[Mark Error]
        E -->|Correct| G[Continue]
    end

    subgraph SAUP["SAUP Decomposition"]
        H[Get 3 Verification<br/>Samples from LLM]
        H --> I[Sample 1: PASS]
        H --> J[Sample 2: PASS]
        H --> K[Sample 3: FAIL]
        I --> L[Compute Disagreement]
        J --> L
        K --> L
        L --> M[Epistemic = 0.33<br/>Model Uncertain]
        L --> N[Aleatoric = 0.1<br/>Low Ambiguity]
    end

    subgraph LLM_REVIEW["LLM Review"]
        O[Full Code Review<br/>by LLM]
        O --> P[Check Requirements]
        O --> Q[Check Edge Cases]
        O --> R[Check Logic]
    end

    F --> S{Final Decision}
    G --> H
    M --> S
    N --> S
    P --> S
    Q --> S
    R --> S

    S -->|All Pass| T[✅ Verified]
    S -->|Issues Found| U[🔄 Refine with Reflexion]
```

## 4. Reflexion Self-Improvement Loop

```mermaid
flowchart TD
    A[Initial Code<br/>Attempt 1] --> B{Verification<br/>Passed?}
    
    B -->|Yes| C[✅ Done]
    B -->|No| D[Reflect on Failure]
    
    D --> E[Add to History:<br/>'Attempt 1 failed:<br/>missing edge case']
    
    E --> F[Generate Improved Code<br/>Attempt 2]
    F --> G[Include in Prompt:<br/>'Previous failures...']
    
    G --> H{Verification<br/>Passed?}
    
    H -->|Yes| C
    H -->|No| I[Reflect Again]
    
    I --> J[Add to History:<br/>'Attempt 2 failed:<br/>wrong logic']
    
    J --> K[Generate Final Code<br/>Attempt 3]
    K --> L[Include ALL<br/>lessons learned]
    
    L --> M{Verification<br/>Passed?}
    
    M -->|Yes| C
    M -->|No, max reached| N[Return Best Attempt]
```

## 5. Complete State Machine

```mermaid
stateDiagram-v2
    [*] --> Analyzing: User Request
    
    Analyzing --> Clarifying: High Uncertainty
    Analyzing --> Generating: Low Uncertainty
    
    Clarifying --> Clarifying: Still Uncertain\n& Questions Left
    Clarifying --> Generating: Confident or\nMax Questions
    
    Generating --> Verifying: Code Generated
    
    Verifying --> Done: All Checks Pass
    Verifying --> Refining: Issues Found\n& Retries Left
    Verifying --> Done: Max Retries
    
    Refining --> Verifying: Refined Code
    
    Done --> [*]: Return Solution

    note right of Analyzing
        • Parse interpretations
        • Calculate uncertainty
        • Chain propagator: observe()
    end note

    note right of Clarifying
        • EVPI question selection
        • Update belief state
        • Refine domains
    end note

    note right of Verifying
        • CoT step verification
        • SAUP decomposition
        • LLM code review
    end note

    note right of Refining
        • Reflexion history
        • Learn from mistakes
        • Improve prompt
    end note
```

## 6. Uncertainty Chain Propagation

```mermaid
flowchart LR
    subgraph Step1["Step 1: Analyze"]
        A1[unc = 0.2]
    end
    
    subgraph Step2["Step 2: Clarify"]
        A2[unc = 0.3]
    end
    
    subgraph Step3["Step 3: Generate"]
        A3[unc = 0.2]
    end
    
    subgraph Step4["Step 4: Verify"]
        A4[unc = 0.4]
    end
    
    subgraph Accumulated["Accumulated"]
        B1[0.20]
        B2[0.44]
        B3[0.55]
        B4[0.73]
    end
    
    A1 --> B1
    B1 --> A2
    A2 --> B2
    B2 --> A3
    A3 --> B3
    B3 --> A4
    A4 --> B4
    
    B4 --> C{> 0.8?}
    C -->|Yes| D[🚨 Escalate]
    C -->|No| E[✅ Continue]

    style B4 fill:#ff6b6b
    style D fill:#ff6b6b
```

## 7. Component Integration

```mermaid
flowchart TB
    subgraph External["External Services"]
        TTS[LLM-TTS Service<br/>Uncertainty Estimation]
        LLM[Base LLM<br/>Ollama / OpenRouter]
    end

    subgraph Core["SAGE-Agent Core"]
        Agent[SageAgent]
        Belief[BeliefState]
        EVPI[EVPI Calculator]
        Propagator[UncertaintyPropagator]
    end

    subgraph Advanced["Advanced Reasoning"]
        Decomposer[UncertaintyDecomposer<br/>SAUP]
        CoT[ChainOfThoughtVerifier]
        Reflexion[ReflexionAgent]
    end

    subgraph Graph["LangGraph Flow"]
        Analyze[analyze_node]
        Clarify[clarify_node]
        Generate[generate_node]
        Verify[verify_node]
        Refine[refine_node]
    end

    TTS --> Agent
    LLM --> Agent
    
    Agent --> Belief
    Agent --> EVPI
    Agent --> Propagator
    
    Verify --> Decomposer
    Verify --> CoT
    Refine --> Reflexion
    
    Analyze --> Clarify
    Clarify --> Generate
    Generate --> Verify
    Verify --> Refine
    Refine --> Verify
```

---

## How to View These Diagrams

1. **GitHub**: GitHub renders Mermaid automatically
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Paste code at https://mermaid.live
4. **Export**: Use Mermaid CLI to export as PNG/SVG


