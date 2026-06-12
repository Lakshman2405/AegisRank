# AegisRank: High-Volume Agentic Retrieval & Behavioral Ranking Pipeline

[![Streamlit App](https://static.streamlit.io/badge/github/streamlit_badge.svg)](https://aegisrank-2sylvkc9vbz2nmpxi6yg7c.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

An enterprise-grade, localized retrieval-augmented talent scoring pipeline engineered to parse, filter, and rank high-volume candidate datasets entirely offline. AegisRank implements a hybrid evaluation strategy that combines dense semantic vector spaces with deterministic behavioral multipliers and strategic disqualification filters to eliminate "keyword-stuffer" profiles, backed by a high-speed local token-matching parser for automated reasoning synthesis.

---

## 🏛️ System Architecture Overview

The system architecture decouples data ingestion, dense retrieval, mathematical heuristic engineering, and generative reasoning into structured, atomic modules.

```
+--------------------------------------------------------------------------+
|                        1. Data Ingestion Layer                           |
|      Optimized Streaming Engine -> JSONL Parsing -> Base Data Cleaning   |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------+
|                      2. Hybrid Vector Indexing                           |
|    FastEmbed (BAAI/bge-small-en-v1.5) -> 384-Dim Dense Cosine Vectors   |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------+
|                         3. In-Process Storage Vector Matrix              |
|         100% Offline Embedded Database Storage via Qdrant Local Engine   |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------+
|                     4. Behavioral Scopes & Disqualifiers                 |
|   Heuristic Multipliers (YoE + GitHub) + Targeted Exclusions (Consulting)|
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------+
|                     5. Deterministic Tie-Breaking                        |
|   Float Rounding (4 Decimals) -> Multi-Key Sort: Score(Desc) + ID(Asc)  |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------+
|                       6. Local Synthesis Layer                           |
|   Rule-Based Token Context Match Engine -> Context-Bound Fact Generation |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------------------+
|                        7. Regulated Artifacts                                        |
|submission/team_6a26b0c93c432ee4828a149d.csv  |  submission/submission_metadata.yaml  |
+--------------------------------------------------------------------------------------+
```

### 🧠 Core Architectural Components

1. **Embedding Processing Layer**: Utilizing `FastEmbed` loaded with the `BAAI/bge-small-en-v1.5` framework, producing high-performance 384-dimensional dense vectors calculated locally on CPU.
2. **Vector Space Topology**: Initialized entirely in local mode (`path="data/local_qdrant_storage"`), running an in-process thread inside the application. Spatial matching leverages standard Cosine Similarity matrices to isolate technical candidate traits completely air-gapped.
3. **Behavioral Optimization & Disqualification Engines**: Rather than relying strictly on raw semantic similarity (which is susceptible to text-manipulation traps), candidate scores are adjusted using mathematical telemetry modifiers and explicit corporate profile penalties:

$$\text{Final Score} = \text{round}\Big(\big((\text{Base Similarity} \times 0.50) + (\text{YoE Modifier} \times 0.25) + (\text{GitHub Modifier} \times 0.25)\big) \times \text{Multiplier}, 4\Big)$$

   - *Years of Experience (YoE) Modifier*: Extracted linearly up to a 10-year normalization index cap ($\min(\text{YoE}, 10) / 10.0$).
   - *GitHub Activity Modifier*: Maps telemetry logs directly against a base $100$-point performance vector scale.
   - *Strategic Disqualification Multipliers*: Automatically penalizes profiles matching explicit Job Description exclusions to prioritize product-focused backgrounds:
     - Large-scale IT Consulting firms (e.g., TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) $\rightarrow$ **Multiplied by 0.45**
     - Pure academic research backgrounds lacking production experience $\rightarrow$ **Multiplied by 0.40**
     - Low engagement/zero platform presence (0 search appearances in 30 days) $\rightarrow$ **Multiplied by 0.85**

4. **Tie-Breaker Resolution Logic**: To eliminate floating-point noise across indexing runs, final metrics are strictly evaluated up to $4$ decimal points of precision. Ties are deterministically broken by routing indices through multi-key sort constraints: Primary Sort $\rightarrow$ `score` (Descending); Secondary Tie-Breaker $\rightarrow$ `candidate_id` (Alphanumeric Ascending).
5. **Contextual Reasoning Generator**: The top $100$ candidate rows are processed through a fast, local token-matching parser that computes intersections against Job Description keywords. It synthesizes a 1-2 sentence fact-driven narrative evaluating the candidate's technical profile, verified experience, open-source activity levels, and structural gaps without external API network overhead.

---

## 🚦 Hard Execution Constraints

The platform is explicitly optimized to fit within the restrictive runtime and compute boundaries established by the Stage 3 grading framework:

| Constraint Parameter | Allocation Profile | Implementation Matrix |
| :--- | :--- | :--- |
| **Compute Engine** | Standard Intel/AMD U-Series CPU | 100% Core Optimized execution paths; no local dedicated GPU required. |
| **RAM Threshold** | Maximum 16GB Allocation | Lazy-loading, streaming data iterators, and chunked vector payloads prevent memory leaks. |
| **Network Context** | Total Air-Gapped during Ranking | **100% Offline.** All vector evaluations, sorting arrays, and reasoning generations happen without external API calls. |
| **Time Threshold** | Maximum 5 Minutes | Complete 96K pipeline index scan and submission asset compilation executes in **under 2 minutes**. |

---

## 📂 Project Directory Structure

```text
AegisRank/
│
├── .gitignore                          # Strict file masking constraints (ignores local environment caches)
├── README.md                           # Comprehensive production documentation and execution blueprint
├── requirements.txt                    # Explicit versioned architecture dependency trees
|
│── data/
│   |── candidates.jsonl                # Raw data candidate list
|   |── clean_candidates.jsonl          # Cleaned data candidate list
|   |── job_description.docx
|   |── sample_candidates.json
|   └── local_qdrant_storage/           # Pre-computed, local 96,796 candidate vector database index
|
├── src/                                # Main Core Application Directory
│   ├── app.py                          # Streamlit UI Dashboard Interface
│   │── main.py
│   └── agents/
│       |── retrieval_agent.py          # Local vector engine, deterministic multi-key sorter, and file writers
|       |── ingestion_agent.py          # Structural schemas, pattern guards, and timeline chronology auditors
│
└── submission/                         # Target Submission Output Folder
    ├── team_6a26b0c93c432ee4828a149d.csv                  # Standardized multi-key sorted ranking artifact table
    └── submission_metadata.yaml        # Schema-matched metadata confirmation log file
```

---

## 🛠️ Installation & Reproduction Setup

### 1. Environment Isolation

Ensure your local environment is running Python 3.11+. Open a terminal inside the root directory and initialize your environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate
```

### 2. Dependency Ingestion

Install all system dependencies defined within the tracking configuration manifest:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Local Operation Guide

To run the pipeline and view the execution logs interactively through the Streamlit dashboard workspace, execute:

```bash
streamlit run src/app.py
```

### Compilation Step-by-Step

1. Open `http://localhost:8501` in your web browser.
2. Paste the target requirements description document inside the application configuration UI field.
3. Trigger **Run Agentic Pipeline & Compile Submission Pack**.
4. The pipeline will query your local vector database storage, apply behavioral adjustments and corporate disqualifiers, resolve scores up to $4$ decimal points, apply secondary alphanumeric tie-breakers, parse contextual reasonings, and generate your submission pack inside the submission/ folder in under 2 minutes total.

---

## ⚖️ AI Disclosures & Declarations

As dictated by the strict transparency directives of the competition, we declare the use of advanced AI tooling across our engineering cycles. These interactions were audited against codebase patterns to protect the underlying scoring logic.

### AI Tools Utilized

- **Gemini**: Algorithmic optimization of pandas dataframe operations, local vector indexing alignment scripts, structural layout pattern guards, and mathematical filtering logic to handle deterministic tie resolutions without altering core system variables.

### Usage Summary

AI tools were strictly treated as paired-programming extensions to increase delivery speeds and verify code quality. Throughout the entire development lifecycle, zero candidate data or profile information was exposed or uploaded to external hosted LLM tokenizers during the ranking step. All telemetry assessments and mathematical scores were processed using deterministic local computing matrices.

---

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for configuration details.
