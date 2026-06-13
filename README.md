# AegisRank: High-Volume Agentic Retrieval & Behavioral Ranking Pipeline

[![Streamlit App](https://static.streamlit.io/badge/github/streamlit_badge.svg)](https://aegisrank-2sylvkc9vbz2nmpxi6yg7c.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

An enterprise-grade, localized retrieval-augmented talent scoring pipeline engineered to parse, filter, and rank high-volume candidate datasets entirely offline. AegisRank implements a hybrid evaluation strategy that combines dense semantic vector spaces with deterministic behavioral multipliers and strategic disqualification filters to eliminate "keyword-stuffer" profiles, backed by a high-speed local token-matching parser for automated reasoning synthesis.

---
## ✨ Key Features

* Fully Offline Candidate Retrieval & Ranking
* Prebuilt Vector Index for Rapid Evaluation
* LangGraph-Based Candidate Enrichment Pipeline
* Semantic Retrieval using BAAI/bge-small-en-v1.5 Embeddings
* Qdrant Local Vector Database
* Retrieval-Augmented Ranking Architecture
* Multi-Factor Hiring Intelligence Scoring
* Explainable Candidate Recommendations
* Deterministic Ranking Output
* Docker-Friendly Evaluation Workflow
* No External APIs Required During Runtime

---

## 🎯 Why AegisRank?

Traditional semantic candidate retrieval systems frequently over-rank keyword-optimized profiles while under-representing candidates with strong behavioral hiring signals. AegisRank addresses this challenge through a hybrid retrieval-and-ranking architecture.

The system separates candidate discovery from candidate evaluation:

1. **Semantic Retrieval Phase** identifies technically relevant candidates using dense vector similarity.
2. **Behavioral Ranking Phase** evaluates practical hiring readiness through recruiter engagement, availability indicators, assessment performance, experience signals, and profile quality metrics.

This architecture improves ranking robustness against keyword stuffing, profile inflation, and semantic noise while preserving strong recall for highly specialized technical job descriptions.

---

## 🏛️ System Architecture Overview

AegisRank is a fully offline Retrieval-Augmented Candidate Ranking System designed to evaluate nearly 100,000 candidate profiles under strict execution and reproducibility constraints.

```text
+--------------------------------------------------------------------------------+
|                           1. Candidate Ingestion Layer                         |
|      JSONL Parsing -> Schema Validation -> Timeline Auditing -> Enrichment     |
+--------------------------------------+-----------------------------------------+
                                       |
                                       v
+--------------------------------------+-----------------------------------------+
|                       2. Narrative & Telemetry Builder                         |
| Professional Summary + Career History + Skills + Behavioral Telemetry Signals |
+--------------------------------------+-----------------------------------------+
                                       |
                                       v
+--------------------------------------+-----------------------------------------+
|                         3. Dense Embedding Generation                          |
|                  BAAI/bge-small-en-v1.5 -> 384-Dimensional Vectors             |
+--------------------------------------+-----------------------------------------+
                                       |
                                       v
+--------------------------------------+-----------------------------------------+
|                         4. Local Vector Storage Layer                          |
|             Qdrant Local Database (96,796 Indexed Candidate Profiles)          |
+--------------------------------------+-----------------------------------------+
                                       |
                                       v
+--------------------------------------+-----------------------------------------+
|                       5. Semantic Candidate Retrieval                          |
|            Job Description Embedding -> Top 5,000 Candidate Retrieval          |
+--------------------------------------+-----------------------------------------+
                                       |
                                       v
+--------------------------------------+-----------------------------------------+
|                         6. Multi-Factor Ranking Engine                         |
| Similarity + Experience + Availability + Engagement + Assessments + GitHub    |
+--------------------------------------+-----------------------------------------+
                                       |
                                       v
+--------------------------------------+-----------------------------------------+
|                     7. Explainable Recommendation Layer                        |
| Deterministic Candidate Reasoning Generation & Evidence Extraction             |
+--------------------------------------+-----------------------------------------+
                                       |
                                       v
+--------------------------------------+-----------------------------------------+
|                         8. Submission Artifact Layer                           |
| Ranked CSV + Submission Metadata + Recommendation Reasoning                    |
+--------------------------------------------------------------------------------+
```
---
### 🧠 Core Architectural Components

#### Candidate Enrichment Pipeline

Raw candidate profiles are processed through a LangGraph-powered ingestion workflow that performs schema validation, timeline consistency checks, anomaly detection, and candidate enrichment. Profiles are transformed into compact semantic narratives while preserving structured behavioral telemetry.

#### Dense Semantic Retrieval

AegisRank uses the BAAI/bge-small-en-v1.5 embedding model to generate 384-dimensional vector representations. All embeddings are stored locally inside an embedded Qdrant vector database operating completely offline.

#### Retrieval-Augmented Candidate Selection

Instead of ranking the entire candidate corpus directly, the target Job Description is embedded and used to retrieve the top 5,000 semantically relevant candidates from a corpus of 96,796 indexed profiles. This improves recall for highly specialized technical positions.

#### Multi-Factor Ranking Engine

Retrieved candidates are evaluated using a weighted scoring framework that combines:

* Semantic Similarity
* Years of Experience
* Recruiter Engagement Signals
* Candidate Availability Indicators
* GitHub Activity
* Technical Assessment Performance
* Profile Completeness
* Retrieval Evidence Strength
* Interview Completion Behavior
* Offer Acceptance Patterns
* Notice Period Adjustments

#### Explainable Recommendation Generation

The final Top-100 candidates are accompanied by deterministic reasoning generated from verified profile evidence, enabling transparent ranking decisions without requiring external LLM inference during evaluation.

#### Tie-Breaker Resolution Logic: 
To eliminate floating-point noise across indexing runs, final metrics are strictly evaluated up to $4$ decimal points of precision. Ties are deterministically broken by routing indices through multi-key sort constraints: Primary Sort $\rightarrow$ `score` (Descending); Secondary Tie-Breaker $\rightarrow$ `candidate_id` (Alphanumeric Ascending).

#### Contextual Reasoning Generator: 
The top $100$ candidate rows are processed through a fast, local token-matching parser that computes intersections against Job Description keywords. It synthesizes a 1-2 sentence fact-driven narrative evaluating the candidate's technical profile, verified experience, open-source activity levels, and structural gaps without external API network overhead.

---

## 🧰 Technology Stack

| Component              | Technology                 |
| ---------------------- | -------------------------- |
| Application Framework  | Streamlit                  |
| Workflow Orchestration | LangGraph                  |
| Embedding Model        | BAAI/bge-small-en-v1.5     |
| Vector Database        | Qdrant                     |
| Embedding Runtime      | FastEmbed                  |
| Data Processing        | Pandas                     |
| Language               | Python 3.11                |
| Storage                | Local SQLite-backed Qdrant |
| Execution Mode         | Fully Offline              |

---

## 🚦 Execution Constraints & Offline Evaluation Strategy

| Constraint Parameter      | Allocation Profile      | Implementation Strategy                        |
| ------------------------- | ----------------------- | ---------------------------------------------- |
| **Compute Engine**        | Standard Intel/AMD CPU  | Optimized for CPU-only execution               |
| **RAM Threshold**         | Maximum 16GB RAM        | Streaming ingestion and local vector retrieval |
| **Network Context**       | Offline Evaluation      | No internet required during ranking            |
| **Execution Environment** | Docker Sandbox          | Local-first architecture                       |
| **Submission Runtime**    | Under Evaluation Limits | Uses prebuilt vector index for rapid execution |

### Offline Evaluation Design

To satisfy offline execution constraints, AegisRank ships with:

* `clean_candidates.jsonl`
* Prebuilt Local Qdrant Vector Index
* Precomputed Candidate Embeddings

This eliminates multi-hour embedding generation during evaluation and enables direct retrieval and ranking execution immediately after repository setup.

---

## 📊 Dataset Statistics

| Metric                     | Value        |
| -------------------------- | ------------ |
| Raw Candidate Profiles     | 96,796       |
| Indexed Candidate Profiles | 96,796       |
| Embedding Dimensions       | 384          |
| Retrieval Depth            | 5,000        |
| Final Recommendations      | Top 100      |
| Vector Database            | Qdrant Local |


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
|   └── local_qdrant_storage/           # Prebuilt offline Qdrant vector index (96,796 candidate embeddings)
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
## 🔄 End-to-End Evaluation Workflow

```text
Raw Candidate Dataset (96,796 Profiles)
                │
                ▼
      LangGraph Ingestion Pipeline
                │
                ▼
     Narrative & Telemetry Generation
                │
                ▼
      Dense Embedding Generation
                │
                ▼
      Local Qdrant Vector Storage
                │
                ▼
      Job Description Submission
                │
                ▼
      Semantic Retrieval (Top 5,000)
                │
                ▼
      Multi-Factor Ranking Engine
                │
                ▼
      Explainable Candidate Selection
                │
                ▼
      Top 100 Recommendations
                │
                ▼
 CSV + Metadata + Reasoning Output
```
---

## 🛠️ Installation & Reproduction Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Lakshman2405/AegisRank.git
cd AegisRank
```

---

### 2. Create a Python 3.11 Virtual Environment

AegisRank is developed, tested, and validated using **Python 3.11**.

Create the virtual environment explicitly using Python 3.11:

```bash
py -3.11 -m venv venv
```

Activate the environment:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source venv/bin/activate
```

Verify the active Python version:

```bash
python --version
```

Expected Output:

```bash
Python 3.11.x
```

---

### 3. Install Project Dependencies

Upgrade pip and install all required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Verify Offline Assets

The repository includes the preprocessed candidate dataset and prebuilt local vector index required for offline evaluation.

Verify the following assets are present:

```text
data/
├── clean_candidates.jsonl
└── local_qdrant_storage/
    └── collection/
        └── hackathon_talent_pool/
            └── storage.sqlite
```

These assets eliminate the need to regenerate embeddings or rebuild the vector database during evaluation.

---

### 5. Launch AegisRank

Start the application:

```bash
streamlit run src/app.py
```

The application will automatically:

1. Load the prebuilt local Qdrant vector index.
2. Skip embedding generation and re-indexing.
3. Accept a Job Description.
4. Retrieve the Top-5,000 semantically relevant candidates.
5. Execute the multi-factor ranking engine.
6. Generate the required submission artifacts.

---

### 6. Generate Candidate Rankings

1. Open the Streamlit interface.
2. Paste the target Job Description.
3. Click **Run Agentic Pipeline & Compile Submission Pack**.
4. Wait for ranking execution to complete.
5. Retrieve generated outputs from:

```text
submission/
├── team_6a26b0c93c432ee4828a149d.csv
└── submission_metadata.yaml
```

---

### 7. Offline Reproducibility

AegisRank is designed for fully offline execution.

During evaluation:

* No external APIs are called.
* No internet connectivity is required.
* No embedding regeneration is required.
* No vector database reconstruction is required.
* All retrieval and ranking operations execute locally using the included assets.

The repository ships with a prebuilt Qdrant vector index containing **96,796 candidate embeddings**, enabling rapid execution within evaluation constraints.


---

## 📋 Evaluation Assumptions

This repository includes:

* Preprocessed Candidate Dataset (`clean_candidates.jsonl`)
* Prebuilt Local Qdrant Index (`local_qdrant_storage`)
* Submission Generation Pipeline

The evaluation workflow is intended to execute directly using the included repository assets without rebuilding candidate embeddings from scratch.

This design ensures compliance with offline execution requirements and strict evaluation runtime constraints.

---

## 📦 Repository Assets

AegisRank includes all assets required for offline evaluation and reproducible execution.

### Included Assets

| Asset                         | Purpose                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `data/candidates.jsonl`       | Original candidate dataset                                                   |
| `data/clean_candidates.jsonl` | Enriched and validated candidate corpus generated by the ingestion pipeline  |
| `data/local_qdrant_storage/`  | Prebuilt local Qdrant vector database containing 96,796 candidate embeddings |
| `submission/`                 | Generated ranking outputs and metadata artifacts                             |

### Git LFS Requirement

Due to the size of the candidate dataset and vector index, this repository uses **Git Large File Storage (Git LFS)**.

Before cloning the repository, ensure Git LFS is installed and initialized:

```bash
git lfs install
```

Clone the repository normally:

```bash
git clone https://github.com/Lakshman2405/AegisRank.git
```

### Why Prebuilt Assets Are Included

Generating embeddings and constructing a vector index for 96,796 candidate profiles requires significant preprocessing time. To satisfy the competition's offline execution and runtime constraints, AegisRank ships with:

* Preprocessed candidate records
* Precomputed dense embeddings
* Prebuilt Qdrant vector index

This allows evaluation environments to immediately execute retrieval, ranking, and submission generation without rebuilding the vector database from scratch.

### Evaluation Behavior

When the application starts, it automatically detects the existing local vector index and skips embedding generation and indexing operations. The system directly loads the prebuilt candidate repository and proceeds to candidate retrieval and ranking.

This design enables deterministic, fully offline execution while remaining compliant with evaluation time constraints.


---

## 🚀 Local Operation Guide

To run the pipeline and view the execution logs interactively through the Streamlit dashboard workspace, execute:

```bash
streamlit run src/app.py
```

### Compilation Step-by-Step

1. Open `http://localhost:8501` in your web browser.
2. Paste the target requirements description document inside the application configuration UI field.
3. Trigger **Execute Agentic Pipeline & Compile Submission Pack**.
4. The pipeline will query your local vector database storage, apply behavioral adjustments and corporate disqualifiers, resolve scores up to $4$ decimal points, apply secondary alphanumeric tie-breakers, parse contextual reasonings, and generate your submission pack inside the submission/ folder in under 2 minutes total.

---

## ⚖️ AI Disclosures & Declarations

### AI Tools Utilized

* Gemini
* ChatGPT
* Claude Sonnet

### Usage Summary

AI-assisted development tools were utilized as engineering productivity accelerators during architecture design, debugging, documentation generation, code review, optimization, and validation activities.

No candidate ranking decisions are delegated to external LLMs during evaluation. All retrieval, ranking, scoring, candidate selection, telemetry analysis, and recommendation generation execute locally using deterministic algorithms and precomputed vector representations.

Candidate data is never transmitted to external AI services during runtime evaluation. The complete retrieval and ranking workflow executes entirely offline using local resources included within the repository.

---

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for configuration details.
