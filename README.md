# AI Exam Study Planner

A multi-agent RAG system that creates personalized study plans from your course syllabi and textbooks.

Built with **Google ADK**, **Gemini 2.5**, and **Pinecone**.

---

## How It Works

```
Syllabi + Textbooks → AI Agents → Personalized Study Plan
```

**Six agents work together:**

1. **Orchestrator** — Collects your study preferences via natural language
2. **Researcher** — Retrieves relevant textbook content for each topic
3. **Validator** — Checks coverage, triggers retry if content is missing
4. **Estimator** — Calculates study time, lets you adjust
5. **Planner** — Creates a day-by-day schedule
6. **Study Guide** — Generates detailed notes with definitions, formulas, and practice questions

The Researcher and Validator form a **generator-critic loop** — if the validator finds gaps, the researcher searches again with targeted queries.

---

## Quick Start

### 1. Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=study-planner
EMBEDDING_GEMINI_API_KEY=your_key
RETRIEVAL_GEMINI_API_KEY=your_key
```

### 2. Prepare Data (run once)

```bash
python -m src.tools.parser           # Parse syllabi → topics
python -m src.ingestion.chunking     # Chunk textbooks
python -m src.ingestion.embedder     # Generate embeddings
python -m src.ingestion.indexer      # Upload to Pinecone
```

### 3. Run

```bash
python main.py
```

**What happens:**
1. Tell the system about your courses ("Physics is hard, exam in 5 days")
2. Watch it retrieve and validate content
3. Review time estimates, adjust if needed
4. Get your study plan in `study_plan.md`

---

## Project Structure

```
├── main.py                    # Entry point
├── src/
│   ├── agents/                # The 6 AI agents
│   ├── ingestion/             # Chunking, embedding, indexing
│   ├── tools/                 # RAG client, parser, utilities
│   ├── models/schema.py       # Data models
│   └── config.py              # Settings
├── data/
│   ├── Syllabi/               # Your course syllabi (PDF)
│   ├── Textbooks/             # Your textbooks (PDF)
│   └── processed/             # Generated chunks & embeddings
└── study_plan.md              # Output
```

---

## Key Design Decisions

- **Parent-Child Chunking**: Small chunks for precise search, large chunks for complete context
- **Deduplication**: Tracks retrieved content to avoid duplicates across queries
- **Natural Language Input**: LLM parses user intent instead of rigid commands
- **Generator-Critic Loop**: Automatic retry for low-coverage topics

---
