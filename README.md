<div align="center">

# 🚗 AutosafeAI

*AI-powered vehicle recall risk checker*

**🔗 URL :** [autosafe-ai.streamlit.app](https://autosafe-ai.streamlit.app/)

</div>

---

Most automotive recalls are announced 2 to 5 years after the first wave of complaints. By that point, the damage is already done. AutosafeAI was built to close that gap. it continuously monitors NHTSA (National Highway Traffic Safety Administration) complaint data, runs machine learning models on complaint patterns, and surfaces early risk signals for any vehicle before a recall ever gets officially announced.

Whether you are buying a used car, monitoring one you already own, or researching automotive safety trends. AutosafeAI gives you a data-backed picture of what is actually happening with a vehicle, not just what has been officially acknowledged.

---

## What It Does

**Vehicle Risk Assessment :**
Enter any make, model, and year and get a risk score from 0 to 100. The score is backed by a trained classification model, color coded by severity, and paired with a confidence value so you know how much weight to put on it.

**Complaint Explorer :**
The database holds over 180,000 real NHTSA complaints. You can browse them by component, date range, or crash involvement flag. Complaint trends are visualized over time so you can see whether an issue is growing or stabilising.

**Recall History :**
Over 1,500 official recall records are stored and searchable. Each entry links back to the original NHTSA campaign page so everything is fully verifiable.

**AI Chatbot with CRAG Pipeline :**
The conversational interface uses a Corrective RAG (CRAG) pipeline under the hood. When you ask a question, the system retrieves the most semantically relevant complaints and recall documents from the vector store, evaluates their relevance, and if confidence is low it supplements retrieval with a live web search before passing context to the LLM. This means answers are grounded in real data and self-correcting, not just model guesswork.

You can ask things like:

> "Is the BMW coolant pump issue serious?"
> "How many complaints has the 2019 VW Tiguan received about fuel leaks?"
> "Which Ford models have the most unresolved brake complaints?"

**REST API**
A FastAPI layer exposes the system programmatically for developers and researchers.

```
GET /api/v1/predict?make=Toyota&model=Camry&year=2021
GET /api/v1/complaints?make=BMW&model=3-Series&year=2020
GET /api/v1/recalls?make=Ford&model=F-150
GET /api/v1/stats
```

Sample prediction response:

```json
{
  "risk_score": 71,
  "prediction": "high_risk",
  "confidence": 0.84,
  "top_components": ["fuel_system", "engine_cooling"]
}
```

Full interactive documentation is available at `/docs` via Swagger UI.

---

## How It Works

```
NHTSA Public API  (complaints + recalls)
         ↓
  Daily Ingestion Pipeline
  Incremental updates, retry logic, structured logging
         ↓
  SQLite Database
  180,000+ complaints  |  1,500+ recalls
         ↓
  Feature Engineering
  TF-IDF on complaint text  +  component encoding  +  temporal features
         ↓
  LightGBM Classifier
  Trained on labeled data (complaints matched to confirmed recalls)
         ↓
  Risk Score  →  Streamlit App  +  FastAPI  +  CRAG Chatbot
```

Complaint text goes through standard NLP preprocessing before TF-IDF vectorization. These text features are combined with structured fields like crash flag, component category, and model year to form the final feature matrix. The model was trained on a labeled dataset where complaints were matched against confirmed NHTSA recalls, with an 80/20 stratified train-test split.

The CRAG pipeline uses sentence-transformers to embed all complaints and recalls into a ChromaDB vector store. At query time, retrieved documents are scored for relevance. If relevance falls below threshold, the pipeline triggers a live web search to supplement context before the LLM generates a response.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web App | Streamlit |
| API | FastAPI |
| Database | SQLite |
| ML Model | LightGBM |
| NLP & Embeddings | TF-IDF, sentence-transformers |
| Vector Store | ChromaDB |
| LLM | Groq API |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud |

---

## Model Performance

| Model | Accuracy | F1 Score | Recall
|---|---|---|---|
| Logistic Regression (baseline) | 71% | 0.67 | 77.3% |
| Random Forest | 91.1% | 0.55 | 74.6% |
| LightGBM (production) | 91.6% | 0.65 | 70.2% |

Evaluated on a held-out 20% test set, stratified by manufacturer. Target thresholds were 75% accuracy and around 0.70 F1.

---

## Getting Started

```bash
git clone https://github.com/yourusername/autosafeai.git
cd autosafeai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
# Ingest data
python scripts/run_ingestion.py

# Train model
python scripts/train_model.py

# Launch app
streamlit run src/app/streamlit_app.py
```

Run with Docker:

```bash
docker-compose up
```

---

## Testing

```bash
pytest
pytest --cov=src --cov-report=term-missing
```

Test coverage sits above 70% across data ingestion, processing, and model modules.

---

## Data Sources

All data is publicly available from official government sources.

- NHTSA Complaints API: `https://www.nhtsa.gov/nhtsa-datasets-and-apis#complaints`
- NHTSA Recalls API: `https://www.nhtsa.gov/nhtsa-datasets-and-apis#recalls`

---

## Active Development

AutosafeAI is under active development. LangGraph-based AI agents are currently being integrated to autonomously monitor emerging complaint clusters, cross-reference data sources, and generate structured safety reports without manual prompting.

---

<div align="center">
<sub>Built on public NHTSA data. Not affiliated with NHTSA or any vehicle manufacturer.</sub>
</div>
