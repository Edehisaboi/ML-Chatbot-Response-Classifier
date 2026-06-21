# Japeto Prediction Studio

Japeto Prediction Studio turns the original research notebook into a reproducible
machine-learning package and a local FastAPI dashboard. Training and serving are
separate: model artifacts are prepared once, then loaded on every server start.

## Project layout

```text
data/raw/dataset.xlsx         immutable source workbook
data/processed/               normalized records and split assignments
data/embeddings/              generated 1,536-dimensional NumPy matrices
artifacts/models/             fitted model pipelines
artifacts/metrics/            evaluation JSON
app/dashboard.html            single-file dashboard
src/japeto_classifier/        application and ML package
notebooks/                    optional EDA/report notebooks
```

Generated data and model artifacts are ignored by Git. JSON manifests record their
provenance so they can be reproduced.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env` before generating embeddings or using an
OpenAI-backed model. Never place the key in source code, notebooks, or reports.

## Explicit workflow

```powershell
# 1. Normalize the workbook without modifying it
python -m japeto_classifier ingest

# 2. Create the shared train/calibration/test assignments
python -m japeto_classifier split

# 3. Generate both text-embedding-3-small matrices (resumable)
python -m japeto_classifier embed --mode all

# 4. Train all 20 compatible model configurations
python -m japeto_classifier train --features all

# 5. Inspect the registry or start the dashboard
python -m japeto_classifier evaluate
python -m japeto_classifier serve
```

For a first local run, `python -m japeto_classifier bootstrap --with-openai`
performs missing preparation stages, trains the catalogue, and starts the server.
Without `--with-openai`, bootstrap prepares and trains the twelve TF-IDF models.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) after starting the server.

## Evaluation policy

- The chatbot response is always the classification target input.
- `response_only` uses only that response; `context_enhanced` additionally uses
  the associated user message.
- All models share deterministic partitions. A complete `session_id` stays in
  one partition to prevent conversation leakage.
- TF-IDF vocabulary fitting and hyperparameter search occur only inside training
  folds. Calibration and test records are isolated.
- The locked test set is evaluated once. Integrity takes priority over matching
  the legacy 84.93% result.

The original `notebook.ipynb` is retained as legacy research evidence; the
application never imports or executes it.

