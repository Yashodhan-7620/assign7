# California Housing — Linear Regression MLOps Pipeline

Refactored from a notebook into a testable pipeline with CI/CD.

## Structure
```
.
├── .github/workflows/ci.yml   # GitHub Actions: lint -> test -> train -> upload model
├── src/train.py                # training pipeline (importable functions)
├── tests/test_train.py         # pytest suite
├── models/                     # regression.pkl + scaler.pkl land here (gitignored)
├── requirements.txt
└── pytest.ini
```

## Run locally
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.train          # trains + saves models/regression.pkl, models/scaler.pkl
pytest                       # runs the test suite with coverage
```

## Push to GitHub and trigger the pipeline

```bash
cd ml-cicd-project
git init
git add .
git commit -m "Add training pipeline, tests, and CI/CD workflow"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

As soon as this lands on `main`, GitHub Actions will:
1. **test** — install deps, lint with flake8, run pytest w/ coverage (on Python 3.10 and 3.11)
2. **train-and-package** (only after tests pass, only on `main`) — retrain the model and
   upload `regression.pkl` + `scaler.pkl` as a downloadable build artifact

You can watch it run under your repo's **Actions** tab. No extra secrets are needed for
this workflow as written — it doesn't push anywhere external. If you want real CD (e.g.
deploy the model to S3, a model registry, or an API), add an `AWS_ACCESS_KEY_ID`-style
secret under **Settings → Secrets and variables → Actions** and uncomment/extend the
`Deploy model` step at the bottom of `ci.yml`.

## Notes on what changed vs. the original notebook
- `test_model.predict()` was being called with no arguments (would've crashed) — fixed to
  predict on `X_test_scaled`.
- The fitted `StandardScaler` wasn't being saved — without it you can't correctly
  transform new data at inference time. It's now pickled alongside the model.
- Everything is wrapped in functions in `src/train.py` so it's unit-testable and
  reusable, instead of living in notebook cell state.
