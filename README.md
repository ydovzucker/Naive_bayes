# 🧠 Naive Bayes Classifier Project

A complete, end‑to‑end **Naive Bayes** workflow written in Python.
It lets you **load tabular data**, **clean it**, **train** a multinomial/ categorical Naive Bayes model, **serve** it behind a **FastAPI** REST service, explore it through a **Streamlit** UI, or interact with it from the command line.

---

## 📑 Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Running the FastAPI Server](#running-the-fastapi-server)
5. [Using the Streamlit App](#using-the-streamlit-app)
6. [CLI Orchestrator](#cli-orchestrator)
7. [Library API](#library-api)
8. [FAQ](#faq)
9. [License](#license)

---

## ✨ Features

* **Flexible data ingestion** – CSV/Excel/SQL via `DataLoader`
* **Automatic cleaning & imputation** – `Cleaner` handles missing values (categorical ➜ mode, numeric ➜ mean)
* **Naive Bayes training** – Laplace‑smoothed priors & conditionals in `Trainer`
* **Prediction & evaluation** – `ModelChecker` (`predict`, `evaluate_accuracy`)
* **REST API** – FastAPI endpoints `/train` & `/predict`
* **Low‑code UI** – Streamlit dashboard for data preview, training & inference
* **One‑shot CLI script** – Starts the server, trains on a dataset, and queries a prediction
* **Stateless deployment** – No external DB; models kept in memory

---

## 🗂️ Project Structure

```
.
├── data/                 # Example datasets
│   └── data.csv
├── api_server.py         # FastAPI instance
├── app.py                # Streamlit front‑end
├── cli_runner.py         # End‑to‑end demo script
├── dataLoader.py
├── cleaner.py
├── trainer.py
├── modelChecker.py
├── requirements.txt
└── README.md             # (this file)
```

---

## 🚀 Quick Start

```bash
# 1 — clone & install deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2 — run the API (background tab)
uvicorn api_server:app --reload

# 3 — open the Streamlit dashboard (new tab)
streamlit run app.py
```

Now browse to [http://localhost:8501](http://localhost:8501) for the Streamlit UI, or hit the API at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger docs.

---

## 🛰️ Running the FastAPI Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Path       | Body                                                | Description            |
| ------ | ---------- | --------------------------------------------------- | ---------------------- |
| POST   | `/train`   | `{ "data": [ { ... } ], "target_column": "label" }` | Train model & persist  |
| POST   | `/predict` | `{ "featureA": "foo", "featureB": "bar", ... }`     | Return predicted class |

---

## 🎨 Using the Streamlit App

1. **Load data** – paste a CSV URL *or* upload a file.
2. **Select target column** – the label you want to predict.
3. **Train model** – click **🚀 Train Model**.
4. **Predict** – choose feature values in the sidebar and hit **🎯 Predict**.

---

## 🖥️ CLI Orchestrator

`cli_runner.py` automates the whole flow:

```bash
python cli_runner.py
```

It will:

1. Spawn the FastAPI server in the background.
2. Ask for a local CSV path & target column.
3. POST to `/train`.
4. Prompt for feature values and call `/predict`.
5. Terminate the server on exit.

---

## 🛠️ Library API (import in your own code)

```python
from dataLoader import DataLoader
from cleaner import Cleaner
from trainer import Trainer
from modelChecker import ModelChecker

loader = DataLoader()
df = loader.load_data("data/data.csv", source_type="csv")

cleaned_df = Cleaner(df).clean()
prior, cond = Trainer(cleaned_df, target_column="label").train()
checker = ModelChecker(prior, cond, "label")

print(checker.predict({"feature1": "A", "feature2": "B"}))
```

---

## 🤔 FAQ

| Question                                             | Answer                                                                                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Does this support continuous (numeric) features?** | Currently every numeric column is binned as-is and treated categorically. Add your own discretisation strategy if needed. |
| **Can I save the trained model?**                    | Persist `prior` & `conditional_prob` (e.g. pickle/JSON) and reload into `ModelChecker`.                                   |
| **GPU required?**                                    | Nope — Naive Bayes is lightweight.                                                                                        |

---

## 📜 License

MIT — see [`LICENSE`](LICENSE) for details. Feel free to fork & adapt.
