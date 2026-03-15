# Tasks for EnsembleAI Hackathon 2026

This repository contains example task submissions for the **EnsembleAI Hackathon 2026**.  
Each task has its own directory containing the necessary code and submission examples.

---

# Solution Overview

## Task 2 - Context Retrieval for Repository-Level Code Completion

For Task 2, I approached the problem as a **context selection and ranking task**, not as raw code generation. The main goal was to build a pipeline that can reliably assemble a compact, high-signal context for repository-level completion across different evaluator models, instead of overfitting to one specific model behavior.

The core idea was to use an **agentic local-retrieval pipeline** operating only on the unpacked repositories for a given datapoint. The agent inspects the target file, explores nearby files and modified paths, searches for relevant symbols and patterns, and then explicitly builds the final context from exact repository snippets. I treated the problem as a trade-off between:

- keeping the **target-local code early** in the final context,
- finding only the most relevant supporting implementation files,
- avoiding long noisy contexts that hurt some completion models,
- preserving **exact repository substrings** in the final output.

In practice, the solution uses:

- a custom local agent in `task2/agent`,
- repository inspection and search tools such as `inspect_target`, `search_pattern`, `search_patterns`, `read_lines`, and `add_context_snippet`,
- a managed snippet buffer with ordering, compression, cleanup, and final packing,
- context sanitization to remove licenses, README-style noise, metadata-only blocks, and weak fallback snippets,
- structural analysis scripts to compare candidate prediction files before full evaluation.

The most important design decision was to prefer a **moderate, target-first context** over a very large context. Empirically, this gave a much better balance across Qwen, Mellum, and Codestral than either:

- very large multi-file contexts,
- or aggressive global reordering that pushed the target toward the end.

Key implementation files for this solution are:

- `task2/agent/solver.py`
- `task2/agent/agent_tools.py`
- `task2/agent/documents.py`
- `task2/agent/context_manager.py`
- `task2/agent/prompts.py`
- `task2/agent/analyze_predictions.py`

Overall, my Task 2 workflow was highly empirical: make one targeted change, run small experiments, inspect context structure, and only then promote the change to a full run.

## Task 3 - Forecasting Monthly Device Load from High-Frequency Telemetry

For Task 3, I treated the problem as a **time-series forecasting task with heavy feature engineering and temporal validation**, rather than as a raw sequence model trained directly on all 5-minute readings end-to-end.

The dataset is large and highly granular, so the first step was to build efficient aggregation pipelines that transform the raw telemetry into more stable modelling tables. Instead of trying to predict from the raw stream directly, I created two complementary views of the data:

- **daily features**, capturing short-term operational patterns,
- **monthly features**, capturing seasonal behavior and device-level trends.

The feature engineering pipeline includes:

- per-device aggregation of temperature and operating variables,
- variability statistics such as means, minima, maxima, and standard deviations,
- derived thermal-difference features,
- activity ratios and usage-profile summaries,
- cyclic calendar features,
- lag and momentum features across time,
- optional geolocation enrichment from `devices.csv`,
- sequence-style summary features built from earlier periods.

On top of these features, I used tree-based models that work well on tabular time-series data:

- **CatBoost** as the main high-capacity model,
- **HistGradientBoostingRegressor** as a complementary residual-style model,
- a blending stage that searches for a robust combination of predictions using rolling validation.

The main Task 3 strategy was to combine:

- chunked feature building for memory efficiency,
- time-based cross-validation that respects the extrapolative nature of the task,
- device-level baselines,
- and a final ensemble that is more stable than any single model alone.

Important scripts for this part are:

- `task3/build_daily_features.py`
- `task3/daily_pipeline.py`
- `task3/build_monthly_features.py`
- `task3/monthly_pipeline.py`
- `task3/train_monthly_direct.py`
- `task3/train_and_submit_ensemble.py`

This approach let me keep the pipeline practical and reproducible while still capturing both short-term operating behavior and longer-term seasonal effects.

---

# How to Use

## Python Environment Setup

We strongly recommend using a **Python virtual environment** before installing dependencies.

### Create a virtual environment

```bash
python3 -m venv .venv
```

### Activate the environment

```bash
source .venv/bin/activate
```


### Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Before running the submission script, create a `.env` file in the root directory of the project.  
This file stores configuration variables required for authentication and communication with the submission server.

## Example `.env`

```
TEAM_TOKEN="mytoken"
SERVER_URL="http://149.156.182.9:6060"
```

## Variables

- **TEAM_TOKEN** – Your team authentication token provided by the hackathon organizers. It is used to authorize submissions.
- **SERVER_URL** – Base URL of the EnsembleAI submission server where results are sent.

### Security Note

Do **not** commit your `.env` file to version control.  
Keep your `TEAM_TOKEN` private.

---

# Submitting Results

To submit results for a task, run:

```bash
python3 example_submission.py
```

After submission, the server will return a **request ID**.  
This ID can be used to check the processing status of your submission.

---

# Checking Submission Status

You can check the status of a submission using the request ID:

```bash
python3 shared/get_task_status.py --request-id <id>
```

Example:

```bash
python3 shared/get_task_status.py --request-id 123456
```

This command will return the current status of the task and the score (when available).

---

# Leaderboard

The leaderboard is available at:

http://149.156.182.6/

## Public Leaderboard

- Each task has its **own leaderboard**.
- During the hackathon, the leaderboard displays the **maximum public score** achieved by each team.

## Final Score

The **final evaluation score** is determined by:

- The **private score**
- From the **last submission sent before the hackathon deadline**

This means that even if an earlier submission achieved a higher public score, the **last submitted solution** will determine your final ranking.

---

# Troubleshooting

If you notice any strange behavior or errors, please contact the **Infrastructure Team** on Discord or on-site.
