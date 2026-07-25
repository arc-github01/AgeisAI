# AEGIS

**Adaptive Behavioral Threat Detection for IT/OT Systems**

## Objective

Signature-based security controls fail against novel intrusions and against
slow, patient attackers who never trip a known rule. AEGIS takes the opposite
approach: it learns what *normal* access behaviour looks like for each
individual user, service account and edge device — their habitual hours,
locations, devices, authentication methods, resources and command sequences —
and then scores every new access event by how far it departs from that
entity's own history. Each deviation is turned into an explainable 0–100 risk
score, an attack-type hypothesis (credential misuse, brute force, impossible
travel, lateral movement, device spoofing, and more), and a plain-language list
of the evidence that triggered it, surfaced to an analyst through a SOC
dashboard. The system is designed for extreme class imbalance, legitimate
behavioural drift, and entities with little or no history.

## Architecture

> Placeholder — the detailed architecture diagram and component walkthrough are
> written in `docs/ARCHITECTURE.md` as each phase lands. Current target shape:

```
Synthetic enterprise/OT environment
        -> access & connection events
        -> event processing
        -> behavioral feature engineering
             |-> entity behavioral profiles -> statistical deviation
             |-> sequence model             -> transition anomaly
        -> IsolationForest anomaly detection
        -> risk engine
             |-> attack classifier
             |-> explainability
        -> alert store
        -> Streamlit SOC UI
```

Design decisions and their justifications are recorded in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Setup

Requires Python 3.11+ (verified on CPython 3.14.5).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
pytest -q
streamlit run app.py            # SOC console at http://localhost:8501
```

The console runs before any data exists: each page renders its final layout and
states which artifact it is waiting for, which phase produces it and the command
that creates it. Nothing is ever displayed as a placeholder number.

All parameters live in [`config/config.yaml`](config/config.yaml). Runs are
reproducible from `seed.master`; `generator.profile` switches between a small
`dev` dataset and the `full` one.

## Repository layout

```
app.py      Streamlit entry point (thin router over dashboard/)
config/     config.yaml - single source of truth for every tunable parameter
src/        library code
  schema.py     canonical event contract + threat taxonomy + leakage guard
  artifacts.py  registry of every pipeline output and the phase that makes it
  evaluation/   imbalance-aware metrics, run manifests, report figures
dashboard/  SOC console: state layer, theme, charts, five analyst pages
tests/      pytest suite for behavioural, metric and UI logic
docs/       decision record (DECISIONS.md), architecture, report
data/       generated datasets (git-ignored, reproducible from seed)
models/     persisted models and entity profiles (git-ignored)
artifacts/  alert store, evaluation figures, metric exports (git-ignored)
```

## Roadmap

| Phase | Scope | Status |
|-------|-------------------------------------------------|--------|
| 1 | Architecture, repository, environment, config, determinism | Done |
| 1.5 | SOC console shell + evaluation/report infrastructure | Done |
| 2 | Synthetic entity population + normal behaviour generator | Pending |
| 3 | Injection of all 7 attack / edge-case behaviours | Pending |
| 4 | Behavioral feature engineering (temporal, geo, device, resource, sequence) | Pending |
| 5 | Entity profiling + cohort-based cold start | Pending |
| 6 | IsolationForest anomaly detection + alert-budget thresholds | Pending |
| 7 | Supervised attack-type classifier | Pending |
| 8 | Risk engine + explainability layer | Pending |
| 9 | Concept-drift adaptive profiles | Pending |
| 10 | Streamlit SOC dashboard | Pending |
| 11 | Live attack simulator through the real pipeline | Pending |
| 12 | Evaluation: PR-AUC, per-attack metrics, alert-budget recall | Pending |
| 13 | Report, architecture documentation, presentation | Pending |

## Evaluation stance

Accuracy is **not** a headline metric: at ~1% attack prevalence a model that
predicts "normal" for everything scores ~99%. AEGIS is measured on PR-AUC,
per-attack precision/recall/F1, false-positive rate, recall achieved within a
realistic top-1% analyst alert budget, and campaign-level time-to-detection.

These metrics were implemented and tested in Phase 1.5, before any model
existed, so the reported quantities are dictated by the problem rather than
chosen after seeing which look flattering. Every metrics file is written with a
manifest recording the seed, config snapshot, git commit and package versions of
the run that produced it.
