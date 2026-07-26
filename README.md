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

## Analyst SOC dashboard (Phase 10)

Launch the console from the repository root after installing dependencies:

```bash
streamlit run app.py
```

Open http://localhost:8501. With `dashboard.data_source: auto` (default), the
console prefers real pipeline artifacts when `events` + `alerts` exist;
otherwise it falls back to a clearly labelled development fixture.

| Page | Purpose |
|------|---------|
| SOC Overview | Risk summary, severity/attack mix, top entities, recent high-priority alerts |
| Alert Queue | Ranked triage list with filters + explainable alert detail |
| Entity Investigation | Baseline profile, risk evolution, alert reasons, event history |
| Streaming Replay | Reliable batch demo of Phase 9 `process_event()` (optional) |
| Attack Simulator | Inject a real generator campaign through `process_event`; alerts join the SOC session overlay |
| Model Performance | **Evaluation / debug** metrics from Phase 12 artifacts only |

Verify the live demonstration end-to-end before presenting (runs all seven
scenarios through the real detection pipeline and prints risk, severity and
latency per scenario):

```bash
python scripts/verify_live_demo.py
```

Recommended demo path once the offline pipeline has been run:

1. Overview → confirm alert counts and high-risk entities match artifacts
2. Alert Queue → open a CRITICAL alert and inspect contribution breakdown
3. Entity Investigation → follow the same entity's risk timeline + history
4. Streaming Replay → click **Replay next batch** (models must be on disk)
5. Attack Simulator → inject a scenario, then return to Overview / Alert Queue
   to find the freshly scored alert (session overlay; **Clear live overlays**
   restores the persisted-artifact view)

Do not treat the Model Performance page as operational SOC output; it exposes
labelled evaluation metrics when present and is marked as evaluation/debug.

## Generating the dataset

```bash
python -m src.generator                 # dev profile, attacks injected
python -m src.generator --profile full
python -m src.generator --benign-only   # clean reference stream, no campaigns
```

This writes `data/generated/entities.json` (the ground-truth behavioural
definition of every entity) and `data/generated/events.parquet` (the labelled
access-event dataset). The printed summary reports the *achieved* attack
prevalence and per-type campaign counts rather than the configured targets, since
campaign sizes are clamped to shapes each attack plausibly has.

## Offline detection pipeline

```bash
python -m src.features                    # causal features + frozen profiles
python -m src.models.anomaly_detector     # IsolationForest + Phase 5 metrics
python -m src.models.attack_classifier    # attack-type classifier + metrics
python -m src.risk                        # hybrid risk scores, alerts, metrics
python -m src.drift                       # risk-gated adaptive profiles + drift eval
python -m src.detection                   # streaming replay + latency metrics
```

For a long-lived event stream, either call the stateful module interface or
own the engine explicitly:

```python
from src.detection import StreamingEngine, process_event

result = process_event(event)  # process-local engine is reused across calls

# Explicit lifecycle control for services/tests:
engine = StreamingEngine.load(apply_drift_updates=True)
result = process_event(event, engine=engine)
```

`result` contains anomaly/risk scores, severity, classifier hypothesis,
explanations, alert/cooldown outcome, profile source, and adaptive-update
decision. Inference strips ground-truth metadata before feature computation.

## Repository layout

```
app.py      Streamlit entry point (thin router over dashboard/)
config/     config.yaml - single source of truth for every tunable parameter
src/        library code
  schema.py     canonical event contract + threat taxonomy + leakage guard
  artifacts.py  registry of every pipeline output and the phase that makes it
  evaluation/   imbalance-aware metrics, run manifests, report figures
dashboard/  SOC console: state layer, theme, charts, analyst pages
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
| 2 | Synthetic entity population + normal behaviour generator | Done |
| 3 | Injection of all 7 attack / edge-case behaviours | Done |
| 4 | Behavioral feature engineering (temporal, geo, device, resource, sequence) | Done |
| 5 | IsolationForest anomaly detection + calibrated operating points + evaluation | Done |
| 6 | Hybrid behavioral risk engine + explanations + alerts | Done |
| 7 | Supervised attack-type classifier (RandomForest) | Done |
| 8 | Concept-drift adaptive profiles (risk-gated EWMA) | Done |
| 9 | Near-real-time streaming `process_event` + replay | Done |
| 10 | Streamlit SOC dashboard polish against live artifacts | Done |
| 11 | Live attack simulator through the real pipeline | Done |
| 12 | Final evaluation report + presentation | Pending |

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








CONTEXT FOR CURSOR
