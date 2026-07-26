# AEGIS - System Architecture

> Sections marked **[PENDING PHASE n]** are written when that phase lands. No
> component is described here before it exists.

## 1. Design goal

Detect intrusions and compromised-credential activity by learning what normal
access behaviour looks like *for each individual entity*, then scoring
deviations from that entity's own history — with an explanation an analyst can
act on, under a realistic triage budget.

## 2. Pipeline

```
                       Synthetic enterprise / OT environment      [PHASE 2-3]
                                     |
                            access & connection events
                                     |
                              event processing                     [PHASE 4]
                                     |
                      behavioral feature engineering               [PHASE 4]
                                     |
                +--------------------+--------------------+
                |                                         |
     entity behavioral profiles                    sequence model
     (personal + cohort blend)     [PHASE 5]       (Markov transitions) [PHASE 4]
                |                                         |
      statistical deviation score               transition anomaly score
                |                                         |
                +--------------------+--------------------+
                                     |
                        IsolationForest anomaly score      [PHASE 5 - Done]
                                     |
                                risk engine                [PHASE 6 - Done]
                                     |
                +--------------------+--------------------+
                |                                         |
        attack classifier                          explainability
        (RandomForest)        [PHASE 7]            (deterministic)  [PHASE 6]
                |                                         |
                +--------------------+--------------------+
                                     |
                      adaptive profiles (risk-gated EWMA)  [PHASE 8 - Done]
                                     |
                    process_event() streaming path         [PHASE 9 - Done]
                                     |
                               alert store
                                     |
                          Streamlit SOC console         [PHASE 1.5 shell; polish pending]
```

## 3. Module map

| Module | Responsibility | Status |
|---|---|---|
| `src/config.py` | YAML single source of truth, dotted access, immutable | Done |
| `src/paths.py` | Project-root-relative locations | Done |
| `src/schema.py` | Canonical event contract, threat taxonomy, leakage guard | Done |
| `src/artifacts.py` | Registry of every pipeline output (path, phase, producer) | Done |
| `src/utils/` | Deterministic seeding, great-circle geography, logging | Done |
| `src/evaluation/` | Imbalance-aware metrics, manifests, report figures | Done |
| `dashboard/` | SOC console: state layer, theme, charts, five pages | Shell done |
| `src/generator/` | Entities, normal behaviour, attack injection | Done |
| `src/features/` | Temporal, geographic, device, resource, sequence features | Done |
| `src/profiling/` | Personal + cohort baselines, cold start | Done |
| `src/models/` | IsolationForest + RandomForest attack classifier + evaluation | Done |
| `src/risk/` | Hybrid risk engine, explanations, alerts, evaluation | Done |
| `src/drift/` | Risk-gated EWMA adaptive profiles + drift evaluation | Done |
| `src/detection/` | Streaming `process_event`, replay, live injection wiring | Done |
| `src/generator/live_injection.py` | Phase 11 live campaign synthesis for the SOC simulator | Done |

## 4. Contracts that hold the system together

**Event contract** (`src/schema.py`). One definition of an access event, used by
the generator, the feature layer, the offline evaluation and the live
simulator. Geography is structured (`country`, `city`, `latitude`,
`longitude`); device fingerprints are decomposed (`device_id`, `device_os`,
`device_firmware`, `device_protocol`, `device_mac`). Labels (`label`,
`is_attack`, `campaign_id`) are a separate column group that
`assert_no_label_leakage()` keeps out of the feature space.

**Artifact registry** (`src/artifacts.py`). Every file the pipeline writes is
declared once, with the phase that produces it and the command that creates it.
Producers and consumers cannot drift apart, and the dashboard derives its
readiness display from the same registry.

**Dashboard state contract** (`dashboard/state.py`). Pages are pure functions of
a `DashboardContext` and must tolerate missing data. This is why the console runs
end-to-end before any model exists, and why the live simulator can later hand it
freshly scored events without a page rewrite.

**Metrics contract** (`src/evaluation/`). Defined before the models, so the
reported quantities are dictated by the problem (rare-event detection under a
fixed analyst budget) rather than chosen to flatter a result.

**Streaming state contract** (`src/detection/`). `process_event()` validates and
strips an incoming observation, computes features from prior same-entity
history, then commits history only after scoring. `StreamingEngine` owns
per-entity rolling history, last fingerprint, hybrid evidence, alert cooldown,
and adaptive profile state. The process-local convenience API reuses one engine;
services can inject an explicitly managed engine.

## 5. Evaluation methodology

- **Temporal split only.** Train on an earlier window, evaluate on a strictly
  later one. Profiles used at evaluation time are built from the training window
  and updated causally.
- **Alert budget.** The operating point is the top `alerting.budget_fraction`
  (default 1%) of events by risk, selected as an exact top-k with deterministic
  tie-breaking, so a tied score cannot silently multiply the analyst's workload.
- **Reported metrics.** PR-AUC, ROC-AUC, precision/recall/F1 at the budget,
  false-positive rate, per-attack precision/recall/F1, confusion matrix,
  campaign-level detection coverage and time-to-first-alert. Accuracy is
  deliberately not a headline metric.
- **Provenance.** Every metrics document is written with a manifest recording
  the master seed, full config snapshot, git commit and package versions.

## 6. Production mapping

The prototype is a single Python process, but the module boundaries are drawn so
that each maps onto a deployable service:

| Prototype | Production |
|---|---|
| Generator / event replay | Kafka (or Kinesis) ingestion from SIEM, IdP, VPN, OT historian |
| `src/features` batch pass | Stateless stream feature processor |
| Entity profiles in `joblib` | Redis / feature store keyed by entity |
| In-process scoring | Model inference service behind gRPC/REST |
| `alerts.parquet` | Alert database + queue |
| Streamlit console | SOC / SIEM integration (Splunk, Sentinel, Chronicle) |

Event-by-event inference is a first-class path in the prototype, not an
afterthought: the live simulator exercises it.

## 7. Known limitations

[PENDING PHASE 13 — populated from real observed behaviour, not speculation.]
