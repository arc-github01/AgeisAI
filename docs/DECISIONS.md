# AEGIS - Architecture Decision Record

This file is the project's memory. Any future change (human or AI-assisted) must
stay consistent with these decisions, or explicitly supersede one by adding a
new entry that says so.

Format: **ADR-n — Title** · Status · Context · Decision · Consequences.

---

## ADR-1 — Hybrid detection instead of a single deep model
**Status:** Accepted (Phase 1)

**Context.** The problem is sequence-aware behavioral anomaly detection under
~1% class prevalence, with a hard 24-hour delivery window and explainability as
a graded criterion. A single end-to-end deep sequence model is attractive on
paper but is slow to tune, opaque to a SOC analyst, and fragile under extreme
imbalance.

**Decision.** Build a hybrid: per-entity statistical baselines + a Markov-style
resource-transition model + IsolationForest as the unsupervised anomaly
detector + a supervised RandomForest for attack-type classification + a
weighted risk engine on top.

**Consequences.** Every stage produces an interpretable intermediate signal
that the explainability layer can quote directly. Deep sequence models
(LSTM/GRU/Transformer) are explicitly out of scope unless the full baseline
system is complete, time remains, and an experiment shows measurable gain.

---

## ADR-2 — Personalised baselines, not global thresholds
**Status:** Accepted (Phase 1)

**Context.** "Is a 02:00 login suspicious?" is unanswerable globally: it is
normal for a night-shift operator and alarming for a day-shift engineer.

**Decision.** Detection asks *"is this unusual for THIS entity?"*. All rarity,
novelty and deviation features are computed against an entity-specific profile,
falling back to a cohort profile when personal history is thin.

**Consequences.** Feature engineering requires an ordered pass over each
entity's history; profiles become a first-class persisted artefact alongside
models.

---

## ADR-3 — One canonical event contract in `src/schema.py`
**Status:** Accepted (Phase 1)

**Context.** The generator, the feature layer, the offline evaluation and the
live attack simulator must all speak about the same object. If the simulator
constructs events differently from the generator, the demo is a lie.

**Decision.** `src/schema.py` defines the column contract, the attack taxonomy
and `validate_events()`. Geography is stored structured (`country`, `city`,
`latitude`, `longitude`) and device fingerprints are decomposed
(`device_id`, `device_os`, `device_firmware`, `device_protocol`, `device_mac`)
rather than as opaque strings; a flattened export can be derived later.

**Consequences.** The simulator injects events through the same validation and
feature path as the batch pipeline. Schema changes are a single-file edit with
an immediate test signal.

---

## ADR-4 — Label leakage is a runtime error, not a convention
**Status:** Accepted (Phase 1)

**Context.** The most common way an anomaly-detection prototype produces
implausible metrics is by letting ground truth reach the feature space.

**Decision.** `LABEL_COLUMNS` (`label`, `is_attack`, `campaign_id`) are the only
label-bearing fields. `assert_no_label_leakage()` raises `LabelLeakageError`
and is invoked by the feature-building layer and asserted in tests.

**Consequences.** Any attempt to shortcut detection with the label fails loudly.

---

## ADR-5 — Temporal splits only
**Status:** Accepted (Phase 1)

**Context.** Behaviour is sequential and per-entity. A random split lets an
entity's future behaviour define its own training baseline.

**Decision.** Train on an earlier period, evaluate on a strictly later, unseen
period (`evaluation.train_fraction`). Profiles used at evaluation time are
built only from the training window and then updated causally.

**Consequences.** Cold-start and drift behaviour become directly observable in
the evaluation rather than being smoothed away.

---

## ADR-6 — Deterministic seeding via derived substreams
**Status:** Accepted (Phase 1)

**Context.** Reviewers must be able to reproduce the dataset and the reported
metrics exactly. A single shared global seed means adding any new random
consumer silently changes every previously generated result.

**Decision.** One `seed.master` in config. Each component derives an
independent seed with `blake2b(master:component)` (`src/utils/seeding.py`), so
streams are stable, independent and insertion-order-proof. Python's builtin
`hash()` is not used because it is salted per process.

**Consequences.** `get_rng("attacks")` always yields the same stream regardless
of what else runs first.

---

## ADR-7 — Config-as-single-source-of-truth
**Status:** Accepted (Phase 1)

**Context.** Judges will ask which numbers are principled and which are chosen.

**Decision.** All behavioural constants live in `config/config.yaml`, loaded
through an immutable `Config` object with dotted access. Risk-engine weights are
labelled in the file itself as *prototype calibration parameters*, not learned
optima.

**Consequences.** No hidden magic numbers; honest provenance for every constant.

---

## ADR-8 — Report metrics under an analyst alert budget
**Status:** Accepted (Phase 1)

**Context.** With ~1% prevalence, accuracy is meaningless and an unbounded
threshold produces an untriageable alert queue.

**Decision.** The headline metrics are PR-AUC, per-attack precision/recall/F1,
false-positive rate, and **recall within the top `alerting.budget_fraction`
(1%) of events by risk**. Accuracy is not reported as a primary metric.

**Consequences.** The threshold strategy is budget-driven rather than an
arbitrary cut-off.

---

## ADR-9 — `INSIDER_DRIFT` is an edge case, scored separately
**Status:** Accepted (Phase 1)

**Context.** Gradual, legitimate-looking privilege expansion is genuinely
ambiguous; forcing it into the malicious class would corrupt precision claims.

**Decision.** `MALICIOUS_CLASSES` excludes `INSIDER_DRIFT`. It is retained in
the classifier's label space and reported separately as a false-positive-tuning
and concept-drift case study.

**Consequences.** Detection metrics stay honest; the drift narrative gets a
concrete demonstration subject.

---

## ADR-10 — Baseline updates are poisoning-resistant
**Status:** Accepted (Phase 1)

**Context.** An adaptive baseline that absorbs everything can be trained by an
attacker to accept the attack.

**Decision.** Profiles update with exponentially decayed statistics
(`profiling.ewma_halflife_days`) but only from events scoring below
`drift.baseline_update_max_risk`.

**Consequences.** Legitimate drift is absorbed within days; high-risk activity
never becomes "normal".

---

## ADR-11 — Run-in-place layout, `src/` package, root `conftest.py`
**Status:** Accepted (Phase 1)

**Context.** A 24-hour prototype does not benefit from packaging ceremony, but
imports must work identically under pytest and `streamlit run`.

**Decision.** Keep the `src/` layout from the project brief, add a root
`conftest.py` that puts the repo root on `sys.path`, and run every entry point
from the repository root. Sub-packages are created by the phase that first
needs them — no empty placeholder modules.

**Consequences.** No install step; `pip install -r requirements.txt` then run.

---

## ADR-12 — Python 3.14 with an unpinned-but-locked dependency set
**Status:** Accepted (Phase 1)

**Context.** The only interpreter available on the build machine is CPython
3.14.5. Installed resolution: numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0,
streamlit 1.60.0.

**Decision.** `requirements.txt` states minimum versions with a documented
reason per dependency; `requirements.lock.txt` records the exact verified
resolution for reproducibility.

**Consequences.** pandas 3.x semantics (copy-on-write by default, string dtype)
apply — feature code must avoid chained assignment. Optional SHAP is deferred:
deterministic feature attribution is the primary explainability mechanism, and
SHAP will only be added if it is stable on this interpreter and adds value.

---

## ADR-13 — Build the console shell and the metrics contract before the models
**Status:** Accepted (Phase 1.5)

**Context.** The two things most likely to be rushed at hour 21 are the analyst
UI and the evaluation. Both are graded criteria, and both constrain the design
of everything upstream of them.

**Decision.** Build a fully navigable Streamlit console with all five pages and
a complete, unit-tested metrics library *before* the generator exists. Pages
render honest "awaiting phase n" states rather than placeholder data.

**Consequences.** Later phases plug into a fixed surface instead of provoking a
UI rewrite. Choosing metrics before seeing results removes the temptation to
report whichever ones look best. The console is also a live build-status board.

---

## ADR-14 — Single artifact registry
**Status:** Accepted (Phase 1.5)

**Context.** Producers and consumers of intermediate files drift apart when both
hardcode paths.

**Decision.** `src/artifacts.py` declares every pipeline output once — path,
producing phase, and the command that creates it. The dashboard's readiness
panel and empty states are generated from the same registry.

**Consequences.** Adding an output is a one-line registry change; a path can
never disagree between the writer and the reader.

---

## ADR-15 — Alert budget is exact top-k, not a score threshold
**Status:** Accepted (Phase 1.5)

**Context.** With a `score >= threshold` rule, a cluster of tied scores can
admit far more alerts than the analyst team can process. Observed in test: 100
tied events at a 10% budget produce 100 alerts instead of 10.

**Decision.** `budget_alert_mask()` selects exactly `ceil(n * fraction)` events
by stable descending sort, breaking ties by event order. A SOC has fixed
capacity, not a fixed score cut-off.

**Consequences.** Reported precision/recall at budget are honest about real
triage capacity. The tie-breaking rule is deterministic and documented.

---

## ADR-16 — `INSIDER_DRIFT` scored separately, campaign-level metrics reported
**Status:** Accepted (Phase 1.5, extends ADR-9)

**Context.** Event-level recall understates a detector that catches 1 of the 50
events in a low-and-slow campaign — operationally, that intrusion was caught.

**Decision.** Alongside event-level metrics, report campaign detection coverage
and time-to-first-alert (`campaign_detection()`).

**Consequences.** Low-and-slow performance becomes measurable rather than
rhetorical.

---

## ADR-17 — Metrics are only valid with a provenance manifest
**Status:** Accepted (Phase 1.5)

**Context.** "Never fabricate evaluation results" needs a mechanism, not a
promise.

**Decision.** `save_metrics()` is the only sanctioned writer, and it stamps
every document with seed, full config snapshot, git commit, Python and package
versions. The performance page reads only from that file and displays the
provenance. Non-finite values are sanitised to `null` before writing with
`allow_nan=False`, so the artifacts are strictly valid JSON.

**Consequences.** Any reported number is traceable to a reproducible run.

---

## ADR-18 — The simulator raises rather than renders a placeholder
**Status:** Accepted (Phase 1.5)

**Context.** A demo feature that fakes its output would invalidate the whole
demonstration, and placeholder code has a habit of surviving to the deadline.

**Decision.** `process_injection` raises `DetectionPipelineNotReadyError` until
prerequisite artifacts exist and the streaming engine can load; the INJECT
control stays disabled until those artifacts are present, and tests assert both.
Phase 9 wires injection to the real `process_event` path — still never a
fabricated alert.

**Consequences.** It is impossible to ship a fake simulator by accident.

---

## ADR-19 — Tests run in an isolated workspace
**Status:** Accepted (Phase 1.5)

**Context.** A metrics round-trip test wrote `artifacts/metrics/latest.json`,
and the dashboard immediately displayed unit-test numbers as a real run.

**Decision.** A session-scoped autouse fixture (`tests/conftest.py`) redirects
every output directory to a temporary root.

**Consequences.** Tests cannot contaminate the deliverable, and the console's
"no data" state stays truthful.

---

## ADR-20 — Console verified headlessly with `AppTest`
**Status:** Accepted (Phase 1.5)

**Context.** A dashboard that is only ever checked by eye breaks silently when a
data contract changes.

**Decision.** `tests/test_dashboard_shell.py` executes the real `app.py` through
`streamlit.testing.v1.AppTest` and renders every page. Navigation therefore uses
a sidebar radio rather than `st.navigation`, which `AppTest` drives reliably.

**Consequences.** Every future phase gets an automatic regression check on the
UI, including that pages survive the hardest case — no data at all.

---

## ADR-21 — `generator.days` is a hard horizon; the budget bends, not the calendar
**Status:** Accepted (Phase 2)

**Context.** The first engine met `target_events` by cycling past the end of the
configured window, so a 21-day `dev` profile produced 186 days of data. Worse,
it took the weekday from `day_offset % days` while stamping the event with the
raw `day_offset`. With `days: 21` (exactly three weeks) that happened to align;
with the `full` profile's 90 days it would have decoupled weekday behaviour from
the actual date, quietly destroying the weekend signal.

**Decision.** Simulate exactly `days` calendar days. Each entity's share of the
event budget is spread across the days it is actually active — decided up front
by `active_days()` — by raising its session rate, capped at `_DAILY_BURST_CAP`
times its mean daily volume so a backlog cannot become an implausible burst.

**Consequences.** Weekday structure stays aligned with real dates over any
horizon, and the temporal train/test split covers the window the config asked
for. Realised volume lands within ~1% of `target_events` rather than exactly on
it; the achieved count is reported in the run summary rather than forced.

---

## ADR-22 — One entity, one event per second
**Status:** Accepted (Phase 2)

**Context.** The original scheduler computed a session step's minute as
`base_minute + offset` and clamped it to 59 instead of carrying into the hour.
Sessions could not cross an hour boundary and their later steps piled onto a
single timestamp: 9% of rows shared an `(entity_id, timestamp)` key. Every
planned feature that divides by the gap to an entity's previous event —
geo-velocity, inter-event frequency, failed-attempts-per-window — would have
divided by zero on those rows.

**Decision.** Timestamps are real datetime arithmetic at second resolution.
Sessions are laid out in non-overlapping slots across the entity's active
window, and the window may run past midnight so a night shift is contiguous
rather than truncated. A final `_deconflict_timestamps()` pass nudges any
residual collision forward a second at a time, and a test asserts the invariant
holds on the generated dataset.

**Consequences.** "The previous event for this entity" is always well defined.
Sequence and velocity features can be written without defensive special-casing.

---

## ADR-23 — The generator writes only to registered artifact paths
**Status:** Accepted (Phase 2)

**Context.** `src/artifacts.py` exists so a producing phase and a consuming
phase cannot drift apart, but the generator ignored it and wrote
`data/generated/normal_events.csv` while the registry — and the dashboard state
layer watching it — expected `entities.json` and `events.parquet`. The entity
population, the most reusable thing the generator builds, was never persisted at
all.

**Decision.** `save()` writes exclusively through `artifact_path()`. The entity
roster is serialised to `entities.json` with its cohort resources, transition
graph, devices and geography; events go to `events.parquet` in canonical
`EVENT_COLUMNS` order. `event_id` is assigned after the global sort so it is
chronological and reproducible — the previous `uuid4()` ids changed on every run
despite the seed, which silently broke the reproducibility claim.

**Consequences.** Phase 4 onward can load a fixed contract, and the console's
build-status board reflects reality. Persisting the generator's ground-truth
entity definition also lets evaluation ask how closely a *learned* profile
recovers it.

---

## ADR-24 — Attacks are injected at mixed difficulty
**Status:** Accepted (Phase 3)

**Context.** The obvious way to build the seven injectors is to make each one
unmistakable: brute force from Moscow at 3am, impossible travel across a
continent, lateral movement straight onto a critical asset. That dataset is
worthless for evaluation. Every attack would sit far outside its entity's
baseline on several axes at once, an IsolationForest would separate it almost
perfectly, and the PR-AUC we report would measure the generator's bluntness
rather than the detector's skill. It would also teach the model nothing about the
cases a real SOC actually misses.

**Decision.** `generator.attacks.stealth_fraction` (0.35) of every type's
campaigns is generated as a subtle variant. The defining signal is always
preserved — a stealth impossible travel still exceeds
`features.max_plausible_kmh`, a stealth spoof still presents an unregistered MAC
— but the supporting signals that would make detection trivial are removed:
stealth campaigns run during working hours, originate from egress points the
population legitimately uses, keep known devices wherever the attack does not
require otherwise, and trade volume for patience. Attack rows also reuse the
benign vocabulary for `auth_method` and `action`, so no model can win by
memorising an attacker-only category. `tests/test_attacks.py` asserts per type
that the stealth variant is measurably less deviant than the obvious one.

**Consequences.** Reported PR-AUC and recall-at-budget become meaningful, and the
per-attack breakdown in `src/evaluation/` can show which behaviours are genuinely
hard. The cost is that a headline metric will look worse than a uniformly blatant
dataset would have produced, which is accepted deliberately.

A second consequence is that the configured `campaigns` count per type became a
preference rather than a guarantee. Prevalence and plausible campaign shape are
the constraints that must hold, and they conflict at small dataset sizes: an
obvious brute force cannot be shorter than its configured minimum attempt count,
so 52 campaigns at 1.5% prevalence is arithmetically impossible in a 20k-event
dataset. `AttackOrchestrator` therefore solves for the campaign count whose
expected event total lands within 10% of the type's budget and is closest to the
configured preference, and the run summary reports what was achieved.

---

## ADR-25 — Insider drift is labelled but is not an attack
**Status:** Accepted (Phase 3)

**Context.** Gradual expansion of an employee's resource footprint is
simultaneously the behaviour most likely to trip a behavioural detector and the
one most likely to be legitimate: a role change, a new project, a team
reorganisation. `src/schema.py` already excludes `INSIDER_DRIFT` from
`MALICIOUS_CLASSES` while keeping it in `ATTACK_CLASSES`. Phase 3 had to decide
what the injector actually emits.

**Decision.** Insider drift campaigns carry `label = INSIDER_DRIFT` and
`is_attack = False`, and are generated as entirely plausible activity: normal
working hours, the entity's own devices and network, successful authentication,
lower-sensitivity resources from a neighbouring cohort, and a footprint that
widens on a linear ramp rather than appearing all at once. They still receive a
`campaign_id`.

**Consequences.** `is_attack` stays exactly `label in MALICIOUS_CLASSES` for
every row, so the binary detector is trained and scored against intrusions only.
A detector that flags drift is counted as a false positive, which is the correct
accounting and gives alert-budget tuning something real to trade against. Because
the campaigns are still identified, evaluation can report drift recall separately
as a sensitivity measure, and Phase 9 inherits a labelled concept-drift exhibit
to demonstrate baseline adaptation against.

---

## ADR-26 — Frozen chronological BENIGN profiles for feature generation
**Status:** Accepted (Phase 4)

**Decision.** Phase 4 fits entity, cohort and transition profiles only from
BENIGN events before the configured chronological training cutoff. All later
events are scored against those frozen statistics; labels, campaign identifiers
and stealth metadata remain evaluation-only metadata.

**Consequences.** Feature values cannot absorb future behaviour or attack
campaigns. Entities without enough pre-cutoff history fall back to their
`(entity_type, role)` cohort profile, making cold-start behaviour explicit rather
than silently using generator ground truth.

---

## ADR-27 — Unsupervised IsolationForest trained on benign-only history
**Status:** Accepted (Phase 5)

**Decision.** The anomaly detector (`src/models/`) fits a `StandardScaler` +
`IsolationForest` on `MODEL_FEATURE_COLUMNS` for rows that are both benign and in
the chronological training split. Labels are used only to *select* those rows and
never enter the feature matrix. `random_state` is derived from `seed.master`, so
the single master seed remains the only reproducibility knob.

Scoring negates `score_samples` so higher always means more anomalous. Ranking
metrics (PR-AUC, ROC-AUC) run on the raw, strictly monotonic score to avoid
saturation; a separate min-max `anomaly_score` in [0, 1] exists only for display.
Operating points (`strict`/`balanced`/`sensitive`) are quantiles of benign
*training* scores, so no evaluation label ever influences a shipped threshold.

**Consequences.** The detector is honestly unsupervised and leakage-free, at the
cost of a rule baseline occasionally matching or beating it on obvious attacks —
which is reported rather than hidden. Two reference detectors (seeded random and a
benign-standardised rule sum) bound the result from below. Evaluation joins
`campaigns.json` only at report time to produce campaign detection rate,
time-to-detection, events-before-detection, and breakdowns by attack type,
obvious-vs-stealth, and entity type.

---

## ADR-28 — Hybrid risk engine over a second supervised model
**Status:** Accepted (Phase 6)

**Decision.** Phase 6 does not add XGBoost, RandomForest, or a neural detector.
It combines the validated IsolationForest score, the honest rule baseline,
causal Phase 4 behavioural features, and entity-scoped exponential time decay
into a saturating evidence score:

```
decayed   = S_prev * 0.5 ** (dt / halflife_seconds)
E         = Σ weight_i * activation_i(event)          # instantaneous
P         = persistence_weight * decayed              # history
risk      = 100 * (1 - exp(-(E + P) / evidence_scale))
S_new     = min(decayed + E, state_cap)
```

Activations are anchored on benign training-period quantiles and Phase 5
operating points. Weights and `evidence_scale` encode a corroboration policy
(one strong detector alone → MEDIUM; two agreeing → HIGH; sustained/multi-signal
→ CRITICAL), not a fit against evaluation labels or known missed campaigns.
Explanations are proportional shares of the evidence sum — they reconcile with
the score by construction. Alerts page at CRITICAL by default (HIGH is a watch
band); cooldown suppresses burst duplicates; severity escalation bypasses
cooldown.

**Why hybrid, and why IF still matters.** On the current synthetic set the rule
baseline can beat IsolationForest on PR-AUC for obvious attacks. That is an
acceptable and reported outcome: IF remains the novelty detector for behaviours
rules were not hand-written to catch (notably lateral-movement style resource
traversal). The hybrid's job is corroboration and temporal accumulation, not to
paper over Phase 5.

**Consequences.** Risk inputs are whitelist-guarded (`RISK_INPUT_COLUMNS`);
labels/campaign metadata join only after scoring for evaluation. Streaming
inference can reproduce the same state trajectory because decay is wall-clock
and profiles/features remain causal. Alert burden and campaign detection are
reported honestly against IF-alone and rule-alone baselines.

---

## ADR-29 — Supervised attack-type classifier is naming, not detection
**Status:** Accepted (Phase 7)

**Decision.** Phase 7 adds a RandomForest multi-class classifier over
`BENIGN` + `ATTACK_CLASSES` (including `INSIDER_DRIFT` as an edge-case class),
trained on chronological training-split rows using `MODEL_FEATURE_COLUMNS` only.
`class_weight=balanced_subsample` addresses rarity. `random_state` is derived
from `seed.master`. The classifier does **not** replace IsolationForest or the
hybrid risk engine: those decide *whether* something is suspicious; this model
proposes *what kind* of behaviour it looks like, with a confidence score.

Predictions enrich alerts (`attack_type`, `attack_confidence`) but never enter
`RISK_INPUT_COLUMNS` or IsolationForest features. Evaluation reports per-class
precision/recall/F1, confusion matrix, malicious top-1 accuracy, and a binary
attack-vs-benign view of the multi-class output.

**Consequences.** Analysts get a named hypothesis on CRITICAL alerts. Mis-naming
an attack type does not by itself open or suppress an alert. Class imbalance and
rare types (especially on the `dev` profile) can yield uneven per-class F1; that
is reported rather than hidden.

---

## ADR-30 — Adaptive profiles are separate from frozen Phase-4 baselines
**Status:** Accepted (Phase 8)

**Context.** Legitimate behaviour drifts (new devices, relocated users,
`INSIDER_DRIFT`-style privilege expansion). An adaptive baseline that absorbs
every event can be poisoned by an attacker who repeats the attack until it looks
"normal". Offline `features.parquet` must remain leakage-safe: profiles used
there were frozen from BENIGN history at or before the train/eval cutoff.

**Decision.** Keep Phase-4 `ProfileBundle` frozen for offline features. Add a
separate `AdaptiveProfileStore` (`src/drift/`) seeded from that bundle. Replay
(or later stream) post-cutoff events in time order and update entity statistics
with EWMA / decayed counts using `profiling.ewma_halflife_days`, but **only when
`risk_score < drift.baseline_update_max_risk`**. The gate uses hybrid risk only —
never attack labels — so evaluation metadata cannot steer adaptation. High-risk
events leave the adaptive profile unchanged (poisoning resistance). Artifacts:
`adaptive_profiles.joblib`, `drift_evaluation.json`. Offline features are not
rewritten from adaptive state.

**Consequences.** Concept drift is demonstrable (low-risk benign and many
`INSIDER_DRIFT` events are absorbed; high-risk / most malicious campaigns are
blocked). Stealthy attacks that score below the risk gate can still update —
reported honestly as a residual gap rather than papered over with label-aware
gating. Streaming `process_event` can later call the same update path without
changing the offline leakage contract.

---

## ADR-31 — Streaming reuses offline models; state lives in the engine
**Status:** Accepted (Phase 9)

**Context.** The brief requires near-real-time feasibility: a single access event
must flow through features → detection → risk → classification → explanation →
alert → safe profile update without a batch job. Duplicating the offline
pipeline would drift; fabricating alerts would invalidate the demo.

**Decision.** `StreamingEngine.process_event` (and module-level `process_event`)
is the sole streaming entry point. The module-level function reuses one
process-local engine so repeated calls preserve state; services may own an
explicit engine when they need controlled lifecycle or durable state. It:

1. Strips labels / evaluation metadata and validates observation columns.
2. Computes causal features via shared `compute_event_features` using only
   prior per-entity history and the current profile view (frozen bundle, or
   adaptive snapshot when drift updates are enabled).
3. Scores with the persisted IsolationForest and the same rule baseline fit
   recipe as Phase 5.
4. Classifies with the persisted RandomForest (names only — does not alter risk).
5. Updates hybrid risk / alert state through `RiskEngine.score_event` (same
   formula as batch `score_frame`).
6. Optionally applies Phase-8 risk-gated adaptive updates.

`python -m src.detection` replays `events.parquet` one-by-one and records
measured local latency (mean/p50/p95/p99) and throughput. Production rates are
not extrapolated from that measurement.

**Consequences.** Offline/streaming consistency holds when drift updates are
disabled and histories are complete. Enabling adaptive updates intentionally
diverges subsequent profile-derived features from frozen `features.parquet`.
Out-of-order events for an entity raise `StreamingOrderError`. Live injection
warms rolling history from the corpus then calls the same path.

---

## ADR-32 — Live simulator uses generator injectors + session alert overlay
**Status:** Accepted (Phase 11)

**Context.** Phase 9 wired `process_injection` to the real streaming engine, but
the dashboard still synthesised attack rows from `mock_data` and never surfaced
resulting alerts in Overview / Alert Queue / Entity Investigation.

**Decision.** Add `src.generator.live_injection.synthesize_live_attack`, which
rebuilds entity digital twins from `entities.json` via `record_to_profile` and
calls the Phase 3 injectors. The simulator posts scored alerts into a Streamlit
session overlay (`dashboard.live_state`) that `DashboardDataProvider` merges on
read. Contribution charts use the real provider / `top_contributors` from the
pipeline alert. Mock injection remains only as a fixture fallback when artifacts
are absent.

**Consequences.** Demo injections are honest mutations of real entities and
appear in SOC pages for the session without rewriting parquet. Clearing the
overlay restores the persisted artifact view.

---

## ADR-33 — Live campaigns are demo-scaled, not corpus-scaled
**Status:** Accepted (Phase 11)

**Context.** The first working live simulator reused the `config.yaml` campaign
shapes unchanged. Those ranges are sized for a 21-90 day corpus, and two things
broke when they were applied to a single interactive injection. Credential
stuffing generated 141 events and took ~15s to score in the browser request,
and insider drift generated ~79 events but had only the few days after the
corpus to spread them over, arriving at ~13 events/day. A 35-day gradual
privilege ramp delivered in under a week is not gradual: it presented as a
resource-breadth explosion, the classifier named it `LATERAL_MOVEMENT`, and it
raised 12 CRITICAL alerts — the exact opposite of the ambiguous, below-threshold
edge case ADR-9 and ADR-25 define it to be.

**Decision.** `src/generator/live_injection.py` holds `_LIVE_CAMPAIGN_SHAPES`,
demo-scaled overrides merged over the attack config for live injections only.
The injectors clamp against these same keys, so `campaign_size_range` continues
to agree with what is actually generated. Credential stuffing is sized as
victims x attempts because fan-out, not depth, is its defining shape. Slow-burn
types no longer receive a `max_days` clamp: a live campaign has no simulation
horizon to fit inside, so the day range governs and the ramp stays gradual.
The offline dataset is untouched.

**Consequences.** Every scenario now scores in under ~3s, and insider drift
lands in the HIGH watch band without alerting, which is the documented intent.
`tests/test_live_injection.py` pins campaign size, slow-burn density and
stuffing fan-out so a future config change cannot silently restore the stall or
the burst. `scripts/verify_live_demo.py` runs all seven scenarios end-to-end.
