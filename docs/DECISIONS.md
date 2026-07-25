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
