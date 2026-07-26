

You can create a file in your repo called `project.md` and paste the following into it. Then Cursor can continuously refer to it.

# AEGIS — Project Overview for Cursor

## 1. Project Goal

Build a selection-grade prototype for Honeywell's **AI-Powered Behavioral Anomaly Detection for Cybersecurity** problem.

The system models normal access and connection behavior for individual users, service accounts, and edge devices, then processes new events to:

```text
1. Detect abnormal behavior
2. Determine what type of attack it resembles
3. Assign an explainable 0–100 risk score
4. Explain exactly why the event was flagged
5. Surface the alert in an analyst-facing SOC dashboard
```

The system must explicitly address:

```text
Sequential behavior
Extreme class imbalance
False-positive control
Cold-start entities
Concept drift
Explainability
Near-real-time inference
Production scalability
```

The prototype uses synthetic data as required by the problem statement.

---

# 2. Core Product Flow

```text
                    SYNTHETIC ENTERPRISE
                           │
                           ▼
                    Generate Entities
                           │
                           ▼
                 Generate Normal Activity
                           │
                           ▼
                    Inject Attacks
                           │
                           ▼
                    Historical Dataset
                           │
                           ▼
                  Temporal Train/Test Split
                           │
                           ▼
                 Build Entity Profiles
                           │
                           ▼
                    Feature Engine
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
       Anomaly Detector             Sequence Model
       Isolation Forest              Markov Model
             │                            │
             └─────────────┬──────────────┘
                           ▼
                    Attack Classifier
                           │
                           ▼
                       Risk Engine
                           │
                           ▼
                  Explanation Engine
                           │
                           ▼
                         Alert
                           │
                           ▼
                     SOC Dashboard
```

For a new live event:

```text
NEW EVENT
   ↓
Find entity
   ↓
Load behavioral profile
   ↓
Calculate behavioral deviations
   ↓
Isolation Forest anomaly score
   ↓
Sequence anomaly score
   ↓
Attack classification
   ↓
Risk score 0–100
   ↓
Generate reasons
   ↓
Alert / no alert
   ↓
Dashboard
   ↓
Safely update behavioral profile
```

---

# 3. Technology Stack

Keep the stack intentionally simple.

```text
Language            Python

Data                 Pandas / NumPy
Synthetic Data       Faker

ML                   scikit-learn

Anomaly Detection    Isolation Forest
Classification       Random Forest
Sequence Detection   Markov transition probabilities

Visualization        Plotly
Dashboard            Streamlit

Model Persistence    joblib

Data Storage         Parquet / CSV

Testing              pytest
```

Do NOT introduce unnecessary infrastructure without explicit approval.

Avoid initially:

```text
TensorFlow
PyTorch
Transformers
Kafka
Redis
React
FastAPI
Docker
LLM APIs
Databases
Cloud services
```

The architecture should allow these later, but the prototype must remain easy to run locally.

---

# 4. Repository Structure

Cursor should gradually build toward:

```text
aegis/
│
├── app.py
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── generated/
│
├── models/
│
├── src/
│   ├── generator/
│   │   ├── entities.py
│   │   ├── normal_behavior.py
│   │   ├── attacks.py
│   │   └── generator.py
│   │
│   ├── profiling/
│   │   ├── entity_profile.py
│   │   └── cohort_profile.py
│   │
│   ├── features/
│   │   ├── behavioral.py
│   │   ├── geographic.py
│   │   ├── temporal.py
│   │   └── sequence.py
│   │
│   ├── models/
│   │   ├── anomaly_detector.py
│   │   ├── attack_classifier.py
│   │   └── sequence_model.py
│   │
│   ├── detection/
│   │   ├── detector.py
│   │   └── risk_engine.py
│   │
│   ├── explainability/
│   │   └── explainer.py
│   │
│   └── adaptation/
│       ├── cold_start.py
│       └── drift.py
│
├── dashboard/
│   ├── overview.py
│   ├── alerts.py
│   ├── entity_view.py
│   ├── simulator.py
│   └── performance.py
│
├── evaluation/
│   └── evaluate.py
│
└── tests/
```

Don't create empty complexity just to match the structure. Files should appear as their functionality is implemented.

---

# 5. Data Schema

Every access event should approximately follow:

```text
event_id

entity_id
entity_type
department
role

timestamp

source_ip
geo_location
latitude
longitude

device_id
device_os
device_fingerprint

resource_accessed
resource_sensitivity

auth_method
auth_success

session_duration

protocol
bytes_transferred

command_sequence

is_anomaly
attack_type
```

`is_anomaly` and `attack_type` are ground-truth fields.

### Critical rule

Ground-truth fields must **never become model input features accidentally**.

They exist for training/evaluation only.

---

# 6. Synthetic Entities

Generate three primary entity classes:

```text
User
Service Account
Edge Device
```

Initial target:

```text
700 Users
150 Service Accounts
150 Edge Devices
--------------------
1000 entities
```

Each entity should have a persistent behavioral personality.

Example:

```text
USER_0042

Type:
user

Department:
engineering

Home:
Chennai

Working hours:
08:30–18:00

Known devices:
DEV_102
DEV_184

Resources:
EMAIL
GITHUB
DEV_SERVER

Authentication:
password

Typical session:
20–90 minutes
```

Normal behavior should contain realistic variation rather than perfectly repeating patterns.

---

# 7. Attack Taxonomy

All seven required behaviors must eventually work.

### Brute Force

```text
One account/source
↓
many authentication attempts
↓
short period
↓
high failure rate
```

### Impossible Travel

```text
Entity: USER_42

10:00
Chennai

↓

10:45
Frankfurt
```

Create physically implausible geographic velocity.

### Credential Stuffing

```text
Small number of attacker IPs
↓
many entity IDs
↓
many failed authentications
```

### Lateral Movement

```text
Compromised entity
↓
unusual resource
↓
another unusual resource
↓
sensitive server
```

### Device Spoofing

```text
Known entity

↓

unexpected fingerprint / OS / device
```

### Low-and-Slow Exfiltration

Gradually increase unusual resource access / transferred bytes over an extended period.

### Insider Drift

Slow legitimate expansion of resource/privilege footprint.

This is intentionally ambiguous and useful for false-positive/concept-drift testing.

---

# 8. Behavioral Profiles

For each entity learn things such as:

```text
Typical login hours
Working-hour distribution

Known countries
Known geographic locations

Known devices
Device frequencies

Known resources
Resource frequencies

Authentication methods

Session-duration distribution

Transfer-volume distribution

Failure-rate baseline

Common resource transitions
```

These profiles represent:

> "What is normal for THIS entity?"

---

# 9. Features

The feature engine should eventually output signals including:

```text
hour_deviation
is_off_hours

geo_distance_from_baseline
geo_velocity
new_country
location_rarity

new_device
device_rarity
fingerprint_changed

new_resource
resource_rarity
resource_sensitivity

auth_method_rarity

failed_auth_5m
failed_auth_1h
failure_rate

session_duration_zscore

bytes_transfer_zscore

event_frequency_5m
event_frequency_1h

resource_transition_probability
sequence_anomaly_score
```

Feature functions should remain independently testable.

---

# 10. Detection

Primary anomaly model:

```text
Isolation Forest
```

It answers:

> "How unusual is this event?"

Return a normalized:

```text
anomaly_score ∈ [0,1]
```

Do not use ground-truth labels to train the unsupervised detector.

---

# 11. Sequence Model

Start simple.

Use resource-transition probabilities.

Example:

```text
LOGIN → EMAIL

P = 0.71
```

versus:

```text
LOGIN → ADMIN_DATABASE

P = 0.002
```

Convert unusual transitions into:

```text
sequence_anomaly_score
```

This provides sequence awareness without requiring a deep neural network.

---

# 12. Attack Classifier

Use:

```text
Random Forest
```

Its responsibility is different from the anomaly detector.

```text
Isolation Forest

"Something is strange."
```

versus:

```text
Random Forest

"This resembles lateral movement."
```

Return:

```text
predicted_attack
attack_confidence
```

---

# 13. Risk Engine

Produce:

```text
risk_score = 0–100
```

using transparent contributions from:

```text
Anomaly score
Sequence anomaly
Attack confidence
Behavioral deviation
Resource sensitivity
```

Severity:

```text
0–30      LOW
31–60     MEDIUM
61–80     HIGH
81–100    CRITICAL
```

The formula must be documented and deterministic.

---

# 14. Explainability

Every alert should contain structured reasons.

Example:

```text
Risk: 94
Attack: Impossible Travel

Reasons:

+ Location 7,100 km from normal region
+ Previous login occurred 47 minutes earlier
+ Implied velocity physically implausible
+ Device fingerprint never observed
+ Sensitive resource never previously accessed
+ Login outside normal working hours
```

Explainability must work without an LLM.

---

# 15. Cold Start

For entities without enough history:

```text
New Entity
   ↓
Determine entity type / department / role
   ↓
Use cohort baseline
   ↓
Gradually collect personal history
   ↓
Blend cohort + personal profile
```

Do not automatically classify every new entity as anomalous merely because it has no history.

---

# 16. Concept Drift

Profiles should gradually adapt to legitimate changes.

Use either:

```text
Rolling historical window
```

or

```text
Exponentially weighted statistics
```

Important:

> High-risk/untrusted events must not automatically update the baseline.

Otherwise attackers can poison normal behavior.

---

# 17. Dashboard

Build with Streamlit.

Required pages:

```text
Overview
Alerts
Entity Investigation
Attack Simulator
Model Performance
```

Overview should show:

```text
Events analyzed
Alerts generated
Critical alerts
Anomaly rate
Threat distribution
Risk timeline
Highest-risk entities
```

Alert page should support filtering.

Entity investigation should show:

```text
Entity history
Behavior profile
Known devices
Known locations
Known resources
Recent events
Risk factors
Alert timeline
```

---

# 18. Attack Simulator

Allow:

```text
Choose entity

Choose attack

Inject attack
```

The injected event must go through the **real inference pipeline**.

Never hard-code the resulting alert.

Example:

```text
USER_042
+
Impossible Travel
       ↓
Generate attack event
       ↓
process_event()
       ↓
Risk 96
       ↓
CRITICAL ALERT
```

This is an important demo feature.

---

# 19. Evaluation

Report:

```text
PR-AUC
ROC-AUC

Precision
Recall
F1

False Positive Rate

Recall @ Top 1% alert budget

Per-attack:
Precision
Recall
F1

Confusion Matrix
```

Because attacks are intentionally rare, do not use accuracy as the primary success metric.

---

# 20. Cursor Development Rules

This part is important. Put it in `PROJECT_PLAN.md`.

> Do not attempt to implement the entire project in one response.

For every task:

```text
1. Inspect existing repository
2. Modify the minimum necessary files
3. Keep functions modular
4. Add type hints where useful
5. Add docstrings
6. Add validation
7. Add/update tests
8. Run the relevant tests
9. Run the implemented module
10. Report what changed
11. Report exact command to verify it
12. Do not start the next task automatically
```

Also tell Cursor:

> Do not replace working modules unnecessarily. Extend them.

And:

> Never generate fake evaluation metrics or hard-coded detection results.

---

# Execution Jobs — Tiny Chunks

This is the part I'd actually follow while working.

## Phase A — Foundation

**Job A1:** Inspect repository and summarize existing files. **A2:** Create missing project directories only. **A3:** Create/update `requirements.txt`. **A4:** Create configuration system and random seed. **A5:** Create basic shared constants/types. **A6:** Run import smoke test.

**Checkpoint A:** project starts without errors.

---

## Phase B — Schema

**B1:** Define entity schema. **B2:** Define access-event schema. **B3:** Define attack labels. **B4:** Define validation functions. **B5:** Write schema tests. **B6:** Generate one manually constructed valid event and validate it.

**Checkpoint B:** one valid standardized cybersecurity event exists.

---

## Phase C — Entity Generator

**C1:** Generate user IDs. **C2:** Add departments/roles. **C3:** Add home locations. **C4:** Add working hours. **C5:** Assign normal resources. **C6:** Assign devices. **C7:** Assign authentication preferences. **C8:** Implement service-account profiles. **C9:** Implement edge-device profiles. **C10:** Generate 1,000 entities. **C11:** Validate distributions. **C12:** Save entity dataset.

**Checkpoint C:** `entities.parquet` exists and looks realistic.

---

## Phase D — Normal Event Generator

**D1:** Generate timestamps. **D2:** Generate normal login hours. **D3:** Generate source locations/IPs. **D4:** Generate devices. **D5:** Generate resource access. **D6:** Generate authentication events. **D7:** Generate session durations. **D8:** Generate protocols. **D9:** Generate transfer volumes. **D10:** Generate command/resource sequences. **D11:** Add realistic noise. **D12:** Generate 1,000 events first. **D13:** Validate them. **D14:** Scale to 100k+ only after validation.

**Checkpoint D:** normal historical activity works.

---

# Phase E — Attacks

Implement attacks **one at a time**.

**E1:** Brute-force injector. Test it.

**E2:** Impossible-travel injector. Test distance/time.

**E3:** Credential-stuffing injector. Verify many entities/few IPs.

**E4:** Lateral-movement injector. Verify unusual resource sequence.

**E5:** Device-spoofing injector. Verify fingerprint mismatch.

**E6:** Low-and-slow-exfiltration injector. Verify gradual behavior.

**E7:** Insider-drift injector. Verify gradual resource expansion.

**E8:** Combined attack orchestrator.

**E9:** Configure anomaly percentage to ~1–2%.

**E10:** Save ground truth separately.

**Checkpoint E:** all Honeywell behaviors can be simulated.

This is your **first major milestone**.

---

# Phase F — Dataset Preparation

**F1:** Sort events chronologically. **F2:** Create temporal split. **F3:** Check attack prevalence. **F4:** Check missing values. **F5:** Check impossible values. **F6:** Ensure labels aren't accidentally features. **F7:** Save train/test datasets.

**Checkpoint F:** clean ML-ready dataset.

---

# Phase G — Behavioral Profiles

Again, tiny chunks.

**G1:** Login-hour profile. **G2:** Location profile. **G3:** Device profile. **G4:** Resource profile. **G5:** Authentication profile. **G6:** Session-duration profile. **G7:** Transfer-volume profile. **G8:** Authentication-failure baseline. **G9:** Resource-transition profile. **G10:** Combine into `EntityProfile`. **G11:** Persist profiles. **G12:** Test profile retrieval.

**Checkpoint G:** ask for any entity ID and receive its normal behavior.

---

# Phase H — Feature Engineering

**H1:** Time deviation. **H2:** Off-hours feature. **H3:** New-device feature. **H4:** Device rarity. **H5:** New-resource feature. **H6:** Resource rarity. **H7:** New-location feature. **H8:** Geographic distance. **H9:** Geo velocity. **H10:** Authentication rarity. **H11:** Failed attempts over 5 minutes. **H12:** Failed attempts over 1 hour. **H13:** Session-duration deviation. **H14:** Transfer-volume deviation. **H15:** Resource-transition probability. **H16:** Sequence anomaly. **H17:** Combine into one feature vector. **H18:** Test on known normal event. **H19:** Test on known attack.

**Checkpoint H:** attacks visibly produce stronger deviations than ordinary events.

---

# Phase I — Anomaly ML

**I1:** Select numerical features. **I2:** Handle missing/infinite values. **I3:** Train Isolation Forest. **I4:** Save model. **I5:** Load model. **I6:** Generate raw anomaly score. **I7:** Normalize to 0–1. **I8:** Score test events. **I9:** Compare normal vs attacks. **I10:** Calculate PR-AUC. **I11:** Calculate Recall @ Top 1%. **I12:** Tune threshold/contamination carefully.

**Checkpoint I:** system can answer:

> "How abnormal is this event?"

---

# Phase J — Attack Classification

**J1:** Prepare labeled attack training data. **J2:** Train Random Forest. **J3:** Handle imbalance/class weights. **J4:** Predict attack class. **J5:** Return confidence. **J6:** Generate confusion matrix. **J7:** Calculate per-class F1. **J8:** Save classifier.

**Checkpoint J:** anomalous events receive meaningful attack labels.

---

# Phase K — Risk + Explanation

**K1:** Design transparent risk formula. **K2:** Normalize all inputs. **K3:** Generate 0–100 score. **K4:** Assign severity. **K5:** Generate time reason. **K6:** Generate geography reason. **K7:** Generate device reason. **K8:** Generate resource reason. **K9:** Generate sequence reason. **K10:** Rank reasons. **K11:** Produce structured explanation object.

**Checkpoint K:**

```text
USER_42
Impossible Travel
Risk 94
CRITICAL

Why:
...
```

---

# Phase L — Cold Start + Drift

**L1:** Create cohort profiles. **L2:** Detect insufficient-history entities. **L3:** Select appropriate cohort. **L4:** Blend cohort/personal profile. **L5:** Implement rolling/weighted updates. **L6:** Block suspicious events from profile updates. **L7:** Write cold-start test. **L8:** Write drift test.

**Checkpoint L:** both explicit Honeywell requirements are demonstrable.

---

# Phase M — Unified Inference

This is critical.

**M1:** Create `process_event()`. **M2:** Load entity profile. **M3:** Calculate features. **M4:** Run anomaly model. **M5:** Run sequence scoring. **M6:** Run classifier. **M7:** Calculate risk. **M8:** Generate explanations. **M9:** Create alert object. **M10:** Determine whether event exceeds alert threshold. **M11:** Add safe profile-update decision. **M12:** Integration test normal event. **M13:** Integration test attack.

**Checkpoint M:** one function runs the entire product.

---

# Phase N — Dashboard

Don't ask Cursor to build the whole dashboard at once.

**N1:** Streamlit shell/navigation. **N2:** Overview KPI cards. **N3:** Threat distribution. **N4:** Risk-over-time graph. **N5:** Top risky entities. **N6:** Alert table. **N7:** Alert filters. **N8:** Alert detail. **N9:** Entity search. **N10:** Entity behavior view. **N11:** Entity event timeline. **N12:** Risk-factor display. **N13:** Performance page. **N14:** Polish layout only after everything works.

**Checkpoint N:** usable SOC interface.

---

# Phase O — Attack Simulator

**O1:** Entity dropdown. **O2:** Attack dropdown. **O3:** Generate selected attack. **O4:** Pass through `process_event()`. **O5:** Display detection result. **O6:** Insert resulting alert into dashboard state. **O7:** Test all seven attacks.

**Checkpoint O:** live Honeywell demo ready.

---

# Phase P — Evaluation

**P1:** Detection precision/recall/F1. **P2:** PR-AUC. **P3:** ROC-AUC. **P4:** False-positive rate. **P5:** Recall @ Top 1%. **P6:** Per-attack metrics. **P7:** Confusion matrix. **P8:** Cold-start experiment. **P9:** Drift experiment. **P10:** Sequence-model ablation. **P11:** Save all results automatically.

**Checkpoint P:** every important Honeywell evaluation criterion has evidence.

---

# Phase Q — Submission

**Q1:** Final README. **Q2:** Architecture diagram. **Q3:** Synthetic-data assumptions. **Q4:** ML methodology. **Q5:** Evaluation results. **Q6:** Limitations. **Q7:** Production scalability design. **Q8:** Dashboard screenshots. **Q9:** Final report PDF. **Q10:** Honeywell presentation. **Q11:** Clean repository. **Q12:** Fresh-install test. **Q13:** Rehearse live demo.

---

# How you should actually use Cursor

Don't paste:

> Build AEGIS.

Instead give Cursor **one job ID at a time**.

For example, your first meaningful prompt should be:

```text
Read PROJECT_PLAN.md completely before making changes.

We are currently working ONLY on Job B1-B4: defining the core data schemas.

Inspect the existing repository first.

Implement standardized schemas/models for:
1. Entity
2. AccessEvent
3. AttackType

Follow PROJECT_PLAN.md exactly.

Important constraints:
- is_anomaly and attack_type are ground-truth fields and must be clearly separated from inference features.
- Support user, service_account and edge_device entity types.
- Keep implementation lightweight.
- Do not work on data generation, ML, dashboard, or future jobs yet.
- Add validation where appropriate.
- Add tests for this job.
- Run the tests after implementation.

When finished, report:
1. Files created/modified
2. Important design decisions
3. Tests executed and results
4. Exact command I can run to verify it
5. Any issue that must be fixed before Job B5

STOP after this job. Do not proceed to another phase.
```

That last line is important.

