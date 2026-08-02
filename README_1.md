# Analyzing Customer Behaviour for E-commerce Insights

**Npontu Technologies — Intelligent Systems Services Engineer, technical assignment**

Churn prediction on a synthetic Ghana-based e-commerce dataset: 867,265 records
across six related tables covering January 2025 to June 2026.

---

## Start here

Open **`ecommerce_customer_behaviour_analysis.ipynb`**. It runs top to bottom in
about four minutes and every cell is already executed, so the outputs and charts
are visible without running anything.

If you would rather run it yourself, `ecommerce_analysis_BLANK.ipynb` is the same
notebook with outputs cleared.

---

## What is in this folder

### Primary deliverables

| File | What it is |
|---|---|
| `ecommerce_customer_behaviour_analysis.ipynb` | The full analysis, executed. 87 cells across eight sections. |
| `ecommerce_analysis_BLANK.ipynb` | Same notebook, outputs cleared, for a fresh run. |
| `Npontu_Customer_Behaviour_Analysis.pptx` | 16-slide presentation of approach and findings. |
| `generate_ecommerce_dataset.py` | The script that produced the dataset. Reproducible from a fixed seed. |
| `DATA_DICTIONARY.md` | Schema for all six tables, plus every defect deliberately injected. |
| `Submission_Cover.docx` | One-page summary of the assignment, method and results. |

### The dataset

| File | Rows | Notes |
|---|---|---|
| `customers.csv` | 20,706 | Demographics, acquisition channel, loyalty tier |
| `products.csv` | 1,555 | Catalogue with price, cost, category |
| `orders.csv` | 62,880 | Order headers, payment, delivery |
| `order_items.csv` | 116,236 | Line items, quantity, discount |
| `events.csv` | 646,969 | Clickstream — the volume table, 78 MB |
| `reviews.csv` | 18,919 | Ratings and review text |

The data is deliberately dirty. Duplicates, mixed date formats, currency strings,
sentinel values, inconsistent categoricals and referential violations are all
injected on purpose so the cleaning stage has something genuine to solve. Every
defect is counted by the generator, which means the cleaning can be verified
rather than assumed. `DATA_DICTIONARY.md` lists them all.

### Streaming stack

| File | What it is |
|---|---|
| `docker-compose.yml` | Kafka (KRaft), Elasticsearch 8.13, Kibana — local dev stack |
| `kafka_producer.py` | Replays the cleaned clickstream into a Kafka topic |
| `kafka_consumer.py` | Hourly tumbling-window aggregation, bulk-indexes into Elasticsearch |
| `es_index_mapping.json` | Explicit index mapping (keyword fields, date parsing) |
| `events_stream_sample.jsonl` | 40,000 events in JSON Lines, ready to replay |
| `STREAMING_RUNBOOK.md` | Step-by-step instructions for running the stack |

### Generated at runtime

Section 8.4 of the notebook writes an `outputs/` folder containing the trained
model, the cleaned tables, the cleaning audit log, and `churn_risk_scores.csv` —
the ranked retention list that is the practical deliverable.

---

## A note on `_hidden_ground_truth.csv`

The generator writes this file: each customer's latent engagement score and the
churn flag that was designed into the simulation.

**It was never used as a model input.** It is written to a separate file
precisely so it cannot leak into the feature matrix, and its only legitimate use
is confirming after the fact that the engineered features recovered the
underlying structure. Section 4.5 of the notebook verifies independently that no
single feature correlates above 0.52 with the target, which is the real check
that the observation/prediction split held.

The file is excluded from this submission. Re-running the generator will
recreate it; it can be ignored or deleted.

---

## Approach

**Sections 1–2 — Exploration.** Profile the six tables and diagnose the data
quality problems before touching anything. The most consequential finding is the
distinction between *structural* missingness (72% of event `order_id`s are empty
because a page view has no order) and *genuine* missingness. Treating those the
same way would corrupt the analysis.

**Section 3 — Cleaning.** 281,569 logged corrections. Governing principles:
recover rather than impute where a reliable source exists, void rather than
invent where a value is unknowable, null the field rather than drop the row, and
log every change so the cleaning is auditable. Three distinct duplicate problems
are handled separately — exact rows, conflicting keys, and near-duplicate people
who re-registered under a new ID.

**Section 4 — Feature engineering.** 98 features across four data sources. The
central design decision is the observation/prediction split: features are built
only from data before a 2026-04-01 cutoff, and the label only from the 90 days
after it. Without this, recency effectively *is* the label and the model scores
near-perfect while being useless in production.

**Section 5 — Modelling.** Random Forest, benchmarked against a majority-class
dummy (the floor) and logistic regression (the baseline). Cross-validated,
tuned, and evaluated on a held-out 25% test set. Section 5.7 measures which data
source carries the model; section 5.8 prunes the 98 features down to 28 using
L1-penalised logistic regression, with no loss of performance.

**Section 6 — Insight.** Revenue trends, demand timing, the conversion funnel,
churn drivers, cohort retention, and RFM segmentation.

**Section 7 — Streaming architecture.** Kafka to Elasticsearch to Kibana, with
working producer and consumer scripts and a local demonstration.

**Section 8 — Conclusions.** Executive summary, six recommendations, limitations.

---

## Results

**Model** — Random Forest, held-out test set:

| Metric | Value |
|---|---|
| ROC-AUC | 0.846 |
| Recall at the tuned threshold | 0.892 |
| Precision at the tuned threshold | 0.715 |
| Cross-validated AUC | 0.844 (matches test — no overfitting to the training process) |

The decision threshold is tuned to 0.337 rather than the default 0.50, because a
missed churner costs the customer while a false alarm costs one voucher. That
raises recall from 68% to 89%.

**Feature-source ablation** — each block retrained in isolation:

| Block | Features | AUC |
|---|---|---|
| Demographics | 11 | 0.590 |
| Reviews | 4 | 0.606 |
| Purchase / RFM | 52 | 0.767 |
| Behavioural / clickstream | 31 | 0.839 |
| All features | 98 | 0.844 |

Clickstream behaviour alone essentially matches the full model. People go quiet
in their browsing before they stop buying, so the clickstream is the earlier
warning signal — which is what justifies the streaming architecture.

**Feature selection** — feature engineering and feature selection are different
steps, and the ablation showed the second one was needed: 67 of the 98 features
were buying 0.006 AUC. Two selectors were compared on the held-out set:

| Selector | Features | Test AUC |
|---|---|---|
| All features | 98 | 0.8456 |
| **L1 / Lasso (C=0.02)** | **28** | **0.8487** |
| Concept-grouped importance | 25 | 0.8471 |
| Behavioural block only | 31 | 0.8485 |

Pruning 71% of the features did not cost accuracy. The honest claim is *no loss*
rather than *a better model* — 0.003 is within fold-to-fold noise — but the
pruned model is explainable, cheaper to maintain, and its feature importances
are interpretable now that the redundant groups masking one another are gone.
The framing is: engineer broadly to explore, select to ship.

**Business** — targeting the riskiest 20% of customers produces a list that is
98% churners against a 57% base rate.

---

## Reproducing the analysis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
jupyter notebook ecommerce_customer_behaviour_analysis.ipynb
```

The notebook locates the CSVs automatically if they sit beside it, or set
`ECOM_DATA_DIR` to point at them. Everything is seeded, so a re-run reproduces
the figures above exactly.

To regenerate the dataset from scratch:

```bash
python generate_ecommerce_dataset.py
```

---

## Running the streaming stack

Requires Docker Desktop with at least 4 GB of memory. Full instructions are in
`STREAMING_RUNBOOK.md`; the short version:

```bash
docker compose up -d                     # wait for kafka and elasticsearch to report healthy
curl -XPUT localhost:9200/ecommerce-events \
     -H "Content-Type: application/json" -d @es_index_mapping.json
pip install confluent-kafka "elasticsearch>=8,<9"

python kafka_consumer.py                 # window 1 — leave running
python kafka_producer.py --speed 0.005   # window 2
```

Kibana is at <http://localhost:5601>. Create a data view on `ecommerce-events`
with `event_timestamp` as the time field, and set the time range to cover June
2026 — the replayed slice is the most recent 40,000 events, not the full
eighteen months.

**This stack was run, not just specified.** Verified locally: 40,640 events
consumed, 640 duplicates suppressed by the consumer's idempotency guard, 40,000
documents indexed into Elasticsearch, 591 hourly windows built. Those 640
duplicates are the double-fired analytics tags found during cleaning, now handled
at the streaming layer rather than in batch. The Kibana dashboard over the indexed
events is shown in the presentation.

---

## Known limitations

- **Synthetic data.** Effect sizes are indicative rather than measured. The
  pipeline transfers unchanged; the coefficients would need re-estimating.
- **A single cutoff date.** Production should use rolling-origin validation
  across several cutoffs to confirm stability over time.
- **Cost-blind threshold.** F1 treats both error types as equal. Real lifetime
  value and voucher cost figures would allow minimising expected cost instead.
- **Random Forest was not a decisive win.** Logistic regression matched it on AUC
  at a fraction of the training cost. Gradient boosting is the obvious next
  benchmark and would handle missing values natively.
- **A sessionisation gap.** Carts and purchases are not stitched into the same
  session in the source data, so session-level funnel metrics are unmeasurable.
  Section 6.4 reports the customer-level figure and documents the gap rather than
  quoting a number that cannot be trusted.
- **Correlation, not causation.** Every driver in section 6 is an association.
  Confirming any of them would require a controlled experiment.
- **Single-broker dev stack.** The streaming setup has not been tuned for
  throughput, tested under load, or configured for consumer-group rebalancing.


