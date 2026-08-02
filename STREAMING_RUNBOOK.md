# Running the streaming stack locally

Kafka + Elasticsearch + Kibana on Docker Desktop, Windows. Roughly 20 minutes
end to end, most of it waiting for images to pull.

Both files belong in your project root, next to the `outputs` folder:

```
Npontu Assessment\
├── docker-compose.yml          <- new
├── STREAMING_RUNBOOK.md        <- new
├── ecommerce_customer_behaviour_analysis.ipynb
└── outputs\
    ├── kafka_producer.py
    ├── kafka_consumer.py
    ├── es_index_mapping.json
    └── events_stream.jsonl     <- produced by notebook section 7.3
```

---

## Before you start

**Give Docker enough memory.** Elasticsearch alone wants 1 GB of heap and will
die silently with less. Docker Desktop → Settings → Resources → set Memory to
**at least 4 GB**, then Apply & Restart.

**Check `events_stream.jsonl` exists.** It is written by notebook section 7.3.
If `outputs\events_stream.jsonl` is missing, re-run that cell first.

**Install the Python clients** into the same environment your notebook uses:

```powershell
conda activate npontu_churn
pip install kafka-python-ng elasticsearch
```

Use `kafka-python-ng`, not `kafka-python` — the original is unmaintained and
breaks on Python 3.12+. Both are imported as `kafka`, so no code changes.

---

## Step 1 — Start the stack

```powershell
cd "C:\Users\agyak\Desktop\Npontu Assessment"
docker compose up -d
```

First run pulls about 1.5 GB of images. Then watch until all three are healthy:

```powershell
docker compose ps
```

Wait for `npontu-kafka` and `npontu-elasticsearch` to show **healthy** (not just
"running"). Elasticsearch usually takes 40–60 seconds. If you get impatient and
run the producer early, you will see `NoBrokersAvailable`.

Verify each service by hand:

```powershell
# Kafka: should print nothing (no topics yet) rather than erroring
docker exec npontu-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list

# Elasticsearch: should return JSON with "status" : "green" or "yellow"
curl http://localhost:9200/_cluster/health
```

---

## Step 2 — Create the Elasticsearch index

Apply the explicit mapping before indexing anything. Without it Elasticsearch
infers `customer_id` as full-text, which silently breaks term aggregations.

PowerShell mangles `curl`, so use `curl.exe` explicitly:

```powershell
curl.exe -XPUT "http://localhost:9200/ecommerce-events" `
  -H "Content-Type: application/json" `
  --data-binary "@outputs/es_index_mapping.json"
```

Expect `{"acknowledged":true,...}`. If you get
`resource_already_exists_exception`, delete and retry:

```powershell
curl.exe -XDELETE "http://localhost:9200/ecommerce-events"
```

---

## Step 3 — Start the consumer

Open a **second** PowerShell window and leave it running:

```powershell
cd "C:\Users\agyak\Desktop\Npontu Assessment"
conda activate npontu_churn
python outputs\kafka_consumer.py --bootstrap localhost:9092 --topic ecommerce.events
```

It will sit waiting for messages. That is correct — start it before the producer
so it sees everything from the beginning.

---

## Step 4 — Run the producer

Back in the first window:

```powershell
python outputs\kafka_producer.py --bootstrap localhost:9092 --topic ecommerce.events
```

You should see `sent 10,000 events` ticking up to about 40,640, then
`done: ... events published to ecommerce.events`.

To watch it behave like live traffic instead of a bulk dump, throttle it:

```powershell
python outputs\kafka_producer.py --speed 0.01
```

That paces roughly 100 events per second, which makes the Kibana dashboard move
in real time — far more convincing as a demo.

---

## Step 5 — Confirm the data landed

```powershell
# How many documents are indexed?
curl.exe "http://localhost:9200/ecommerce-events/_count"

# Funnel counts straight out of Elasticsearch
curl.exe -XGET "http://localhost:9200/ecommerce-events/_search?size=0" `
  -H "Content-Type: application/json" `
  -d "{\"aggs\":{\"funnel\":{\"terms\":{\"field\":\"event_type\",\"size\":10}}}}"
```

The count should be lower than the number of events published — that gap is the
idempotency guard suppressing the deliberately re-injected duplicates, which is
exactly the behaviour section 7.4 of the notebook demonstrates.

Check the topic filled up too:

```powershell
docker exec npontu-kafka /opt/kafka/bin/kafka-run-class.sh `
  kafka.tools.GetOffsetShell --bootstrap-server localhost:9092 --topic ecommerce.events
```

---

## Step 6 — Build a dashboard in Kibana

1. Open <http://localhost:5601>
2. **Stack Management → Data Views → Create data view**
   - Name: `ecommerce-events`
   - Index pattern: `ecommerce-events`
   - Time field: `event_timestamp`
3. **Analytics → Visualize Library → Create visualization**

Four panels worth building, in rough order of value:

| Panel | Type | Configuration |
|---|---|---|
| Event volume over time | Area | X: `event_timestamp` (Date histogram, hourly); Y: Count |
| Funnel by stage | Horizontal bar | Y: Terms on `event_type`; X: Count |
| Device split | Pie | Slice by: Terms on `device_type` |
| Traffic source mix | Bar | X: Terms on `traffic_source`; Y: Count |

Save each, then add them to a Dashboard. Screenshot it — a live dashboard over
real indexed events is far more persuasive in an interview than a diagram.

**Note on the time filter.** Your events are dated 2025–2026. Kibana defaults to
"Last 15 minutes" and will show nothing. Set the range to **Last 2 years** or an
absolute window covering 2025-01-01 to 2026-06-30.

---

## Step 7 — Shut down

```powershell
docker compose down        # stops containers, keeps the data volumes
docker compose down -v     # also deletes the data -- full reset
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `NoBrokersAvailable` | Kafka not ready yet. `docker compose ps` and wait for **healthy**. |
| `ModuleNotFoundError: kafka` | Client not installed, or installed into a different environment. `conda activate npontu_churn` first, then `pip install kafka-python-ng`. |
| Elasticsearch container exits immediately | Not enough memory. Raise Docker Desktop to 4 GB+ and restart. |
| `curl: (52) Empty reply` on port 9200 | ES still starting. Wait 30 s and retry. |
| Kibana stuck on "Server is not ready yet" | Normal for the first 1–2 minutes while it initialises against ES. |
| Ports 9092 / 9200 / 5601 already in use | Something else is bound. `netstat -ano \| findstr :9092`, then stop that process or change the host-side port in `docker-compose.yml`. |
| Kibana shows no data | Time filter. Your events are 2025–2026; widen the range. |
| PowerShell `curl` behaves oddly | It aliases to `Invoke-WebRequest`. Use `curl.exe` explicitly. |

---

## What to say about this in the interview

Running the stack changes your claim, so update it. Rather than "I proposed an
architecture," you can now say:

> I ran the full pipeline locally on Docker — Kafka in KRaft mode, Elasticsearch,
> and Kibana. The producer replays roughly 40,000 cleaned clickstream events into
> a topic partitioned by customer ID, the consumer maintains hourly tumbling-window
> funnel metrics and bulk-indexes into Elasticsearch with an explicit keyword
> mapping, and Kibana serves the dashboard. The consumer's idempotency guard drops
> the duplicate events, so the indexed count is lower than the published count —
> which is the double-fired-tag problem from the cleaning stage, handled at the
> streaming layer instead of in batch.

Stay honest about the boundary. This is a **single-broker development stack**.
You have not tuned partition counts for throughput, handled consumer-group
rebalancing under load, configured retention policies, or run it under
production traffic. Saying so is stronger than implying operational experience
you do not have — and it turns a hard follow-up question into "good question,
here is how I would approach it."
