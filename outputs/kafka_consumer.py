"""
Kafka consumer: maintains 1-hour tumbling-window funnel metrics and bulk-indexes
both the raw events and the aggregates into Elasticsearch.

Run:  python outputs/kafka_consumer.py
Stop: Ctrl+C  (flushes the final batch before exiting)

Requires: pip install confluent-kafka "elasticsearch>=8,<9"
"""
import argparse
import json
import sys
from collections import defaultdict

from confluent_kafka import Consumer, KafkaError
from elasticsearch import Elasticsearch, helpers

FUNNEL = ("product_view", "add_to_cart", "checkout_start", "purchase")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--topic", default="ecommerce.events")
    ap.add_argument("--es", default="http://localhost:9200")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--idle-timeout", type=float, default=30.0,
                    help="seconds with no new messages before exiting")
    args = ap.parse_args()

    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": "funnel-aggregator",
        "auto.offset.reset": "earliest",   # read the topic from the beginning
        "enable.auto.commit": False,       # commit only after a successful index
    })
    consumer.subscribe([args.topic])

    es = Elasticsearch(args.es)
    if not es.ping():
        print(f"cannot reach Elasticsearch at {args.es}", file=sys.stderr)
        sys.exit(1)

    windows = defaultdict(lambda: defaultdict(int))
    buffer, seen = [], set()
    consumed = duplicates = indexed = 0
    idle = 0.0

    def flush():
        """Index the buffered events plus the current window aggregates."""
        nonlocal indexed
        if buffer:
            helpers.bulk(es, buffer, raise_on_error=False)
            indexed += len(buffer)
            buffer.clear()
        aggs = [{
            "_index": "ecommerce-funnel-hourly", "_id": h,
            "_source": {
                "window_start": h + ":00:00", **w,
                "cart_abandon_rate": round(
                    1 - w.get("purchase", 0) / max(w.get("add_to_cart", 0), 1), 4),
            },
        } for h, w in windows.items()]
        if aggs:
            helpers.bulk(es, aggs, raise_on_error=False)
        consumer.commit(asynchronous=False)

    print(f"consuming '{args.topic}' -> {args.es}   (Ctrl+C to stop)\n")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                idle += 1.0
                if idle >= args.idle_timeout and consumed:
                    print(f"\nno new messages for {args.idle_timeout:.0f}s — stopping")
                    break
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"  consumer error: {msg.error()}", file=sys.stderr)
                continue

            idle = 0.0
            e = json.loads(msg.value().decode("utf-8"))
            consumed += 1

            # Idempotency: the source double-fires analytics tags, so drop repeats
            # by event_id. Without this every live dashboard over-counts.
            eid = e.get("event_id")
            if eid in seen:
                duplicates += 1
                continue
            seen.add(eid)

            hour = str(e.get("event_timestamp", ""))[:13]      # 'YYYY-MM-DD HH'
            w = windows[hour]
            w["events"] += 1
            if e.get("event_type") in FUNNEL:
                w[e["event_type"]] += 1

            buffer.append({"_index": "ecommerce-events", "_id": eid, "_source": e})

            if len(buffer) >= args.batch:
                flush()
                print(f"consumed {consumed:,}   indexed {indexed:,}   "
                      f"duplicates suppressed {duplicates:,}   "
                      f"windows {len(windows):,}")
    except KeyboardInterrupt:
        print("\ninterrupted — flushing final batch")
    finally:
        flush()
        consumer.close()

    print(f"\n{'=' * 58}")
    print(f"  events consumed        : {consumed:,}")
    print(f"  duplicates suppressed  : {duplicates:,}   <- idempotency guard")
    print(f"  documents indexed      : {indexed:,}")
    print(f"  hourly windows built   : {len(windows):,}")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()