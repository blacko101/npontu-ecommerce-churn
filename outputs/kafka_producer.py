"""
Kafka producer: replays the cleaned clickstream into a topic as if it were live
traffic. Built on confluent-kafka, the officially maintained client.

Run:  python outputs/kafka_producer.py --speed 0.005
Requires: pip install confluent-kafka
"""
import argparse
import json
import sys
import time

from confluent_kafka import Producer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--topic", default="ecommerce.events")
    ap.add_argument("--file", default="outputs/events_stream.jsonl")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="seconds to sleep between events (0 = as fast as possible)")
    args = ap.parse_args()

    producer = Producer({
        "bootstrap.servers": args.bootstrap,
        "acks": "all",              # wait for the broker to confirm the write
        "retries": 3,
        "linger.ms": 20,            # small batching window for throughput
        "compression.type": "gzip",
    })

    delivered = failed = 0

    def on_delivery(err, msg):
        """Called once per message when the broker acknowledges (or rejects) it."""
        nonlocal delivered, failed
        if err is None:
            delivered += 1
        else:
            failed += 1
            if failed <= 5:
                print(f"  delivery failed: {err}", file=sys.stderr)

    sent = 0
    with open(args.file, encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            # Key by customer so all of one customer's events land on the same
            # partition, in order -- required for correct sessionisation.
            key = event.get("customer_id") or "guest"
            producer.produce(args.topic, key=key.encode("utf-8"),
                             value=json.dumps(event).encode("utf-8"),
                             on_delivery=on_delivery)
            sent += 1

            # Serve delivery callbacks and keep the internal queue from filling
            producer.poll(0)
            if sent % 10_000 == 0:
                producer.flush()
                print(f"sent {sent:,} events  (delivered {delivered:,})")
            if args.speed:
                time.sleep(args.speed)

    producer.flush()
    print(f"\ndone: {sent:,} events published to '{args.topic}'")
    print(f"      delivered {delivered:,}   failed {failed:,}")


if __name__ == "__main__":
    main()