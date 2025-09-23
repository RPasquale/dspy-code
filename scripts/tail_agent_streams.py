from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def tail_reddb(stream: str, start: int = 0, interval: float = 1.0, workspace: Path | None = None) -> None:
    from dspy_agent.db.factory import get_storage
    st = get_storage()
    off = start
    while True:
        rows = list(st.read(stream, start=off, count=100))  # type: ignore
        if rows:
            for off, val in rows:
                try:
                    print(json.dumps({"offset": off, "value": val}, ensure_ascii=False))
                except Exception:
                    print({"offset": off, "value": str(val)})
            off = rows[-1][0] + 1
        time.sleep(interval)


def tail_kafka(topic: str, bootstrap: str | None = None) -> None:
    try:
        from confluent_kafka import Consumer
    except Exception as e:
        print("Kafka not available:", e)
        return
    conf = {
        'bootstrap.servers': bootstrap or 'localhost:9092',
        'group.id': 'dspy-tail',
        'auto.offset.reset': 'latest',
        'enable.partition.eof': False,
    }
    c = Consumer(conf)
    c.subscribe([topic])
    try:
        while True:
            msg = c.poll(0.5)
            if msg is None:
                continue
            if msg.error():
                continue
            try:
                print(msg.value().decode('utf-8', errors='ignore'))
            except Exception:
                print(str(msg.value()))
    except KeyboardInterrupt:
        pass
    finally:
        try: c.close()
        except Exception: pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Tail agent streams from RedDB or Kafka")
    ap.add_argument('--stream', default='agent.actions', help='RedDB stream to tail (default: agent.actions)')
    ap.add_argument('--kafka', action='store_true', help='Tail Kafka topic instead of RedDB stream')
    ap.add_argument('--bootstrap', default=None, help='Kafka bootstrap servers (default: localhost:9092)')
    ap.add_argument('--start', type=int, default=0, help='Start offset for RedDB stream')
    ap.add_argument('--interval', type=float, default=1.0, help='Polling interval for RedDB stream')
    args = ap.parse_args()

    if args.kafka:
        tail_kafka(args.stream, args.bootstrap)
    else:
        tail_reddb(args.stream, start=args.start, interval=args.interval)


if __name__ == '__main__':
    main()

