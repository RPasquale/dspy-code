#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request


def post(url: str, data: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:19000', help='InferMesh base URL')
    ap.add_argument('--model', default='BAAI/bge-small-en-v1.5')
    ap.add_argument('--texts', type=int, default=256)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    texts = [f"Hello world {i}" for i in range(args.texts)]
    batches = [texts[i:i+args.batch] for i in range(0, len(texts), args.batch)]
    t0 = time.time()
    total = 0
    for b in batches:
        r = post(args.url.rstrip('/') + '/embed', {'model': args.model, 'inputs': b})
        total += len(r.get('vectors') or [])
    dt = time.time() - t0
    rate = total / dt if dt > 0 else 0.0
    print(json.dumps({'texts': len(texts), 'batch': args.batch, 'seconds': round(dt, 3), 'rate_sec': round(rate, 2)}))


if __name__ == '__main__':
    main()

