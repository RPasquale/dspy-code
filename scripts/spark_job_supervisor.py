from __future__ import annotations

import os
import time
import shlex
import subprocess
from typing import List, Dict, Any

try:
    from dspy_agent.streaming.events import log_spark_app, log_training_dataset
except Exception:  # pragma: no cover
    def log_spark_app(event: str, name: str, state: str = "", namespace: str = "default"):  # type: ignore
        pass
    def log_training_dataset(kind: str, detail):  # type: ignore
        pass


SPARK_BIN = os.environ.get('SPARK_SUBMIT', '/opt/bitnami/spark/bin/spark-submit')
MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')
WAREHOUSE = os.environ.get('WAREHOUSE_BASE', '/warehouse')
CHECKPOINTS = os.environ.get('CHECKPOINT_BASE', '/checkpoints')
LANDING = os.environ.get('LANDING_BASE', '/landing')


def run(cmd: list[str]) -> subprocess.Popen:
    print('[spark-job] starting:', ' '.join(cmd))
    return subprocess.Popen(cmd)


def _app_name(app: str) -> str:
    try:
        base = app.split('/')[-1]
        if base.endswith('.py'):
            base = base[:-3]
        return base or app
    except Exception:
        return app


def submit(app: str, *args: str, streaming: bool = False) -> subprocess.Popen:
    # Dynamic allocation & AQE confs (standalone-friendly with shuffle tracking)
    min_exec = os.getenv('DA_MIN_EXECUTORS', '1')
    max_exec = os.getenv('DA_MAX_EXECUTORS', '4')
    init_exec = os.getenv('DA_INITIAL_EXECUTORS', '1')
    base_cmd = [
        SPARK_BIN,
        '--master', MASTER,
        '--conf', 'spark.sql.adaptive.enabled=true',
        '--conf', 'spark.dynamicAllocation.enabled=true',
        '--conf', 'spark.dynamicAllocation.shuffleTracking.enabled=true',
        '--conf', f'spark.dynamicAllocation.minExecutors={min_exec}',
        '--conf', f'spark.dynamicAllocation.maxExecutors={max_exec}',
        '--conf', f'spark.dynamicAllocation.initialExecutors={init_exec}',
        app
    ]
    if args:
        base_cmd.extend(list(args))
    proc = run(base_cmd)
    try:
        ns = os.getenv('SPARK_NAMESPACE', os.getenv('K8S_NAMESPACE', 'local'))
        log_spark_app('submitted', _app_name(app), state='SUBMITTED', namespace=ns)
    except Exception:
        pass
    # For streaming apps, keep process reference; batch jobs can complete
    return proc


def main() -> None:
    # Env toggles
    ingest_csv = os.getenv('INGEST_CSV', '1') == '1'
    ingest_excel = os.getenv('INGEST_EXCEL', '1') == '1'
    ingest_json = os.getenv('INGEST_JSON', '1') == '1'
    ingest_parquet = os.getenv('INGEST_PARQUET', '1') == '1'
    ingest_avro = os.getenv('INGEST_AVRO', '1') == '1'
    ingest_docs = os.getenv('INGEST_DOCS', '1') == '1'

    # Start long-running streaming jobs
    streams: List[Dict[str, Any]] = []
    if ingest_csv:
        streams.append({'name': 'file_ingest_csv', 'proc': submit('local:///opt/app/streaming/topics/file_ingest_csv.py', streaming=True)})
    if ingest_json:
        streams.append({'name': 'file_ingest_json', 'proc': submit('local:///opt/app/streaming/topics/file_ingest_json.py', streaming=True)})
    if ingest_parquet:
        streams.append({'name': 'file_ingest_parquet', 'proc': submit('local:///opt/app/streaming/topics/file_ingest_parquet.py', streaming=True)})
    if ingest_avro:
        streams.append({'name': 'file_ingest_avro', 'proc': submit('local:///opt/app/streaming/topics/file_ingest_avro.py', streaming=True)})
    if ingest_docs:
        streams.append({'name': 'file_ingest_docs', 'proc': submit('local:///opt/app/streaming/topics/file_ingest_docs.py', streaming=True)})

    # Optional: stream tool join groups if Kafka configured
    kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP') or os.getenv('KAFKA_BOOTSTRAP_SERVERS')
    if kafka_bootstrap:
        streams.append({'name': 'tool_join_groups', 'proc': submit('local:///opt/app/streaming/topics/tool_join_groups.py', streaming=True)})

    # Trainer config
    trainer_type = os.getenv('TRAINER_TYPE', 'tiny').lower()  # 'tiny' or 'hf'
    hf_model = os.getenv('HF_MODEL', 'mistralai/Mistral-7B-Instruct')
    hf_epochs = os.getenv('HF_EPOCHS', '1')
    hf_batch = os.getenv('HF_BATCH', '2')
    hf_maxlen = os.getenv('HF_MAXLEN', '1024')
    hf_lr = os.getenv('HF_LR', '1e-5')

    # Periodic batch jobs (every N minutes)
    last_promote = 0.0
    last_metrics = 0.0
    last_features = 0.0
    last_train = 0.0
    last_vectorize = 0.0
    last_export = 0.0
    last_code_ds = 0.0
    last_code_train = 0.0
    promote_iv = int(os.getenv('PROMOTE_INTERVAL_SEC', '600'))
    metrics_iv = int(os.getenv('METRICS_INTERVAL_SEC', '900'))
    features_iv = int(os.getenv('FEATURES_INTERVAL_SEC', '900'))
    train_iv = int(os.getenv('TRAIN_INTERVAL_SEC', '86400'))
    train_min_fresh = int(os.getenv('TRAIN_MIN_FRESH_SEC', '600'))
    train_min_shards = int(os.getenv('TRAIN_MIN_SHARDS', '1'))
    train_min_rows = int(os.getenv('TRAIN_MIN_ROWS', '100'))
    vectorize_iv = int(os.getenv('VECTORIZE_INTERVAL_SEC', '900'))
    export_iv = int(os.getenv('EXPORT_INTERVAL_SEC', '1800'))
    code_ds_iv = int(os.getenv('CODELOG_BUILD_INTERVAL_SEC', '3600'))
    code_trainer = os.getenv('CODELOG_TRAINER', 'hf').lower()  # 'hf' or 'tiny'
    code_model = os.getenv('CODELOG_MODEL', 'Salesforce/codet5p-220m')
    code_epochs = os.getenv('CODELOG_EPOCHS', '1')
    code_batch = os.getenv('CODELOG_BATCH', '4')
    code_max_code = os.getenv('CODELOG_MAX_CODE', '1024')
    code_max_log = os.getenv('CODELOG_MAX_LOG', '256')
    code_train_iv = int(os.getenv('CODELOG_TRAIN_INTERVAL_SEC', '86400'))
    code_min_shards = int(os.getenv('CODELOG_MIN_SHARDS', '5'))

    batch_jobs: List[Dict[str, Any]] = []
    while True:
        now = time.time()
        # Restart any dead streaming processes
        for i, item in enumerate(list(streams)):
            p = item['proc']
            if p.poll() is not None:
                try:
                    ns = os.getenv('SPARK_NAMESPACE', os.getenv('K8S_NAMESPACE', 'local'))
                    log_spark_app('stopped', item['name'], state='FAILED' if p.returncode else 'STOPPED', namespace=ns)
                except Exception:
                    pass
                print('[spark-job] restarting streaming job', item['name'])
                streams.pop(i)
                newp = submit(f"local:///opt/app/streaming/topics/{item['name']}.py", streaming=True)
                streams.insert(i, {'name': item['name'], 'proc': newp})
        # Submit promotion
        if now - last_promote >= promote_iv:
            p = submit('local:///opt/app/streaming/topics/file_promote_approvals.py')
            batch_jobs.append({'name': 'file_promote_approvals', 'proc': p, 't0': now})
            last_promote = now
        # Submit metrics
        if now - last_metrics >= metrics_iv:
            p = submit('local:///opt/app/streaming/topics/tool_metrics.py')
            batch_jobs.append({'name': 'tool_metrics', 'proc': p, 't0': now})
            last_metrics = now
        # Submit file feature builder
        if now - last_features >= features_iv:
            p = submit('local:///opt/app/streaming/topics/feature_build_ingest_files.py')
            batch_jobs.append({'name': 'feature_build_ingest_files', 'proc': p, 't0': now})
            last_features = now
        # Submit vectorizer for GRPO tokens
        if now - last_vectorize >= vectorize_iv:
            p = submit('local:///opt/app/streaming/topics/tool_vectorize_tokens.py')
            batch_jobs.append({'name': 'tool_vectorize_tokens', 'proc': p, 't0': now})
            last_vectorize = now
        # Submit batch export to build dataset manifest automatically
        if now - last_export >= export_iv:
            p = submit('local:///opt/app/streaming/topics/tool_batch_export.py')
            batch_jobs.append({'name': 'tool_batch_export', 'proc': p, 't0': now})
            last_export = now
        # Build code→log dataset periodically
        if now - last_code_ds >= code_ds_iv:
            p = submit('local:///opt/app/streaming/topics/code_log_build_dataset.py')
            batch_jobs.append({'name': 'code_log_build_dataset', 'proc': p, 't0': now})
            last_code_ds = now
        # Submit training (tiny or HF) periodically
        if now - last_train >= train_iv:
            manifest = os.path.join(WAREHOUSE, 'datasets', 'grpo_tool_batches', 'manifest.json')
            # Safety: require fresh manifest newer by train_min_fresh seconds since last_train
            fresh_ok = True
            try:
                mtime = os.path.getmtime(manifest)
                if last_train > 0 and (mtime - last_train) < train_min_fresh:
                    fresh_ok = False
            except Exception:
                fresh_ok = False
            if fresh_ok:
                # Threshold guard: minimum shards/rows
                try:
                    import json as _json
                    paths = _json.loads(open(manifest).read()) if os.path.exists(manifest) else []
                    shards = len(paths or [])
                    rows = sum(int(x.get('rows', 0)) for x in (paths or []))
                    if shards < train_min_shards or rows < train_min_rows:
                        fresh_ok = False
                except Exception:
                    fresh_ok = False

            if fresh_ok:
                if trainer_type == 'hf':
                    p = submit('local:///opt/app/dspy_agent/training/train_grpo_tool_hf.py', '--manifest', manifest, '--model', hf_model, '--epochs', hf_epochs, '--batch-size', hf_batch, '--max-len', hf_maxlen, '--lr', hf_lr)
                    batch_jobs.append({'name': 'train_grpo_tool_hf', 'proc': p, 't0': now})
                else:
                    p = submit('local:///opt/app/dspy_agent/training/train_grpo_tool.py', '--manifest', manifest, '--epochs', '1', '--batch-size', '16')
                    batch_jobs.append({'name': 'train_grpo_tool', 'proc': p, 't0': now})
                last_train = now

        # Train code→log model periodically if dataset ready
        if now - last_code_train >= code_train_iv:
            try:
                ds_dir = os.path.join(WAREHOUSE, 'datasets', 'code_log_predict')
                shards = 0
                for root, dirs, files in os.walk(ds_dir):
                    for f in files:
                        if f.endswith('.parquet'):
                            shards += 1
                if shards >= code_min_shards:
                    if code_trainer == 'hf':
                        p = submit('local:///opt/app/dspy_agent/training/train_code_to_log_hf.py', '--dataset-dir', ds_dir, '--model', code_model, '--epochs', code_epochs, '--batch', code_batch, '--max-code', code_max_code, '--max-log', code_max_log)
                    else:
                        p = submit('local:///opt/app/dspy_agent/training/train_code_to_log.py', '--dataset-dir', ds_dir, '--epochs', code_epochs, '--batch-size', code_batch, '--max-code', code_max_code, '--max-log', code_max_log)
                    batch_jobs.append({'name': 'train_code_to_log', 'proc': p, 't0': now})
                    last_code_train = now
            except Exception:
                pass

        # Immediate trigger via control file for code→log trainer
        ctrl_code = os.path.join(WAREHOUSE, 'controls', 'train_code_log.json')
        if os.path.exists(ctrl_code):
            try:
                import json as _json
                with open(ctrl_code, 'r') as fh:
                    cfg = _json.load(fh)
                ds_dir = cfg.get('dataset_dir') or os.path.join(WAREHOUSE, 'datasets', 'code_log_predict')
                trainer = (cfg.get('trainer') or code_trainer).lower()
                args = cfg.get('args') or {}
                if trainer == 'hf':
                    model = args.get('model') or code_model
                    epochs = str(args.get('epochs') or code_epochs)
                    batch = str(args.get('batch') or code_batch)
                    max_code = str(args.get('max_code') or code_max_code)
                    max_log = str(args.get('max_log') or code_max_log)
                    p = submit('local:///opt/app/dspy_agent/training/train_code_to_log_hf.py', '--dataset-dir', ds_dir, '--model', model, '--epochs', epochs, '--batch', batch, '--max-code', max_code, '--max-log', max_log)
                else:
                    epochs = str(args.get('epochs') or code_epochs)
                    batch = str(args.get('batch') or code_batch)
                    max_code = str(args.get('max_code') or code_max_code)
                    max_log = str(args.get('max_log') or code_max_log)
                    p = submit('local:///opt/app/dspy_agent/training/train_code_to_log.py', '--dataset-dir', ds_dir, '--epochs', epochs, '--batch-size', batch, '--max-code', max_code, '--max-log', max_log)
                batch_jobs.append({'name': 'train_code_to_log', 'proc': p, 't0': time.time()})
                os.remove(ctrl_code)
            except Exception as e:
                print('[spark-job] code-log control error:', e)

        # Immediate trigger via control file
        ctrl_path = os.path.join(WAREHOUSE, 'controls', 'train_tool.json')
        if os.path.exists(ctrl_path):
            try:
                import json
                with open(ctrl_path, 'r') as fh:
                    cfg = json.load(fh)
                trainer = (cfg.get('trainer') or trainer_type).lower()
                args = cfg.get('args') or {}
                manifest = args.get('manifest') or os.path.join(WAREHOUSE, 'datasets', 'grpo_tool_batches', 'manifest.json')
                if trainer == 'hf':
                    model = args.get('model') or hf_model
                    epochs = str(args.get('epochs') or hf_epochs)
                    batch = str(args.get('batch_size') or hf_batch)
                    maxlen = str(args.get('max_len') or hf_maxlen)
                    lr = str(args.get('lr') or hf_lr)
                    p = submit('local:///opt/app/dspy_agent/training/train_grpo_tool_hf.py', '--manifest', manifest, '--model', model, '--epochs', epochs, '--batch-size', batch, '--max-len', maxlen, '--lr', lr)
                    batch_jobs.append({'name': 'train_grpo_tool_hf', 'proc': p, 't0': time.time()})
                else:
                    epochs = str(args.get('epochs') or '1')
                    batch = str(args.get('batch_size') or '16')
                    p = submit('local:///opt/app/dspy_agent/training/train_grpo_tool.py', '--manifest', manifest, '--epochs', epochs, '--batch-size', batch)
                    batch_jobs.append({'name': 'train_grpo_tool', 'proc': p, 't0': time.time()})
                # Remove control file after submission
                os.remove(ctrl_path)
            except Exception as e:
                print('[spark-job] train control error:', e)
        # Harvest completed batch jobs and emit events
        for j in list(batch_jobs):
            rc = j['proc'].poll()
            if rc is not None:
                try:
                    ns = os.getenv('SPARK_NAMESPACE', os.getenv('K8S_NAMESPACE', 'local'))
                    log_spark_app('completed', j['name'], state='SUCCEEDED' if rc == 0 else 'FAILED', namespace=ns)
                    # If dataset export finished, emit dataset-ready event
                    if j['name'] == 'tool_batch_export' and rc == 0:
                        manifest = os.path.join(WAREHOUSE, 'datasets', 'grpo_tool_batches', 'manifest.json')
                        try:
                            import json as _json
                            paths = _json.loads(open(manifest).read()) if os.path.exists(manifest) else []
                            total = sum(int(x.get('rows', 0)) for x in (paths or []))
                            log_training_dataset('grpo_tool_batches', {'manifest': manifest, 'shards': len(paths or []), 'rows': total})
                        except Exception:
                            log_training_dataset('grpo_tool_batches', {'manifest': manifest, 'error': 'manifest_unavailable'})
                    if j['name'] == 'code_log_build_dataset' and rc == 0:
                        try:
                            ds_dir = os.path.join(WAREHOUSE, 'datasets', 'code_log_predict')
                            shards = 0
                            for root, dirs, files in os.walk(ds_dir):
                                for f in files:
                                    if f.endswith('.parquet'):
                                        shards += 1
                            log_training_dataset('code_log_predict', {'dataset_dir': ds_dir, 'shards': shards})
                        except Exception:
                            log_training_dataset('code_log_predict', {'dataset_dir': os.path.join(WAREHOUSE, 'datasets', 'code_log_predict'), 'error': 'unavailable'})
                except Exception:
                    pass
                batch_jobs.remove(j)
        time.sleep(10)


if __name__ == '__main__':
    main()
