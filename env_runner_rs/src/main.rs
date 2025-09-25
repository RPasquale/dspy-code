use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use env_runner::{prefetch_from, PrefetchQueue, Runner, WorkItem, WorkloadClass};

fn main() {
    let queue = Arc::new(PrefetchQueue::new(128));
    let processed = Arc::new(Mutex::new(Vec::new()));
    let runner = Runner { queue: queue.clone(), processed: processed.clone() };
    let _h = runner.start();

    // File-queue integration: read from logs/env_queue/pending and push into queue
    let base = std::env::var("ENV_QUEUE_DIR").unwrap_or_else(|_| "logs/env_queue".to_string());
    let pend = Path::new(&base).join("pending");
    let done = Path::new(&base).join("done");
    let _ = fs::create_dir_all(&pend);
    let _ = fs::create_dir_all(&done);

    // feeder thread to scan pending files
    let feeder_q = queue.clone();
    let feeder = thread::spawn(move || {
        loop {
            if let Ok(entries) = fs::read_dir(&pend) {
                for ent in entries.flatten() {
                    let path = ent.path();
                    if !path.is_file() { continue; }
                    // claim file by renaming to .lock
                    let lock_path = path.with_extension("lock");
                    if fs::rename(&path, &lock_path).is_err() { continue; }
                    if let Some(item) = read_item(&lock_path) {
                        feeder_q.push(item);
                        // move to done to acknowledge enqueue
                        let fname = lock_path.file_name().unwrap().to_string_lossy().replace(".lock", "");
                        let done_path = done.join(fname);
                        let _ = fs::rename(&lock_path, &done_path);
                    } else {
                        // failed parse, drop
                        let _ = fs::remove_file(&lock_path);
                    }
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
    });

    // Let it run a bit (demo mode)
    let _ = feeder.thread().id();
    thread::sleep(Duration::from_millis(100));
    println!("processed={}", processed.lock().unwrap().len());
}

fn read_item(path: &PathBuf) -> Option<WorkItem> {
    let mut f = fs::File::open(path).ok()?;
    let mut s = String::new();
    let _ = f.read_to_string(&mut s);
    parse_item(&s)
}

fn parse_item(s: &str) -> Option<WorkItem> {
    // very naive JSON parsing of {"ID":"..","Class":"..","Payload":".."}
    let id = extract_str(s, "ID").or_else(|| extract_str(s, "id"))?;
    let class_str = extract_str(s, "Class").or_else(|| extract_str(s, "class")).unwrap_or_else(|| "cpu_short".to_string());
    let class = match class_str.to_lowercase().as_str() {
        "gpu" => WorkloadClass::Gpu,
        "cpu_long" => WorkloadClass::CpuLong,
        _ => WorkloadClass::CpuShort,
    };
    let payload = extract_str(s, "Payload").or_else(|| extract_str(s, "payload")).unwrap_or_default();
    Some(WorkItem { id, class, payload })
}

fn extract_str(s: &str, key: &str) -> Option<String> {
    let k = format!("\"{}\"", key);
    let idx = s.find(&k)?;
    let rest = &s[idx + k.len()..];
    let colon = rest.find(':')?;
    let after = &rest[colon + 1..];
    let first_quote = after.find('"')?;
    let after_q = &after[first_quote + 1..];
    let end_q = after_q.find('"')?;
    Some(after_q[..end_q].to_string())
}
