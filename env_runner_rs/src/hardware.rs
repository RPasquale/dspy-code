use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Command;
use std::str::FromStr;
use sysinfo::System;
use which::which;

/// Summary of the host hardware that the environment runner can leverage.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct HardwareSnapshot {
    pub hostname: String,
    pub detected_at: String,
    pub cpu: CpuSummary,
    pub memory: MemorySummary,
    pub accelerators: Vec<AcceleratorInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CpuSummary {
    pub brand: String,
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub min_frequency_mhz: u64,
    pub max_frequency_mhz: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MemorySummary {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AcceleratorInfo {
    pub vendor: String,
    pub model: String,
    pub memory_mb: u64,
    pub count: u32,
    pub compute_capability: Option<String>,
}

/// Detects the local hardware configuration.
pub fn detect() -> HardwareSnapshot {
    let mut sys = System::new_all();
    sys.refresh_all();

    let host = System::host_name().unwrap_or_else(|| "unknown".to_string());
    let logical = sys.cpus().len();
    let physical = sys.physical_core_count().unwrap_or(logical);
    let brand = sys
        .cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let min_freq = sys
        .cpus()
        .iter()
        .map(|cpu| cpu.frequency())
        .min()
        .unwrap_or_default();
    let max_freq = sys
        .cpus()
        .iter()
        .map(|cpu| cpu.frequency())
        .max()
        .unwrap_or_default();

    let total_bytes = sys.total_memory() * 1024;
    let available_bytes = sys.available_memory() * 1024;

    let mut accelerators = Vec::new();
    accelerators.extend(detect_nvidia_gpus());
    accelerators.extend(detect_rocm_gpus());
    accelerators.extend(detect_apple_metal());

    HardwareSnapshot {
        hostname: host,
        detected_at: Utc::now().to_rfc3339(),
        cpu: CpuSummary {
            brand,
            logical_cores: logical,
            physical_cores: physical,
            min_frequency_mhz: min_freq,
            max_frequency_mhz: max_freq,
        },
        memory: MemorySummary {
            total_bytes,
            available_bytes,
        },
        accelerators,
    }
}

/// Returns a conservative default inflight value derived from the detected hardware.
pub fn recommended_inflight(snapshot: &HardwareSnapshot) -> usize {
    if !snapshot.accelerators.is_empty() {
        let gpu_total: u32 = snapshot.accelerators.iter().map(|acc| acc.count).sum();
        return usize::max(4, gpu_total as usize * 4);
    }
    let logical = snapshot.cpu.logical_cores.max(1);
    usize::max(2, logical / 2)
}

fn detect_nvidia_gpus() -> Vec<AcceleratorInfo> {
    if let Ok(nvidia_smi) = which("nvidia-smi") {
        let query = [
            "--query-gpu=name,memory.total,compute_cap,gpu_bus_id",
            "--format=csv,noheader,nounits",
        ];
        if let Ok(output) = Command::new(nvidia_smi).args(query).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return stdout.lines().filter_map(parse_nvidia_row).collect();
            }
        }
    }
    Vec::new()
}

fn parse_nvidia_row(row: &str) -> Option<AcceleratorInfo> {
    let parts: Vec<_> = row.split(',').map(|token| token.trim()).collect();
    if parts.is_empty() {
        return None;
    }
    let model = parts.first()?.to_string();
    let memory_mb = parts
        .get(1)
        .and_then(|val| u64::from_str(val).ok())
        .unwrap_or_default();
    let compute_capability = parts.get(2).map(|s| s.to_string());
    Some(AcceleratorInfo {
        vendor: "nvidia".to_string(),
        model,
        memory_mb,
        count: 1,
        compute_capability,
    })
}

#[allow(clippy::unnecessary_filter_map)]
fn detect_rocm_gpus() -> Vec<AcceleratorInfo> {
    if let Ok(rocm_smi) = which("rocm-smi") {
        if let Ok(output) = Command::new(rocm_smi)
            .args(["--showproductname", "--json"])
            .output()
        {
            if output.status.success() {
                if let Ok(json) = serde_json::from_slice::<Value>(&output.stdout) {
                    if let Some(map) = json.as_object() {
                        return map
                            .iter()
                            .filter_map(|(_, value)| {
                                let product = value
                                    .get("Card Series")
                                    .or_else(|| value.get("card_series"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("AMD GPU");
                                let mem = value
                                    .get("VRAM Total Memory")
                                    .or_else(|| value.get("vram_total"))
                                    .and_then(|v| v.as_f64())
                                    .map(|gb| (gb * 1024.0) as u64)
                                    .unwrap_or_default();
                                Some(AcceleratorInfo {
                                    vendor: "amd".to_string(),
                                    model: product.to_string(),
                                    memory_mb: mem,
                                    count: 1,
                                    compute_capability: None,
                                })
                            })
                            .collect();
                    }
                }
            }
        }
    }
    Vec::new()
}

// Best-effort detection for Apple Metal capable GPUs when running on macOS.
#[cfg(target_os = "macos")]
#[allow(clippy::unnecessary_filter_map)]
fn detect_apple_metal() -> Vec<AcceleratorInfo> {
    if let Ok(output) = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
    {
        if output.status.success() {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                if let Some(array) = json
                    .get("SPDisplaysDataType")
                    .and_then(|value| value.as_array())
                {
                    return array
                        .iter()
                        .filter_map(|entry| {
                            let model = entry
                                .get("_name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Apple GPU");
                            let memory = entry
                                .get("spdisplays_vram")
                                .and_then(|v| v.as_str())
                                .and_then(parse_memory_to_mb)
                                .unwrap_or_default();
                            Some(AcceleratorInfo {
                                vendor: "apple".to_string(),
                                model: model.to_string(),
                                memory_mb: memory,
                                count: 1,
                                compute_capability: Some("metal".to_string()),
                            })
                        })
                        .collect();
                }
            }
        }
    }
    Vec::new()
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_filter_map)]
fn detect_apple_metal() -> Vec<AcceleratorInfo> {
    Vec::new()
}

#[cfg(target_os = "macos")]
fn parse_memory_to_mb(input: &str) -> Option<u64> {
    let lower = input.to_lowercase();
    if let Some(value) = lower.strip_suffix(" gb") {
        return value
            .trim()
            .parse::<f64>()
            .ok()
            .map(|gb| (gb * 1024.0) as u64);
    }
    if let Some(value) = lower.strip_suffix(" mb") {
        return value.trim().parse::<f64>().ok().map(|mb| mb as u64);
    }
    None
}

#[cfg(not(target_os = "macos"))]
fn parse_memory_to_mb(_input: &str) -> Option<u64> {
    None
}
