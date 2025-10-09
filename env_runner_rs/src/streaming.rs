use crate::hardware::HardwareSnapshot;
use crate::pb;
use serde_json::Value;

pub type RunnerTaskAssignment = pb::runner::TaskAssignment;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkflowMeta {
    pub workflow_id: Option<String>,
    pub workflow_run_id: Option<String>,
    pub tenant: Option<String>,
}

pub fn extract_workflow_meta(assignment: &RunnerTaskAssignment) -> WorkflowMeta {
    match serde_json::from_slice::<Value>(&assignment.payload) {
        Ok(value) => extract_meta_from_value(&value),
        Err(_) => WorkflowMeta {
            tenant: fallback_tenant_from_assignment(assignment),
            ..WorkflowMeta::default()
        },
    }
}

/// Builds dynamic Kafka header entries derived from workflow context and hardware hints.
pub fn collect_assignment_headers(
    assignment: &RunnerTaskAssignment,
    hardware: Option<&HardwareSnapshot>,
) -> Vec<(&'static str, Vec<u8>)> {
    let meta = extract_workflow_meta(assignment);
    let mut extras: Vec<(&'static str, Vec<u8>)> = Vec::new();
    if let Some(snapshot) = hardware {
        let gpu_total: u32 = snapshot.accelerators.iter().map(|acc| acc.count).sum();
        if gpu_total > 0 {
            extras.push(("runner_gpu_total", gpu_total.to_string().into_bytes()));
        }
        if let Ok(serialized) = serde_json::to_string(snapshot) {
            extras.push(("runner_hardware", serialized.into_bytes()));
        }
    }
    if let Some(wf_id) = meta.workflow_id.as_ref() {
        extras.push(("workflow_id", wf_id.as_bytes().to_vec()));
    }
    if let Some(run_id) = meta.workflow_run_id.as_ref() {
        extras.push(("workflow_run_id", run_id.as_bytes().to_vec()));
    }
    if let Some(tenant) = meta.tenant.as_ref() {
        extras.push(("workflow_tenant", tenant.as_bytes().to_vec()));
    }
    extras
}

/// Derives a skill routing hint based on assignment metadata or workflow context.
pub fn derive_skill_from_assignment(assignment: &RunnerTaskAssignment) -> Option<String> {
    if let Some(tenant) = fallback_tenant_from_assignment(assignment) {
        if !tenant.is_empty() {
            return Some(tenant.to_lowercase());
        }
    }
    if let Some(topic) = assignment.topic.strip_prefix("workflow::") {
        return Some(format!("workflow::{topic}"));
    }
    let meta = extract_workflow_meta(assignment);
    if let Some(wf_id) = meta.workflow_id {
        if !wf_id.is_empty() {
            return Some(format!("workflow::{}", wf_id.to_lowercase()));
        }
    }
    None
}

fn fallback_tenant_from_assignment(assignment: &RunnerTaskAssignment) -> Option<String> {
    let tenant = assignment.tenant.trim();
    if !tenant.is_empty() {
        return Some(tenant.to_lowercase());
    }
    let topic = assignment.topic.trim();
    if !topic.is_empty() {
        return Some(topic.to_lowercase());
    }
    None
}

fn extract_meta_from_value(value: &Value) -> WorkflowMeta {
    let mut meta = WorkflowMeta::default();
    if let Some(wf_id) = value
        .get("workflow_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        meta.workflow_id = Some(wf_id.to_string());
    }
    if let Some(run_id) = value
        .get("workflow_run_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        meta.workflow_run_id = Some(run_id.to_string());
    }
    if let Some(tenant) = value
        .get("tenant")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        meta.tenant = Some(tenant.to_lowercase());
    }
    if let Some(ctx) = value.get("workflow_context") {
        if meta.workflow_id.is_none() {
            if let Some(ctx_id) = ctx
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                meta.workflow_id = Some(ctx_id.to_string());
            }
        }
        if meta.tenant.is_none() {
            if let Some(ctx_tenant) = ctx
                .get("tenant")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                meta.tenant = Some(ctx_tenant.to_lowercase());
            }
        }
    }
    if meta.tenant.is_none() {
        meta.tenant = fallback_tenant_from_payload(value);
    }
    meta
}

fn fallback_tenant_from_payload(value: &Value) -> Option<String> {
    value
        .get("tenant")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{AcceleratorInfo, HardwareSnapshot};
    use prost::bytes::Bytes;
    use serde_json::json;

    fn assignment_with_payload(payload: Value) -> RunnerTaskAssignment {
        RunnerTaskAssignment {
            task_id: "task-1".to_string(),
            tenant: String::new(),
            topic: String::new(),
            payload: Bytes::from(serde_json::to_vec(&payload).unwrap()),
            offset: 0,
            partition: String::new(),
        }
    }

    #[test]
    fn builds_headers_from_workflow_and_hardware() {
        let snapshot = HardwareSnapshot {
            accelerators: vec![AcceleratorInfo {
                vendor: "nvidia".to_string(),
                model: "A100".to_string(),
                memory_mb: 40960,
                count: 2,
                compute_capability: Some("8.0".to_string()),
            }],
            ..HardwareSnapshot::default()
        };
        let payload = json!({
            "workflow_id": "wf-123",
            "workflow_run_id": "run-1",
            "workflow_context": {"tenant": "demo"}
        });
        let assignment = assignment_with_payload(payload);
        let headers = collect_assignment_headers(&assignment, Some(&snapshot));
        let mut map = std::collections::HashMap::new();
        for (key, value) in headers {
            map.insert(key, String::from_utf8(value).unwrap());
        }
        assert_eq!(map.get("runner_gpu_total"), Some(&"2".to_string()));
        assert_eq!(map.get("workflow_id"), Some(&"wf-123".to_string()));
        assert_eq!(map.get("workflow_run_id"), Some(&"run-1".to_string()));
        assert_eq!(map.get("workflow_tenant"), Some(&"demo".to_string()));
    }

    #[test]
    fn derives_skill_prefers_tenant_then_workflow() {
        let payload = json!({
            "workflow_context": {"id": "WF-01", "tenant": "ACME"}
        });
        let mut assignment = assignment_with_payload(payload);
        assignment.tenant = "PromptOps".into();
        let skill = derive_skill_from_assignment(&assignment).expect("skill");
        assert_eq!(skill, "promptops");

        let assignment = assignment_with_payload(json!({
            "workflow_context": {"id": "WF-01"}
        }));
        let skill = derive_skill_from_assignment(&assignment).expect("skill");
        assert_eq!(skill, "workflow::wf-01");
    }

    #[test]
    fn extract_meta_reads_nested_fields() {
        let payload = json!({
            "workflow_context": {"id": "wf-55", "tenant": "TEAM"},
            "workflow_run_id": "run-9"
        });
        let assignment = assignment_with_payload(payload);
        let meta = extract_workflow_meta(&assignment);
        assert_eq!(meta.workflow_id.as_deref(), Some("wf-55"));
        assert_eq!(meta.workflow_run_id.as_deref(), Some("run-9"));
        assert_eq!(meta.tenant.as_deref(), Some("team"));
    }
}
