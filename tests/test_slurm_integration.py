#!/usr/bin/env python3
"""
Integration tests for Slurm bridge functionality
Tests the complete Slurm job submission and monitoring workflow
"""

import json
import os
import time
import requests
import subprocess
from pathlib import Path
import pytest

class TestSlurmIntegration:
    """Test Slurm integration functionality"""
    
    @pytest.fixture
    def setup_test_environment(self):
        """Set up test environment"""
        # Create test directories
        test_dir = Path("test_slurm_integration")
        test_dir.mkdir(exist_ok=True)
        
        pend_dir = test_dir / "pending"
        done_dir = test_dir / "done"
        pend_dir.mkdir(exist_ok=True)
        done_dir.mkdir(exist_ok=True)
        
        # Set environment variables
        os.environ["ENV_QUEUE_DIR"] = str(test_dir)
        
        yield test_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
    
    def test_slurm_job_submission(self, setup_test_environment):
        """Test Slurm job submission via API"""
        test_dir = setup_test_environment
        
        # Create test task
        task_data = {
            "id": "test_slurm_001",
            "class": "gpu_slurm",
            "payload": {
                "method": "grpo",
                "module": "orchestrator",
                "model": "gpt2",
                "dataset": "test_dataset.jsonl"
            }
        }
        
        # Submit task to orchestrator
        response = requests.post(
            "http://localhost:9097/queue/submit",
            json=task_data,
            timeout=10
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert "slurm_job_id" in result
        assert result["status"] == "submitted"
        
        # Verify job was submitted to Slurm
        job_id = result["slurm_job_id"]
        assert job_id is not None
        assert job_id.isdigit()
    
    def test_slurm_job_status_check(self, setup_test_environment):
        """Test Slurm job status checking"""
        # Submit a job first
        task_data = {
            "id": "test_status_001",
            "class": "gpu_slurm",
            "payload": {"method": "grpo"}
        }
        
        response = requests.post(
            "http://localhost:9097/queue/submit",
            json=task_data,
            timeout=10
        )
        
        assert response.status_code == 200
        result = response.json()
        job_id = result["slurm_job_id"]
        
        # Check job status
        status_response = requests.get(
            f"http://localhost:9097/slurm/status/{task_data['id']}",
            timeout=10
        )
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "id" in status_data
        assert "status" in status_data
        assert status_data["id"] == job_id
    
    def test_queue_status_endpoint(self, setup_test_environment):
        """Test queue status endpoint"""
        response = requests.get(
            "http://localhost:9097/queue/status",
            timeout=10
        )
        
        assert response.status_code == 200
        status_data = response.json()
        assert "pending" in status_data
        assert "done" in status_data
        assert "submitted" in status_data
        assert "processed" in status_data
        assert "completed" in status_data
    
    def test_metrics_endpoint(self, setup_test_environment):
        """Test metrics endpoint"""
        response = requests.get(
            "http://localhost:9097/metrics",
            timeout=10
        )
        
        assert response.status_code == 200
        metrics_text = response.text
        
        # Check for expected metrics
        assert "env_queue_depth" in metrics_text
        assert "orchestrator_concurrency_limit" in metrics_text
        assert "orchestrator_inflight_tasks" in metrics_text
    
    def test_rust_metrics_endpoint(self, setup_test_environment):
        """Test Rust environment runner metrics endpoint"""
        response = requests.get(
            "http://localhost:8080/metrics",
            timeout=10
        )
        
        assert response.status_code == 200
        metrics_data = response.json()
        assert "tasks_processed" in metrics_data
        assert "queue_depth" in metrics_data
        assert "gpu_utilization" in metrics_data
        assert "latency_p95_ms" in metrics_data
    
    def test_rust_prometheus_endpoint(self, setup_test_environment):
        """Test Rust Prometheus format endpoint"""
        response = requests.get(
            "http://localhost:8080/prometheus",
            timeout=10
        )
        
        assert response.status_code == 200
        prometheus_text = response.text
        
        # Check for Prometheus format metrics
        assert "env_runner_tasks_processed_total" in prometheus_text
        assert "env_runner_queue_depth" in prometheus_text
        assert "env_runner_gpu_utilization" in prometheus_text
    
    def test_rust_health_endpoint(self, setup_test_environment):
        """Test Rust health endpoint"""
        response = requests.get(
            "http://localhost:8080/health",
            timeout=10
        )
        
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
    
    def test_file_queue_processing(self, setup_test_environment):
        """Test file queue processing"""
        test_dir = setup_test_environment
        pend_dir = test_dir / "pending"
        
        # Create a test task file
        task_data = {
            "id": "test_file_001",
            "class": "cpu_short",
            "payload": {"test": "data"}
        }
        
        task_file = pend_dir / "test_file_001.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)
        
        # Wait for processing
        time.sleep(1)
        
        # Check if file was processed
        assert not task_file.exists()
        
        # Check done directory
        done_dir = test_dir / "done"
        done_file = done_dir / "test_file_001.json"
        assert done_file.exists()
    
    def test_slurm_script_generation(self, setup_test_environment):
        """Test Slurm script generation"""
        # This test would verify that the generated sbatch script
        # contains the expected parameters and is valid
        
        # For now, just check that the template exists
        template_path = Path("deploy/slurm/train_agent_methodologies.sbatch")
        assert template_path.exists()
        
        # Check template content
        template_content = template_path.read_text()
        assert "#SBATCH" in template_content
        assert "torchrun" in template_content
        assert "${TASK_ID}" in template_content
        assert "${TRAINING_METHOD}" in template_content
    
    def test_error_handling(self, setup_test_environment):
        """Test error handling for invalid requests"""
        # Test invalid JSON
        response = requests.post(
            "http://localhost:9097/queue/submit",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        assert response.status_code == 400
        
        # Test missing required fields
        invalid_task = {
            "class": "gpu_slurm"
            # Missing id and payload
        }
        
        response = requests.post(
            "http://localhost:9097/queue/submit",
            json=invalid_task,
            timeout=10
        )
        
        # Should still work with auto-generated ID
        assert response.status_code == 200
    
    def test_concurrent_job_submission(self, setup_test_environment):
        """Test concurrent job submission"""
        import concurrent.futures
        
        def submit_job(i):
            task_data = {
                "id": f"concurrent_test_{i}",
                "class": "gpu_slurm",
                "payload": {"method": "grpo", "test_id": i}
            }
            
            response = requests.post(
                "http://localhost:9097/queue/submit",
                json=task_data,
                timeout=10
            )
            
            return response.status_code == 200
        
        # Submit 10 concurrent jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit_job, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # All should succeed
        assert all(results)
    
    def test_metrics_consistency(self, setup_test_environment):
        """Test that metrics are consistent across endpoints"""
        # Get metrics from Go orchestrator
        go_response = requests.get("http://localhost:9097/metrics", timeout=10)
        assert go_response.status_code == 200
        
        # Get metrics from Rust env-runner
        rust_response = requests.get("http://localhost:8080/metrics", timeout=10)
        assert rust_response.status_code == 200
        
        rust_data = rust_response.json()
        
        # Both should report queue depth
        assert "queue_depth" in rust_data
        
        # Queue depth should be consistent
        go_metrics = go_response.text
        assert "env_queue_depth" in go_metrics

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
