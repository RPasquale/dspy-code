#!/usr/bin/env python3
"""
Comprehensive test suite for InferMesh optimization with Go/Rust/Slurm integration.
Tests the enhanced InferMesh performance with orchestrator coordination.
"""

import asyncio
import json
import time
import requests
import pytest
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferMeshOptimizationTester:
    """Test suite for InferMesh optimization components."""
    
    def __init__(self):
        self.base_url = "http://localhost:9000"
        self.orchestrator_url = "http://localhost:9097"
        self.rust_runner_url = "http://localhost:8080"
        self.test_results = {}
        
    def test_basic_infermesh_health(self) -> Dict[str, Any]:
        """Test basic InferMesh health and functionality."""
        results = {
            "test": "basic_infermesh_health",
            "passed": False,
            "metrics": {},
            "errors": []
        }
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                results["metrics"]["status"] = health_data.get("status")
                results["metrics"]["model_loaded"] = health_data.get("model_loaded", False)
                results["metrics"]["gpu_available"] = health_data.get("gpu_available", False)
                results["metrics"]["memory_usage_mb"] = health_data.get("memory_usage_mb", 0)
                results["passed"] = True
            else:
                results["errors"].append(f"Health check failed with status {response.status_code}")
                
        except Exception as e:
            results["errors"].append(f"Health check failed: {str(e)}")
        
        return results
    
    def test_embedding_performance(self, num_texts: int = 100) -> Dict[str, Any]:
        """Test embedding performance with various batch sizes."""
        results = {
            "test": "embedding_performance",
            "passed": False,
            "metrics": {},
            "errors": []
        }
        
        try:
            # Generate test texts
            test_texts = [f"This is test text number {i} for performance testing" for i in range(num_texts)]
            
            # Test single embedding
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/embed",
                json={"model": "BAAI/bge-small-en-v1.5", "inputs": test_texts[:1]},
                timeout=30
            )
            single_time = time.time() - start_time
            
            if response.status_code == 200:
                single_data = response.json()
                results["metrics"]["single_embedding_time"] = single_time
                results["metrics"]["single_embedding_vectors"] = len(single_data.get("vectors", []))
            else:
                results["errors"].append(f"Single embedding failed: {response.status_code}")
                return results
            
            # Test batch embedding
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/embed",
                json={"model": "BAAI/bge-small-en-v1.5", "inputs": test_texts},
                timeout=60
            )
            batch_time = time.time() - start_time
            
            if response.status_code == 200:
                batch_data = response.json()
                results["metrics"]["batch_embedding_time"] = batch_time
                results["metrics"]["batch_embedding_vectors"] = len(batch_data.get("vectors", []))
                results["metrics"]["throughput_per_second"] = num_texts / batch_time
                results["metrics"]["efficiency_ratio"] = batch_time / (single_time * num_texts)
                results["passed"] = True
            else:
                results["errors"].append(f"Batch embedding failed: {response.status_code}")
                
        except Exception as e:
            results["errors"].append(f"Performance test failed: {str(e)}")
        
        return results
    
    def test_go_orchestrator_integration(self) -> Dict[str, Any]:
        """Test Go orchestrator integration with InferMesh."""
        results = {
            "test": "go_orchestrator_integration",
            "passed": False,
            "metrics": {},
            "errors": []
        }
        
        try:
            # Test orchestrator health
            response = requests.get(f"{self.orchestrator_url}/metrics", timeout=5)
            if response.status_code == 200:
                metrics_data = response.json()
                results["metrics"]["orchestrator_healthy"] = True
                results["metrics"]["queue_depth"] = metrics_data.get("queue_depth", 0)
                results["metrics"]["gpu_wait_seconds"] = metrics_data.get("gpu_wait_seconds", 0)
                results["metrics"]["error_rate"] = metrics_data.get("error_rate", 0)
            else:
                results["errors"].append(f"Orchestrator health check failed: {response.status_code}")
                return results
            
            # Test concurrent requests through orchestrator
            test_texts = [f"Orchestrator test text {i}" for i in range(50)]
            
            start_time = time.time()
            response = requests.post(
                f"{self.orchestrator_url}/embed",
                json={"model": "BAAI/bge-small-en-v1.5", "inputs": test_texts},
                timeout=30
            )
            orchestrator_time = time.time() - start_time
            
            if response.status_code == 200:
                results["metrics"]["orchestrator_embedding_time"] = orchestrator_time
                results["metrics"]["orchestrator_throughput"] = len(test_texts) / orchestrator_time
                results["passed"] = True
            else:
                results["errors"].append(f"Orchestrator embedding failed: {response.status_code}")
                
        except Exception as e:
            results["errors"].append(f"Orchestrator integration test failed: {str(e)}")
        
        return results
    
    def test_rust_runner_integration(self) -> Dict[str, Any]:
        """Test Rust environment runner integration."""
        results = {
            "test": "rust_runner_integration",
            "passed": False,
            "metrics": {},
            "errors": []
        }
        
        try:
            # Test Rust runner health
            response = requests.get(f"{self.rust_runner_url}/health", timeout=5)
            if response.status_code == 200:
                results["metrics"]["rust_runner_healthy"] = True
            else:
                results["errors"].append(f"Rust runner health check failed: {response.status_code}")
                return results
            
            # Test high-performance I/O
            test_texts = [f"Rust runner test text {i}" for i in range(200)]
            
            start_time = time.time()
            response = requests.post(
                f"{self.rust_runner_url}/embed",
                json={"model": "BAAI/bge-small-en-v1.5", "inputs": test_texts},
                timeout=60
            )
            rust_time = time.time() - start_time
            
            if response.status_code == 200:
                results["metrics"]["rust_embedding_time"] = rust_time
                results["metrics"]["rust_throughput"] = len(test_texts) / rust_time
                results["passed"] = True
            else:
                results["errors"].append(f"Rust runner embedding failed: {response.status_code}")
                
        except Exception as e:
            results["errors"].append(f"Rust runner integration test failed: {str(e)}")
        
        return results
    
    def test_concurrent_load(self, num_concurrent: int = 10, texts_per_request: int = 20) -> Dict[str, Any]:
        """Test concurrent load handling."""
        results = {
            "test": "concurrent_load",
            "passed": False,
            "metrics": {},
            "errors": []
        }
        
        try:
            import concurrent.futures
            
            def make_request(request_id: int):
                test_texts = [f"Concurrent test {request_id}-{i}" for i in range(texts_per_request)]
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/embed",
                    json={"model": "BAAI/bge-small-en-v1.5", "inputs": test_texts},
                    timeout=30
                )
                
                request_time = time.time() - start_time
                return {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "time": request_time,
                    "success": response.status_code == 200
                }
            
            # Execute concurrent requests
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
                request_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            total_time = time.time() - start_time
            
            # Analyze results
            successful_requests = sum(1 for r in request_results if r["success"])
            failed_requests = num_concurrent - successful_requests
            avg_request_time = sum(r["time"] for r in request_results) / len(request_results)
            
            results["metrics"]["total_requests"] = num_concurrent
            results["metrics"]["successful_requests"] = successful_requests
            results["metrics"]["failed_requests"] = failed_requests
            results["metrics"]["total_time"] = total_time
            results["metrics"]["avg_request_time"] = avg_request_time
            results["metrics"]["throughput_per_second"] = (successful_requests * texts_per_request) / total_time
            results["metrics"]["success_rate"] = successful_requests / num_concurrent
            
            if successful_requests >= num_concurrent * 0.9:  # 90% success rate
                results["passed"] = True
            else:
                results["errors"].append(f"Low success rate: {successful_requests}/{num_concurrent}")
                
        except Exception as e:
            results["errors"].append(f"Concurrent load test failed: {str(e)}")
        
        return results
    
    def test_memory_efficiency(self, num_large_batches: int = 5) -> Dict[str, Any]:
        """Test memory efficiency with large batches."""
        results = {
            "test": "memory_efficiency",
            "passed": False,
            "metrics": {},
            "errors": []
        }
        
        try:
            # Create large batches
            large_texts = [f"Large batch text {i} for memory testing" * 10 for i in range(1000)]
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/embed",
                json={"model": "BAAI/bge-small-en-v1.5", "inputs": large_texts},
                timeout=120
            )
            batch_time = time.time() - start_time
            
            if response.status_code == 200:
                batch_data = response.json()
                results["metrics"]["large_batch_time"] = batch_time
                results["metrics"]["large_batch_size"] = len(large_texts)
                results["metrics"]["large_batch_vectors"] = len(batch_data.get("vectors", []))
                results["metrics"]["memory_efficiency"] = len(large_texts) / batch_time
                results["passed"] = True
            else:
                results["errors"].append(f"Large batch processing failed: {response.status_code}")
                
        except Exception as e:
            results["errors"].append(f"Memory efficiency test failed: {str(e)}")
        
        return results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        results = {
            "test": "error_handling",
            "passed": False,
            "metrics": {},
            "errors": []
        }
        
        try:
            # Test invalid model
            response = requests.post(
                f"{self.base_url}/embed",
                json={"model": "invalid-model", "inputs": ["test"]},
                timeout=10
            )
            results["metrics"]["invalid_model_status"] = response.status_code
            
            # Test empty inputs
            response = requests.post(
                f"{self.base_url}/embed",
                json={"model": "BAAI/bge-small-en-v1.5", "inputs": []},
                timeout=10
            )
            results["metrics"]["empty_inputs_status"] = response.status_code
            
            # Test malformed JSON
            try:
                response = requests.post(
                    f"{self.base_url}/embed",
                    data="invalid json",
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                results["metrics"]["malformed_json_status"] = response.status_code
            except Exception:
                results["metrics"]["malformed_json_status"] = "connection_error"
            
            # Test recovery with valid request
            response = requests.post(
                f"{self.base_url}/embed",
                json={"model": "BAAI/bge-small-en-v1.5", "inputs": ["recovery test"]},
                timeout=10
            )
            
            if response.status_code == 200:
                results["metrics"]["recovery_successful"] = True
                results["passed"] = True
            else:
                results["errors"].append("Recovery test failed")
                
        except Exception as e:
            results["errors"].append(f"Error handling test failed: {str(e)}")
        
        return results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all optimization tests."""
        logger.info("Starting comprehensive InferMesh optimization tests...")
        
        tests = [
            self.test_basic_infermesh_health,
            self.test_embedding_performance,
            self.test_go_orchestrator_integration,
            self.test_rust_runner_integration,
            self.test_concurrent_load,
            self.test_memory_efficiency,
            self.test_error_handling,
        ]
        
        results = {
            "total_tests": len(tests),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "summary": {}
        }
        
        for test_func in tests:
            logger.info(f"Running {test_func.__name__}...")
            test_result = test_func()
            results["test_results"].append(test_result)
            
            if test_result["passed"]:
                results["passed_tests"] += 1
            else:
                results["failed_tests"] += 1
        
        # Generate summary
        results["summary"] = {
            "total_tests": results["total_tests"],
            "passed_tests": results["passed_tests"],
            "failed_tests": results["failed_tests"],
            "success_rate": results["passed_tests"] / results["total_tests"] if results["total_tests"] > 0 else 0
        }
        
        logger.info(f"Test results: {results['passed_tests']}/{results['total_tests']} tests passed")
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("# InferMesh Optimization Test Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        summary = results["summary"]
        report.append("## Summary")
        report.append(f"- Total Tests: {summary['total_tests']}")
        report.append(f"- Passed Tests: {summary['passed_tests']}")
        report.append(f"- Failed Tests: {summary['failed_tests']}")
        report.append(f"- Success Rate: {summary['success_rate']:.2%}")
        report.append("")
        
        # Test details
        report.append("## Test Details")
        for test_result in results["test_results"]:
            report.append(f"### {test_result['test']}")
            
            if test_result["passed"]:
                report.append("✅ **PASSED**")
            else:
                report.append("❌ **FAILED**")
                for error in test_result["errors"]:
                    report.append(f"  - Error: {error}")
            
            # Metrics
            if test_result["metrics"]:
                report.append("**Metrics:**")
                for key, value in test_result["metrics"].items():
                    report.append(f"  - {key}: {value}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    """Main test execution."""
    tester = InferMeshOptimizationTester()
    results = tester.run_comprehensive_tests()
    
    # Generate and save report
    report = tester.generate_report(results)
    
    # Save report to file
    report_path = Path("infermesh_optimization_test_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Test report saved to: {report_path}")
    print(f"Summary: {results['summary']}")
    
    # Return appropriate exit code
    if results["summary"]["failed_tests"] > 0:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()
