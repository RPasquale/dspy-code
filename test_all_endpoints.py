#!/usr/bin/env python3
"""
Comprehensive endpoint testing script for RedDB API.
Tests every single endpoint systematically.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8083"

def test_endpoint(method, endpoint, data=None, expected_status=200):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=5)
        elif method == "DELETE":
            response = requests.delete(url, timeout=5)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        result = {
            "endpoint": endpoint,
            "method": method,
            "status": response.status_code,
            "success": response.status_code == expected_status,
            "response_size": len(response.text),
            "content_type": response.headers.get('content-type', 'unknown')
        }
        
        if response.status_code == 200:
            try:
                result["data"] = response.json()
            except:
                result["data"] = response.text[:200] + "..." if len(response.text) > 200 else response.text
        else:
            result["error"] = response.text[:200] + "..." if len(response.text) > 200 else response.text
            
        return result
        
    except Exception as e:
        return {
            "endpoint": endpoint,
            "method": method,
            "error": str(e),
            "success": False
        }

def test_all_endpoints():
    """Test all available endpoints"""
    print("🔍 Testing All RedDB API Endpoints")
    print("=" * 60)
    
    # Define all endpoints to test
    endpoints = [
        # Health and telemetry
        ("GET", "/health"),
        ("GET", "/metrics"),
        ("GET", "/api/status"),
        ("GET", "/api/logs"),
        ("GET", "/api/metrics"),
        ("GET", "/api/system/resources"),
        ("GET", "/api/containers"),
        ("GET", "/api/bus-metrics"),
        ("GET", "/api/stream-metrics"),
        ("GET", "/api/rl-metrics"),
        ("GET", "/api/learning-metrics"),
        ("GET", "/api/performance-history"),
        ("GET", "/api/kafka-topics"),
        ("GET", "/api/spark-workers"),
        ("GET", "/api/overview"),
        ("GET", "/api/overview/stream"),
        ("GET", "/api/overview/stream-diff"),
        ("GET", "/api/system-topology"),
        ("GET", "/api/profile"),
        ("GET", "/api/config"),
        ("GET", "/api/reddb/health"),

        # Control endpoints
        ("POST", "/api/command", {"command": "status"}),
        ("POST", "/api/chat", {"message": "hello"}),
        ("POST", "/api/restart"),

        # Signatures endpoints
        ("GET", "/api/signatures"),
        ("POST", "/api/signatures", {
            "name": "test_signature",
            "type": "test",
            "description": "Test signature for endpoint testing"
        }),
        ("GET", "/api/signatures/test_signature"),
        ("GET", "/api/signatures/test_signature/schema"),
        ("GET", "/api/signatures/test_signature/analytics"),
        ("PUT", "/api/signatures/test_signature", {
            "name": "test_signature",
            "type": "test_updated",
            "description": "Updated test signature"
        }),
        ("POST", "/api/signature/optimize", {
            "signature_name": "test_signature",
            "type": "performance"
        }),
        ("GET", "/api/signature/graph"),
        ("GET", "/api/signature/optimization-history?name=test_signature"),
        ("DELETE", "/api/signatures/test_signature"),

        # Verifiers endpoints
        ("GET", "/api/verifiers"),
        ("POST", "/api/verifiers", {
            "name": "test_verifier",
            "description": "Test verifier for endpoint testing"
        }),
        ("PUT", "/api/verifiers/test_verifier", {
            "name": "test_verifier",
            "description": "Updated test verifier"
        }),
        ("DELETE", "/api/verifiers/test_verifier"),

        # Action results
        ("POST", "/api/action/record-result", {
            "signature_name": "test_signature",
            "reward": 0.9,
            "verifier_scores": {"unit": 0.95}
        }),
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for i, endpoint_data in enumerate(endpoints, 1):
        if len(endpoint_data) == 2:
            method, endpoint = endpoint_data
            data = None
        else:
            method, endpoint, data = endpoint_data
        
        print(f"\n📋 Test {i:2d}: {method} {endpoint}")
        print("-" * 40)
        
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        
        if result.get("success", False):
            print(f"✅ PASS - Status: {result['status']}")
            passed += 1
        else:
            print(f"❌ FAIL - Status: {result.get('status', 'ERROR')}")
            if "error" in result:
                print(f"   Error: {result['error']}")
            failed += 1
        
        # Show response preview for successful requests
        if result.get("success") and "data" in result:
            data_preview = str(result["data"])[:100]
            if len(str(result["data"])) > 100:
                data_preview += "..."
            print(f"   Response: {data_preview}")
        
        time.sleep(0.1)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(results)*100):.1f}%")
    
    # Show failed tests
    if failed > 0:
        print(f"\n❌ FAILED TESTS:")
        for result in results:
            if not result.get("success", False):
                print(f"   {result['method']} {result['endpoint']} - {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    test_all_endpoints()
