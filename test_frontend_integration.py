#!/usr/bin/env python3
"""
Test frontend-backend integration by simulating frontend API calls.
"""

import requests
import json
import time
from datetime import datetime

def test_frontend_integration():
    """Test all endpoints that the frontend uses"""
    print("🔗 Testing Frontend-Backend Integration")
    print("=" * 60)
    
    base_url = "http://localhost:8083"
    
    # Test all the endpoints that the React frontend will call
    frontend_endpoints = [
        # Dashboard Overview
        ("GET", "/api/overview", "Dashboard Overview"),
        ("GET", "/api/status", "System Status"),
        ("GET", "/api/metrics", "System Metrics"),
        
        # Learning & Training
        ("GET", "/api/learning-metrics", "Learning Metrics"),
        ("GET", "/api/performance-history", "Performance History"),
        
        # Signatures & Verifiers
        ("GET", "/api/signatures", "Signatures List"),
        ("GET", "/api/verifiers", "Verifiers List"),
        
        # Infrastructure
        ("GET", "/api/kafka-topics", "Kafka Topics"),
        ("GET", "/api/spark-workers", "Spark Workers"),
        ("GET", "/api/system-topology", "System Topology"),
        ("GET", "/api/system/resources", "System Resources"),
        
        # Configuration
        ("GET", "/api/config", "System Config"),
        ("GET", "/api/profile", "User Profile"),
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for i, (method, endpoint, description) in enumerate(frontend_endpoints, 1):
        print(f"\n📋 Test {i:2d}: {description}")
        print(f"    {method} {endpoint}")
        print("-" * 40)
        
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ PASS - Status: {response.status_code}")
                    print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    passed += 1
                except:
                    print(f"✅ PASS - Status: {response.status_code}")
                    print(f"   Response: {response.text[:100]}...")
                    passed += 1
            else:
                print(f"❌ FAIL - Status: {response.status_code}")
                print(f"   Error: {response.text[:100]}...")
                failed += 1
                
        except Exception as e:
            print(f"❌ FAIL - Connection Error: {str(e)}")
            failed += 1
        
        time.sleep(0.1)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FRONTEND INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(frontend_endpoints)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(frontend_endpoints)*100):.1f}%")
    
    if passed == len(frontend_endpoints):
        print("\n🎉 ALL FRONTEND ENDPOINTS WORKING!")
        print("   The React frontend should now display data correctly.")
    else:
        print(f"\n⚠️  {failed} endpoints need attention.")
    
    return passed == len(frontend_endpoints)

if __name__ == "__main__":
    test_frontend_integration()
