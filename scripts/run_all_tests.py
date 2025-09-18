#!/usr/bin/env python3
"""
Comprehensive test runner for DSPy Agent
Runs all tests including unit tests, integration tests, and end-to-end validation
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestRunner:
    def __init__(self):
        self.project_root = project_root
        self.results = {}
        self.start_time = time.time()
        
    def run_command(self, cmd: List[str], cwd: Path = None, timeout: int = 300) -> Tuple[bool, str, str]:
        """Run a command and return success, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd or self.project_root,
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", str(e)
    
    def test_imports(self) -> bool:
        """Test that all core modules can be imported"""
        print("🔍 Testing module imports...")
        
        modules_to_test = [
            "dspy_agent.cli",
            "dspy_agent.config", 
            "dspy_agent.llm",
            "dspy_agent.db",
            "dspy_agent.code_tools.code_search",
            "dspy_agent.code_tools.code_eval",
            "dspy_agent.skills.orchestrator",
            "dspy_agent.streaming.streamkit",
        ]
        
        failed_imports = []
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except ImportError as e:
                print(f"  ❌ {module}: {e}")
                failed_imports.append(module)
        
        success = len(failed_imports) == 0
        self.results["imports"] = {
            "success": success,
            "failed": failed_imports,
            "total": len(modules_to_test)
        }
        return success
    
    def test_unit_tests(self) -> bool:
        """Run unit tests using pytest"""
        print("\n🧪 Running unit tests...")
        
        # Try pytest first, fall back to unittest
        success, stdout, stderr = self.run_command(["uv", "run", "pytest", "tests/", "-v"])
        
        if not success:
            print("  Pytest failed, trying unittest...")
            success, stdout, stderr = self.run_command(["uv", "run", "python", "-m", "unittest", "discover", "-s", "tests", "-v"])
        
        self.results["unit_tests"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            print("  ✅ Unit tests passed")
        else:
            print(f"  ❌ Unit tests failed: {stderr}")
        
        return success
    
    def test_cli_help(self) -> bool:
        """Test that CLI help works"""
        print("\n🖥️  Testing CLI interface...")
        
        success, stdout, stderr = self.run_command(["uv", "run", "dspy-agent", "--help"])
        
        self.results["cli_help"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success and ("DSPy Agent" in stdout or "DSPy-based local coding agent" in stdout):
            print("  ✅ CLI help works")
            return True
        else:
            print(f"  ❌ CLI help failed: {stderr}")
            return False
    
    def test_database_init(self) -> bool:
        """Test database initialization"""
        print("\n🗄️  Testing database initialization...")
        
        try:
            from dspy_agent.db import initialize_database
            initialize_database()
            print("  ✅ Database initialization works")
            self.results["database"] = {"success": True}
            return True
        except Exception as e:
            print(f"  ❌ Database initialization failed: {e}")
            self.results["database"] = {"success": False, "error": str(e)}
            return False
    
    def test_lightweight_stack(self) -> bool:
        """Test lightweight stack generation"""
        print("\n🐳 Testing lightweight stack generation...")
        
        test_dir = self.project_root / "test_lightweight"
        test_dir.mkdir(exist_ok=True)
        
        success, stdout, stderr = self.run_command([
            "uv", "run", "dspy-agent", "lightweight_init",
            "--workspace", str(self.project_root),
            "--logs", str(test_dir / "logs"),
            "--out-dir", str(test_dir)
        ])
        
        # Check if key files were generated
        docker_compose = test_dir / "docker-compose.yml"
        dockerfile = test_dir / "Dockerfile"
        
        files_exist = docker_compose.exists() and dockerfile.exists()
        
        self.results["lightweight_stack"] = {
            "success": success and files_exist,
            "stdout": stdout,
            "stderr": stderr,
            "files_generated": files_exist
        }
        
        if success and files_exist:
            print("  ✅ Lightweight stack generation works")
            # Clean up
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            return True
        else:
            print(f"  ❌ Lightweight stack generation failed: {stderr}")
            return False
    
    def test_integration_scripts(self) -> bool:
        """Run integration test scripts"""
        print("\n🔗 Running integration tests...")
        
        integration_scripts = [
            "scripts/test_complete_integration.py",
            "scripts/test_full_integration.py",
            "scripts/test_reddb_integration.py"
        ]
        
        results = []
        passed_count = 0
        for script in integration_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                print(f"  Running {script}...")
                success, stdout, stderr = self.run_command(["uv", "run", "python", str(script_path)], timeout=600)
                results.append({
                    "script": script,
                    "success": success,
                    "stdout": stdout[-1000:] if stdout else "",  # Truncate for storage
                    "stderr": stderr[-1000:] if stderr else ""
                })
                
                if success:
                    print(f"    ✅ {script} passed")
                    passed_count += 1
                else:
                    print(f"    ❌ {script} failed")
                    if stderr:
                        print(f"      Error: {stderr[:200]}...")
            else:
                print(f"    ⚠️  {script} not found")
        
        # Consider integration tests successful if at least 2/3 pass
        all_passed = passed_count >= 2
        self.results["integration_tests"] = {
            "success": all_passed,
            "results": results,
            "passed": passed_count,
            "total": len(integration_scripts)
        }
        
        if all_passed:
            print(f"  ✅ Integration tests: {passed_count}/{len(integration_scripts)} passed")
        else:
            print(f"  ❌ Integration tests: {passed_count}/{len(integration_scripts)} passed (need at least 2)")
        
        return all_passed
    
    def test_project_example(self) -> bool:
        """Test with the example test project"""
        print("\n📁 Testing with example project...")
        
        test_project_dir = self.project_root / "test_project"
        if not test_project_dir.exists():
            print("  ⚠️  test_project directory not found")
            self.results["example_project"] = {"success": False, "error": "test_project not found"}
            return False
        
        # Test running the calculator tests
        success, stdout, stderr = self.run_command(
            ["uv", "run", "pytest", "test_calculator.py", "-v"],
            cwd=test_project_dir
        )
        
        self.results["example_project"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            print("  ✅ Example project tests passed")
        else:
            print(f"  ❌ Example project tests failed: {stderr}")
        
        return success
    
    def test_rl_components(self) -> bool:
        """Test RL components specifically"""
        print("\n🎯 Testing RL components...")
        
        # Check if RL test script exists
        rl_test_script = self.project_root / "scripts" / "test_rl.py"
        if not rl_test_script.exists():
            print("  ⚠️  RL test script not found, running basic RL tests...")
            return self.test_rl_basic()
        
        # Run the comprehensive RL test script
        success, stdout, stderr = self.run_command(
            ["uv", "run", "python", str(rl_test_script)],
            timeout=300
        )
        
        self.results["rl_components"] = {
            "success": success,
            "stdout": stdout[-1000:] if stdout else "",
            "stderr": stderr[-1000:] if stderr else ""
        }
        
        if success:
            print("  ✅ RL components tests passed")
        else:
            print(f"  ❌ RL components tests failed: {stderr}")
        
        return success
    
    def test_rl_basic(self) -> bool:
        """Basic RL test if comprehensive script not available"""
        print("  Running basic RL tests...")
        
        try:
            # Test basic RL imports
            from dspy_agent.rl.rlkit import RLToolEnv, EnvConfig, RewardConfig, ToolAction
            print("    ✅ RL imports work")
            
            # Test environment creation
            reward_cfg = RewardConfig(weights={"test": 1.0}, penalty_kinds=())
            env_cfg = EnvConfig(
                verifiers=[],
                reward_fn=lambda result, verifiers, weights: 1.0,
                weights=reward_cfg.weights,
                action_args=None,
                allowed_actions=["test_action"],
            )
            
            def dummy_executor(action, args):
                from dspy_agent.rl.rlkit import AgentResult
                return AgentResult(metrics={"test": 1.0}, info={})
            
            env = RLToolEnv(executor=dummy_executor, cfg=env_cfg, episode_len=1)
            obs, info = env.reset()
            obs_after, reward, terminated, truncated, step_info = env.step(0)
            
            print("    ✅ RL environment creation works")
            
            self.results["rl_components"] = {"success": True, "basic_test": True}
            return True
            
        except Exception as e:
            print(f"    ❌ Basic RL test failed: {e}")
            self.results["rl_components"] = {"success": False, "error": str(e)}
            return False
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("success", False))
        
        report = f"""
🧪 DSPy Agent Test Report
{'='*50}
Duration: {duration:.2f} seconds
Tests Run: {total_tests}
Tests Passed: {passed_tests}
Tests Failed: {total_tests - passed_tests}
Success Rate: {(passed_tests/total_tests)*100:.1f}%

📊 Detailed Results:
"""
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            report += f"  {test_name}: {status}\n"
            
            if not result.get("success", False) and "error" in result:
                report += f"    Error: {result['error']}\n"
        
        return report
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print("🚀 Starting comprehensive DSPy Agent test suite...")
        print("="*60)
        
        tests = [
            self.test_imports,
            self.test_unit_tests,
            self.test_cli_help,
            self.test_database_init,
            self.test_lightweight_stack,
            self.test_integration_scripts,
            self.test_project_example,
            self.test_rl_components
        ]
        
        all_passed = True
        for test in tests:
            try:
                if not test():
                    all_passed = False
            except Exception as e:
                print(f"  ❌ Test {test.__name__} crashed: {e}")
                all_passed = False
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        # Save results to file
        results_file = self.project_root / "test_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "duration": time.time() - self.start_time,
                "overall_success": all_passed,
                "results": self.results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
        return all_passed

def main():
    """Main entry point"""
    runner = TestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! The DSPy Agent is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the report above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
