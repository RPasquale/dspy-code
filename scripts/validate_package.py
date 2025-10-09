#!/usr/bin/env python3
"""
Comprehensive Package Validation Script

This script runs a complete validation suite to ensure the package is ready
for publishing. It includes all tests, quality checks, and build validation.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


class PackageValidator:
    """Comprehensive package validation system"""
    
    def __init__(self, workspace: Path = None):
        self.workspace = workspace or Path.cwd()
        self.results: Dict[str, Any] = {}
        self.start_time = time.time()
        self.python_cmd = os.getenv("PYTHON_BIN") or shutil.which("python3") or sys.executable
        if not self.python_cmd:
            self.python_cmd = "python3"
        self.env = os.environ.copy()
        cache_dir = self.workspace / ".uv_cache"
        cache_dir.mkdir(exist_ok=True)
        self.env.setdefault("UV_CACHE_DIR", str(cache_dir))
        self.env.setdefault("UV_LINK_MODE", "copy")
        self.env.setdefault("UV_PYTHON_BIN", self.python_cmd)

    def run_command(self, cmd: Union[str, List[str]], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result"""
        if isinstance(cmd, list):
            printable = " ".join(cmd)
        else:
            printable = cmd
        print(f"  🔧 Running: {printable}")
        try:
            result = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                capture_output=True, 
                text=True, 
                cwd=self.workspace,
                check=check,
                env=self.env,
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Command failed: {e}")
            return e
    
    def validate_package_structure(self) -> bool:
        """Validate package structure"""
        print("📋 Validating package structure...")
        
        required_files = [
            "pyproject.toml",
            "README.md",
            "dspy_agent/__init__.py",
            "dspy_agent/cli.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.workspace / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"  ❌ Missing required files: {missing_files}")
            self.results["package_structure"] = {"status": "failed", "missing": missing_files}
            return False
        
        print("  ✅ Package structure is valid")
        self.results["package_structure"] = {"status": "passed"}
        return True
    
    def validate_dependencies(self) -> bool:
        """Validate dependencies"""
        print("🔍 Validating dependencies...")
        
        # Check pyproject.toml syntax
        result = self.run_command([
            "uv",
            "run",
            self.python_cmd,
            "-c",
            "import tomllib; tomllib.load(open('pyproject.toml','rb'))",
        ], check=False)
        if result.returncode != 0:
            print("  ❌ Invalid pyproject.toml syntax")
            self.results["dependencies"] = {"status": "failed", "error": "Invalid pyproject.toml"}
            return False
        
        # Check dependency resolution
        result = self.run_command("uv tree", check=False)
        if result.returncode != 0:
            print("  ❌ Dependency resolution failed")
            self.results["dependencies"] = {"status": "failed", "error": "Dependency resolution failed"}
            return False
        
        print("  ✅ Dependencies are valid")
        self.results["dependencies"] = {"status": "passed"}
        return True
    
    def run_unit_tests(self) -> bool:
        """Run unit tests"""
        print("📋 Running unit tests...")
        
        result = self.run_command("uv run pytest tests/ -v --tb=short", check=False)
        if result.returncode != 0:
            print("  ❌ Unit tests failed")
            self.results["unit_tests"] = {"status": "failed", "output": result.stdout}
            return False
        
        print("  ✅ Unit tests passed")
        self.results["unit_tests"] = {"status": "passed"}
        return True
    
    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        print("🔗 Running integration tests...")
        
        result = self.run_command(["uv", "run", self.python_cmd, "scripts/test_full_integration.py"], check=False)
        if result.returncode != 0:
            print("  ❌ Integration tests failed")
            self.results["integration_tests"] = {"status": "failed", "output": result.stdout}
            return False
        
        print("  ✅ Integration tests passed")
        self.results["integration_tests"] = {"status": "passed"}
        return True
    
    def run_rl_tests(self) -> bool:
        """Run RL tests"""
        print("🧠 Running RL tests...")
        
        result = self.run_command(["uv", "run", self.python_cmd, "scripts/test_rl.py"], check=False)
        if result.returncode != 0:
            print("  ❌ RL tests failed")
            self.results["rl_tests"] = {"status": "failed", "output": result.stdout}
            return False
        
        print("  ✅ RL tests passed")
        self.results["rl_tests"] = {"status": "passed"}
        return True
    
    def run_agent_tests(self) -> bool:
        """Run agent functionality tests"""
        print("🤖 Running agent tests...")
        
        result = self.run_command(["uv", "run", self.python_cmd, "scripts/test_agent_simple.py"], check=False)
        if result.returncode != 0:
            print("  ❌ Agent tests failed")
            self.results["agent_tests"] = {"status": "failed", "output": result.stdout}
            return False
        
        print("  ✅ Agent tests passed")
        self.results["agent_tests"] = {"status": "passed"}
        return True
    
    def run_code_quality_tests(self) -> bool:
        """Run code quality tests"""
        print("🔍 Running code quality tests...")
        
        result = self.run_command("uv run ruff check . --output-format=text", check=False)
        if result.returncode != 0:
            print("  ❌ Code quality checks failed")
            self.results["code_quality"] = {"status": "failed", "output": result.stdout}
            return False
        
        print("  ✅ Code quality checks passed")
        self.results["code_quality"] = {"status": "passed"}
        return True
    
    def run_import_tests(self) -> bool:
        """Run import tests"""
        print("📦 Running import tests...")
        
        result = self.run_command([
            "uv",
            "run",
            self.python_cmd,
            "-c",
            "import dspy_agent; print('✅ Package imports work')",
        ], check=False)
        if result.returncode != 0:
            print("  ❌ Import tests failed")
            self.results["import_tests"] = {"status": "failed", "output": result.stdout}
            return False
        
        print("  ✅ Import tests passed")
        self.results["import_tests"] = {"status": "passed"}
        return True
    
    def run_cli_tests(self) -> bool:
        """Run CLI tests"""
        print("💻 Running CLI tests...")
        
        result = self.run_command("uv run blampert --help", check=False)
        if result.returncode != 0:
            print("  ❌ CLI tests failed")
            self.results["cli_tests"] = {"status": "failed", "output": result.stdout}
            return False
        
        print("  ✅ CLI tests passed")
        self.results["cli_tests"] = {"status": "passed"}
        return True
    
    def validate_build(self) -> bool:
        """Validate package build"""
        print("📦 Validating package build...")
        
        # Clean previous builds
        self.run_command("rm -rf dist/ build/ *.egg-info/", check=False)
        
        # Build package
        result = self.run_command("uv build", check=False)
        if result.returncode != 0:
            print("  ❌ Package build failed")
            self.results["build"] = {"status": "failed", "output": result.stdout}
            return False
        
        # Validate built package
        dist_files = list(self.workspace.glob("dist/*"))
        if not dist_files:
            print("  ❌ No package files found in dist/")
            self.results["build"] = {"status": "failed", "error": "No dist files"}
            return False
        
        # Test wheel file validity
        wheel_file = next((f for f in dist_files if f.suffix == '.whl'), None)
        if wheel_file:
            # Check that wheel file exists and has reasonable size
            if wheel_file.stat().st_size < 1000:  # At least 1KB
                print("  ❌ Wheel file too small")
                self.results["build"] = {"status": "failed", "error": "Wheel file too small"}
                return False
            print(f"  📦 Wheel file: {wheel_file.name} ({wheel_file.stat().st_size} bytes)")
        
        print("  ✅ Package build validation passed")
        self.results["build"] = {"status": "passed"}
        return True
    
    def run_comprehensive_validation(self) -> bool:
        """Run comprehensive validation suite"""
        print("🚀 Starting comprehensive package validation...")
        print("=" * 60)
        
        validation_steps = [
            ("Package Structure", self.validate_package_structure),
            ("Dependencies", self.validate_dependencies),
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("RL Tests", self.run_rl_tests),
            ("Agent Tests", self.run_agent_tests),
            ("Code Quality", self.run_code_quality_tests),
            ("Import Tests", self.run_import_tests),
            ("CLI Tests", self.run_cli_tests),
            ("Build Validation", self.validate_build),
        ]
        
        passed = 0
        failed = 0
        
        for step_name, step_func in validation_steps:
            print(f"\n📋 {step_name}")
            print("-" * 40)
            
            try:
                if step_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ❌ {step_name} failed with exception: {e}")
                self.results[step_name.lower().replace(" ", "_")] = {"status": "failed", "error": str(e)}
                failed += 1
        
        # Summary
        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        
        if failed == 0:
            print("\n🎉 ALL VALIDATIONS PASSED! Package is ready for publishing.")
            self.results["overall"] = {"status": "passed", "passed": passed, "failed": failed, "time": total_time}
            return True
        else:
            print(f"\n⚠️  {failed} VALIDATIONS FAILED! Package is NOT ready for publishing.")
            self.results["overall"] = {"status": "failed", "passed": passed, "failed": failed, "time": total_time}
            return False
    
    def save_results(self, output_file: str = "validation_results.json"):
        """Save validation results to file"""
        self.results["timestamp"] = time.time()
        self.results["workspace"] = str(self.workspace)
        
        with open(self.workspace / output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Package Validation")
    parser.add_argument("--workspace", type=Path, help="Workspace directory")
    parser.add_argument("--output", default="validation_results.json", help="Output file for results")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--type", choices=["structure", "deps", "unit", "integration", "rl", "agent", "quality", "import", "cli", "build"], 
                       help="Run specific validation type")
    
    args = parser.parse_args()
    
    validator = PackageValidator(args.workspace)
    
    if args.type:
        # Run specific validation type
        validation_map = {
            "structure": validator.validate_package_structure,
            "deps": validator.validate_dependencies,
            "unit": validator.run_unit_tests,
            "integration": validator.run_integration_tests,
            "rl": validator.run_rl_tests,
            "agent": validator.run_agent_tests,
            "quality": validator.run_code_quality_tests,
            "import": validator.run_import_tests,
            "cli": validator.run_cli_tests,
            "build": validator.validate_build,
        }
        
        if args.type in validation_map:
            success = validation_map[args.type]()
            validator.save_results(args.output)
            sys.exit(0 if success else 1)
        else:
            print(f"❌ Unknown validation type: {args.type}")
            sys.exit(1)
    
    elif args.quick:
        # Run quick validation
        print("⚡ Running quick validation...")
        success = (
            validator.validate_package_structure() and
            validator.validate_dependencies() and
            validator.run_unit_tests() and
            validator.run_code_quality_tests() and
            validator.run_import_tests()
        )
        validator.save_results(args.output)
        sys.exit(0 if success else 1)
    
    else:
        # Run comprehensive validation
        success = validator.run_comprehensive_validation()
        validator.save_results(args.output)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
