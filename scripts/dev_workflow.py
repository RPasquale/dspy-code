#!/usr/bin/env python3
"""
Streamlined Development Workflow for DSPy Agent
Makes it super easy for human users to publish packages, push to GitHub, and update versions.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

class DevWorkflow:
    def __init__(self, workspace: Path = None):
        self.workspace = workspace or Path.cwd()
        self.pyproject_path = self.workspace / "pyproject.toml"
        self.package_name = "blampert"
        
    def run_command(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result"""
        print(f"🔄 Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.workspace)
        if check and result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result
    
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml"""
        if not self.pyproject_path.exists():
            raise FileNotFoundError("pyproject.toml not found")
        
        content = self.pyproject_path.read_text()
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if not match:
            raise ValueError("Version not found in pyproject.toml")
        return match.group(1)
    
    def update_version(self, version_type: str = "patch") -> str:
        """Update version number in pyproject.toml"""
        current_version = self.get_current_version()
        major, minor, patch = map(int, current_version.split('.'))
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        elif version_type == "patch":
            patch += 1
        else:
            raise ValueError("Version type must be 'major', 'minor', or 'patch'")
        
        new_version = f"{major}.{minor}.{patch}"
        
        # Update pyproject.toml
        content = self.pyproject_path.read_text()
        content = re.sub(
            r'version\s*=\s*["\'][^"\']+["\']',
            f'version = "{new_version}"',
            content
        )
        self.pyproject_path.write_text(content)
        
        print(f"✅ Updated version: {current_version} → {new_version}")
        return new_version
    
    def run_tests(self) -> bool:
        """Run comprehensive test suite and return success status"""
        print("🧪 Running comprehensive test suite...")
        
        # Test 1: Unit tests
        print("  📋 Running unit tests...")
        result = self.run_command("uv run pytest tests/ -v", check=False)
        if result.returncode != 0:
            print("❌ Unit tests failed!")
            print(result.stdout)
            return False
        
        # Test 2: Integration tests
        print("  🔗 Running integration tests...")
        result = self.run_command("uv run python scripts/test_full_integration.py", check=False)
        if result.returncode != 0:
            print("❌ Integration tests failed!")
            print(result.stdout)
            return False
        
        # Test 3: RL tests
        print("  🧠 Running RL tests...")
        result = self.run_command("uv run python scripts/test_rl.py", check=False)
        if result.returncode != 0:
            print("❌ RL tests failed!")
            print(result.stdout)
            return False
        
        # Test 4: Agent functionality tests
        print("  🤖 Running agent functionality tests...")
        result = self.run_command("uv run python scripts/test_agent_simple.py", check=False)
        if result.returncode != 0:
            print("❌ Agent tests failed!")
            print(result.stdout)
            return False
        
        # Test 5: Code quality tests
        print("  🔍 Running code quality tests...")
        result = self.run_command("uv run ruff check . --output-format=text", check=False)
        if result.returncode != 0:
            print("❌ Code quality checks failed!")
            print(result.stdout)
            return False
        
        # Test 6: Import tests
        print("  📦 Testing package imports...")
        result = self.run_command("uv run python -c 'import dspy_agent; print(\"✅ Package imports work\")'", check=False)
        if result.returncode != 0:
            print("❌ Package import tests failed!")
            print(result.stdout)
            return False
        
        # Test 7: CLI tests
        print("  💻 Testing CLI functionality...")
        result = self.run_command("uv run blampert --help", check=False)
        if result.returncode != 0:
            print("❌ CLI tests failed!")
            print(result.stdout)
            return False
        
        print("✅ All tests passed!")
        return True
    
    def run_quick_tests(self) -> bool:
        """Run quick tests for development"""
        print("⚡ Running quick tests...")
        
        # Test 1: Unit tests only
        print("  📋 Running unit tests...")
        result = self.run_command("uv run pytest tests/ -v -x", check=False)
        if result.returncode != 0:
            print("❌ Unit tests failed!")
            return False
        
        # Test 2: Code quality
        print("  🔍 Running code quality checks...")
        result = self.run_command("uv run ruff check .", check=False)
        if result.returncode != 0:
            print("❌ Code quality checks failed!")
            return False
        
        # Test 3: Import test
        print("  📦 Testing imports...")
        result = self.run_command("uv run python -c 'import dspy_agent'", check=False)
        if result.returncode != 0:
            print("❌ Import test failed!")
            return False
        
        print("✅ Quick tests passed!")
        return True
    
    def run_specific_test(self, test_type: str) -> bool:
        """Run a specific type of test"""
        print(f"🧪 Running {test_type} tests...")
        
        if test_type == "unit":
            result = self.run_command("uv run pytest tests/ -v", check=False)
        elif test_type == "integration":
            result = self.run_command("uv run python scripts/test_full_integration.py", check=False)
        elif test_type == "rl":
            result = self.run_command("uv run python scripts/test_rl.py", check=False)
        elif test_type == "agent":
            result = self.run_command("uv run python scripts/test_agent_simple.py", check=False)
        elif test_type == "quality":
            result = self.run_command("uv run ruff check .", check=False)
        elif test_type == "import":
            result = self.run_command("uv run python -c 'import dspy_agent; print(\"✅ Imports work\")'", check=False)
        elif test_type == "cli":
            result = self.run_command("uv run blampert --help", check=False)
        else:
            print(f"❌ Unknown test type: {test_type}")
            return False
        
        if result.returncode == 0:
            print(f"✅ {test_type} tests passed!")
            return True
        else:
            print(f"❌ {test_type} tests failed!")
            print(result.stdout)
            return False
    
    def build_package(self) -> bool:
        """Build the package with comprehensive validation"""
        print("📦 Building package with validation...")
        
        # Step 1: Clean previous builds
        print("  🧹 Cleaning previous builds...")
        self.run_command("rm -rf dist/ build/ *.egg-info/", check=False)
        
        # Step 2: Validate package structure
        print("  📋 Validating package structure...")
        if not self._validate_package_structure():
            print("❌ Package structure validation failed!")
            return False
        
        # Step 3: Check dependencies
        print("  🔍 Checking dependencies...")
        if not self._validate_dependencies():
            print("❌ Dependency validation failed!")
            return False
        
        # Step 4: Build the package
        print("  🔨 Building package...")
        result = self.run_command("uv build", check=False)
        if result.returncode != 0:
            print("❌ Package build failed!")
            print(result.stdout)
            return False
        
        # Step 5: Validate built package
        print("  ✅ Validating built package...")
        if not self._validate_built_package():
            print("❌ Built package validation failed!")
            return False
        
        print("✅ Package built and validated successfully!")
        return True
    
    def _validate_package_structure(self) -> bool:
        """Validate package structure"""
        required_files = [
            "pyproject.toml",
            "README.md",
            "dspy_agent/__init__.py",
            "dspy_agent/cli.py"
        ]
        
        for file_path in required_files:
            if not (self.workspace / file_path).exists():
                print(f"  ❌ Missing required file: {file_path}")
                return False
        
        print("  ✅ Package structure is valid")
        return True
    
    def _validate_dependencies(self) -> bool:
        """Validate dependencies"""
        try:
            # Check if pyproject.toml is valid
            result = self.run_command("uv run python -c 'import tomllib; tomllib.load(open(\"pyproject.toml\", \"rb\"))'", check=False)
            if result.returncode != 0:
                print("  ❌ Invalid pyproject.toml")
                return False
            
            # Check if dependencies can be resolved
            result = self.run_command("uv tree", check=False)
            if result.returncode != 0:
                print("  ❌ Dependency resolution failed")
                return False
            
            print("  ✅ Dependencies are valid")
            return True
        except Exception as e:
            print(f"  ❌ Dependency validation error: {e}")
            return False
    
    def _validate_built_package(self) -> bool:
        """Validate the built package"""
        dist_files = list(self.workspace.glob("dist/*"))
        if not dist_files:
            print("  ❌ No package files found in dist/")
            return False
        
        # Check if wheel can be installed
        wheel_file = next((f for f in dist_files if f.suffix == '.whl'), None)
        if wheel_file:
            print(f"  📦 Testing wheel installation: {wheel_file.name}")
            result = self.run_command(f"uv run pip install {wheel_file} --force-reinstall --no-deps", check=False)
            if result.returncode != 0:
                print("  ❌ Wheel installation test failed")
                return False
        
        print("  ✅ Built package is valid")
        return True
    
    def lint_code(self) -> bool:
        """Run linter and return success status"""
        print("🔍 Running linter...")
        result = self.run_command("uv run ruff check .", check=False)
        if result.returncode == 0:
            print("✅ Linting passed!")
            return True
        else:
            print("❌ Linting failed!")
            print(result.stdout)
            return False
    
    def format_code(self) -> bool:
        """Format code and return success status"""
        print("🎨 Formatting code...")
        result = self.run_command("uv run ruff format .", check=False)
        if result.returncode == 0:
            print("✅ Code formatted!")
            return True
        else:
            print("❌ Code formatting failed!")
            print(result.stdout)
            return False
    
    def git_status(self) -> Dict[str, Any]:
        """Get git status information"""
        result = self.run_command("git status --porcelain", check=False)
        files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        result = self.run_command("git branch --show-current", check=False)
        current_branch = result.stdout.strip()
        
        return {
            "current_branch": current_branch,
            "modified_files": [f for f in files if f.startswith(' M')],
            "new_files": [f for f in files if f.startswith('??')],
            "staged_files": [f for f in files if f.startswith('A ') or f.startswith('M ')],
        }
    
    def git_commit(self, message: str, files: List[str] = None) -> bool:
        """Commit changes to git"""
        if files:
            self.run_command(f"git add {' '.join(files)}")
        else:
            self.run_command("git add .")
        
        result = self.run_command(f'git commit -m "{message}"', check=False)
        if result.returncode == 0:
            print(f"✅ Committed: {message}")
            return True
        else:
            print("❌ Commit failed!")
            print(result.stdout)
            return False
    
    def git_push(self, branch: str = None) -> bool:
        """Push to GitHub"""
        if not branch:
            status = self.git_status()
            branch = status["current_branch"]
        
        result = self.run_command(f"git push origin {branch}", check=False)
        if result.returncode == 0:
            print(f"✅ Pushed to GitHub: {branch}")
            return True
        else:
            print("❌ Push failed!")
            print(result.stdout)
            return False
    
    def create_github_release(self, version: str, message: str = None) -> bool:
        """Create a GitHub release"""
        if not message:
            message = f"Release {version}"
        
        # Create tag
        self.run_command(f"git tag -a v{version} -m '{message}'")
        
        # Push tag
        result = self.run_command(f"git push origin v{version}", check=False)
        if result.returncode == 0:
            print(f"✅ Created GitHub release: v{version}")
            return True
        else:
            print("❌ GitHub release failed!")
            print(result.stdout)
            return False
    
    def publish_to_pypi(self, test: bool = False) -> bool:
        """Publish package to PyPI with comprehensive pre-publish validation"""
        print("📤 Preparing to publish to PyPI...")
        
        # Pre-publish validation
        print("🔍 Running pre-publish validation...")
        
        # 1. Run comprehensive tests
        if not self.run_tests():
            print("❌ Pre-publish tests failed! Aborting publish.")
            return False
        
        # 2. Run linting
        if not self.lint_code():
            print("❌ Pre-publish linting failed! Aborting publish.")
            return False
        
        # 3. Build and validate package
        if not self.build_package():
            print("❌ Pre-publish build failed! Aborting publish.")
            return False
        
        # 4. Check git status
        print("  📋 Checking git status...")
        result = self.run_command("git status --porcelain", check=False)
        if result.stdout.strip():
            print("⚠️  Warning: You have uncommitted changes!")
            print("   Consider committing before publishing.")
        
        # 5. Check if version exists
        current_version = self.get_current_version()
        print(f"  📦 Publishing version: {current_version}")
        
        # 6. Final confirmation
        if not test:
            print("🚨 WARNING: You are about to publish to PRODUCTION PyPI!")
            print("   This will be visible to all users.")
            response = input("   Type 'PUBLISH' to confirm: ")
            if response != "PUBLISH":
                print("❌ Publish cancelled.")
                return False
        
        # 7. Publish
        if test:
            print("📤 Publishing to Test PyPI...")
            result = self.run_command("uv publish --repository testpypi", check=False)
        else:
            print("📤 Publishing to PyPI...")
            result = self.run_command("uv publish", check=False)
        
        if result.returncode == 0:
            print("✅ Published to PyPI!")
            
            # Post-publish validation
            print("🔍 Running post-publish validation...")
            if not self._validate_published_package(test):
                print("⚠️  Post-publish validation failed, but package was published.")
            
            return True
        else:
            print("❌ PyPI publish failed!")
            print(result.stdout)
            return False
    
    def _validate_published_package(self, test: bool = False) -> bool:
        """Validate the published package"""
        try:
            current_version = self.get_current_version()
            package_name = self.package_name
            
            print(f"  📦 Testing installation of {package_name}=={current_version}")
            
            # Test installation from PyPI
            if test:
                install_cmd = f"uv run pip install --index-url https://test.pypi.org/simple/ {package_name}=={current_version}"
            else:
                install_cmd = f"uv run pip install {package_name}=={current_version}"
            
            result = self.run_command(install_cmd, check=False)
            if result.returncode != 0:
                print("  ❌ Published package installation test failed")
                return False
            
            # Test that the package works
            result = self.run_command(f"uv run {package_name} --help", check=False)
            if result.returncode != 0:
                print("  ❌ Published package functionality test failed")
                return False
            
            print("  ✅ Published package validation successful")
            return True
            
        except Exception as e:
            print(f"  ❌ Post-publish validation error: {e}")
            return False
    
    def full_release(self, version_type: str = "patch", skip_tests: bool = False, 
                    skip_lint: bool = False, test_pypi: bool = False) -> bool:
        """Complete release workflow"""
        print("🚀 Starting full release workflow...")
        
        # 1. Check git status
        status = self.git_status()
        if status["modified_files"] or status["new_files"]:
            print("⚠️  You have uncommitted changes. Please commit them first.")
            return False
        
        # 2. Run tests (unless skipped)
        if not skip_tests:
            if not self.run_tests():
                print("❌ Tests failed. Aborting release.")
                return False
        
        # 3. Run linter (unless skipped)
        if not skip_lint:
            if not self.lint_code():
                print("❌ Linting failed. Aborting release.")
                return False
        
        # 4. Update version
        new_version = self.update_version(version_type)
        
        # 5. Build package
        if not self.build_package():
            print("❌ Package build failed. Aborting release.")
            return False
        
        # 6. Commit version change
        self.git_commit(f"Bump version to {new_version}", ["pyproject.toml"])
        
        # 7. Push to GitHub
        if not self.git_push():
            print("❌ Push to GitHub failed. Aborting release.")
            return False
        
        # 8. Create GitHub release
        if not self.create_github_release(new_version):
            print("❌ GitHub release failed. Aborting release.")
            return False
        
        # 9. Publish to PyPI
        if not self.publish_to_pypi(test=test_pypi):
            print("❌ PyPI publish failed. Aborting release.")
            return False
        
        print(f"🎉 Release {new_version} completed successfully!")
        return True
    
    def quick_dev_cycle(self, message: str = None) -> bool:
        """Quick development cycle: format, test, commit, push"""
        print("⚡ Starting quick dev cycle...")
        
        # 1. Format code
        self.format_code()
        
        # 2. Run tests
        if not self.run_tests():
            print("❌ Tests failed. Aborting dev cycle.")
            return False
        
        # 3. Commit and push
        if not message:
            message = f"Quick dev update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        if not self.git_commit(message):
            print("❌ Commit failed. Aborting dev cycle.")
            return False
        
        if not self.git_push():
            print("❌ Push failed. Aborting dev cycle.")
            return False
        
        print("✅ Quick dev cycle completed!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Blampert Development Workflow")
    parser.add_argument("--workspace", type=Path, help="Workspace directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Version commands
    version_parser = subparsers.add_parser("version", help="Version management")
    version_parser.add_argument("--type", choices=["major", "minor", "patch"], 
                               default="patch", help="Version bump type")
    version_parser.add_argument("--show", action="store_true", help="Show current version")
    
    # Test commands
    test_parser = subparsers.add_parser("test", help="Run comprehensive test suite")
    test_parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    test_parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    test_parser.add_argument("--type", choices=["unit", "integration", "rl", "agent", "quality", "import", "cli"], 
                           help="Run specific test type")
    
    # Build commands
    build_parser = subparsers.add_parser("build", help="Build package")
    
    # Lint commands
    lint_parser = subparsers.add_parser("lint", help="Run linter")
    lint_parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    
    # Git commands
    git_parser = subparsers.add_parser("git", help="Git operations")
    git_parser.add_argument("--status", action="store_true", help="Show git status")
    git_parser.add_argument("--commit", type=str, help="Commit message")
    git_parser.add_argument("--push", action="store_true", help="Push to GitHub")
    
    # Release commands
    release_parser = subparsers.add_parser("release", help="Full release workflow")
    release_parser.add_argument("--type", choices=["major", "minor", "patch"], 
                               default="patch", help="Version bump type")
    release_parser.add_argument("--skip-tests", action="store_true", help="Skip tests")
    release_parser.add_argument("--skip-lint", action="store_true", help="Skip linting")
    release_parser.add_argument("--test-pypi", action="store_true", help="Publish to Test PyPI")
    
    # Quick dev cycle
    quick_parser = subparsers.add_parser("quick", help="Quick development cycle")
    quick_parser.add_argument("--message", type=str, help="Commit message")
    
    # Publish commands
    publish_parser = subparsers.add_parser("publish", help="Publish to PyPI")
    publish_parser.add_argument("--test", action="store_true", help="Publish to Test PyPI")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    workflow = DevWorkflow(args.workspace)
    
    try:
        if args.command == "version":
            if args.show:
                version = workflow.get_current_version()
                print(f"Current version: {version}")
            else:
                workflow.update_version(args.type)
        
        elif args.command == "test":
            if args.type:
                workflow.run_specific_test(args.type)
            elif args.quick:
                workflow.run_quick_tests()
            else:
                workflow.run_tests()
        
        elif args.command == "build":
            workflow.build_package()
        
        elif args.command == "lint":
            if args.fix:
                workflow.format_code()
            workflow.lint_code()
        
        elif args.command == "git":
            if args.status:
                status = workflow.git_status()
                print(f"Current branch: {status['current_branch']}")
                if status['modified_files']:
                    print("Modified files:", status['modified_files'])
                if status['new_files']:
                    print("New files:", status['new_files'])
                if status['staged_files']:
                    print("Staged files:", status['staged_files'])
            if args.commit:
                workflow.git_commit(args.commit)
            if args.push:
                workflow.git_push()
        
        elif args.command == "release":
            workflow.full_release(
                version_type=args.type,
                skip_tests=args.skip_tests,
                skip_lint=args.skip_lint,
                test_pypi=args.test_pypi
            )
        
        elif args.command == "quick":
            workflow.quick_dev_cycle(args.message)
        
        elif args.command == "publish":
            workflow.publish_to_pypi(test=args.test)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
