#!/usr/bin/env python3
"""
Comprehensive Test Runner for DSPy Agent
This script runs all tests with detailed reporting and coverage analysis.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import logging
import tempfile
import shutil


class TestRunner:
    """Comprehensive test runner for DSPy agent."""
    
    def __init__(self, workspace: Path, verbose: bool = False):
        self.workspace = Path(workspace)
        self.verbose = verbose
        self.log_dir = self.workspace / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "test_runner.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Test results
        self.test_results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> Dict[str, Any]:
        """Run a command and return results."""
        if cwd is None:
            cwd = self.workspace
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'success': False
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        self.logger.info("Running unit tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/',
            '-v',
            '--tb=short',
            '--maxfail=5',
            '--junitxml=test-results/unit-tests.xml',
            '--html=test-results/unit-tests.html',
            '--self-contained-html',
            '--cov=dspy_agent',
            '--cov-report=xml:test-results/coverage.xml',
            '--cov-report=html:test-results/coverage-html'
        ]
        
        # Create test results directory
        test_results_dir = self.workspace / "test-results"
        test_results_dir.mkdir(exist_ok=True)
        
        result = self.run_command(cmd)
        
        return {
            'name': 'unit_tests',
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        self.logger.info("Running integration tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/test_comprehensive_agent.py::TestAgentComprehensive',
            '-v',
            '--junitxml=test-results/integration-tests.xml',
            '--html=test-results/integration-tests.html',
            '--self-contained-html'
        ]
        
        result = self.run_command(cmd)
        
        return {
            'name': 'integration_tests',
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_docker_tests(self) -> Dict[str, Any]:
        """Run Docker-related tests."""
        self.logger.info("Running Docker tests...")
        
        # Test Docker build
        docker_dir = self.workspace / "docker" / "lightweight"
        if not docker_dir.exists():
            return {
                'name': 'docker_tests',
                'result': {
                    'returncode': -1,
                    'stdout': '',
                    'stderr': 'Docker directory not found',
                    'success': False
                },
                'timestamp': datetime.now().isoformat()
            }
        
        # Test Docker Compose configuration
        compose_result = self.run_command(['docker-compose', 'config'], cwd=docker_dir)
        
        # Test Docker build
        build_result = self.run_command(['docker', 'build', '-t', 'dspy-test', '.'], cwd=docker_dir)
        
        # Test Docker image
        test_result = self.run_command(['docker', 'run', '--rm', 'dspy-test', '--help'])
        
        # Cleanup
        self.run_command(['docker', 'rmi', 'dspy-test'])
        
        return {
            'name': 'docker_tests',
            'result': {
                'returncode': 0 if all(r['success'] for r in [compose_result, build_result, test_result]) else 1,
                'stdout': f"Compose: {compose_result['stdout']}\nBuild: {build_result['stdout']}\nTest: {test_result['stdout']}",
                'stderr': f"Compose: {compose_result['stderr']}\nBuild: {build_result['stderr']}\nTest: {test_result['stderr']}",
                'success': all(r['success'] for r in [compose_result, build_result, test_result])
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        self.logger.info("Running performance tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/test_comprehensive_agent.py::TestAgentComprehensive::test_performance_metrics',
            '-v',
            '--benchmark-only',
            '--benchmark-save=performance_results'
        ]
        
        result = self.run_command(cmd)
        
        return {
            'name': 'performance_tests',
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        self.logger.info("Running security tests...")
        
        # Check for common security issues
        security_issues = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            'password=',
            'secret=',
            'api_key=',
            'token=',
            'private_key='
        ]
        
        for pattern in secret_patterns:
            result = self.run_command(['grep', '-r', '-i', pattern, 'dspy_agent/'])
            if result['returncode'] == 0:
                security_issues.append(f"Potential hardcoded secret found: {pattern}")
        
        # Check for SQL injection patterns
        sql_patterns = [
            'SELECT.*FROM',
            'INSERT.*INTO',
            'UPDATE.*SET',
            'DELETE.*FROM'
        ]
        
        for pattern in sql_patterns:
            result = self.run_command(['grep', '-r', '-i', pattern, 'dspy_agent/'])
            if result['returncode'] == 0:
                security_issues.append(f"Potential SQL query found: {pattern}")
        
        # Check file permissions
        permission_issues = []
        for root, dirs, files in os.walk(self.workspace / "dspy_agent"):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix == '.py':
                    stat = file_path.stat()
                    if stat.st_mode & 0o777 > 0o644:
                        permission_issues.append(f"File {file_path} has overly permissive permissions")
        
        # Determine overall security status
        all_issues = security_issues + permission_issues
        if all_issues:
            status = "warning" if len(all_issues) < 5 else "critical"
            message = f"Found {len(all_issues)} security issues"
        else:
            status = "healthy"
            message = "No security issues found"
        
        return {
            'name': 'security_tests',
            'result': {
                'returncode': 0 if status == "healthy" else 1,
                'stdout': f"Security status: {status}\n{message}",
                'stderr': '\n'.join(all_issues) if all_issues else '',
                'success': status == "healthy"
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def run_lint_tests(self) -> Dict[str, Any]:
        """Run linting tests."""
        self.logger.info("Running linting tests...")
        
        # Run flake8
        flake8_result = self.run_command([
            'python', '-m', 'flake8',
            'dspy_agent/',
            '--count',
            '--select=E9,F63,F7,F82',
            '--show-source',
            '--statistics'
        ])
        
        # Run black check
        black_result = self.run_command([
            'python', '-m', 'black',
            '--check',
            'dspy_agent/'
        ])
        
        # Run mypy
        mypy_result = self.run_command([
            'python', '-m', 'mypy',
            'dspy_agent/',
            '--ignore-missing-imports',
            '--no-strict-optional'
        ])
        
        # Combine results
        all_success = all(r['success'] for r in [flake8_result, black_result, mypy_result])
        
        return {
            'name': 'lint_tests',
            'result': {
                'returncode': 0 if all_success else 1,
                'stdout': f"Flake8: {flake8_result['stdout']}\nBlack: {black_result['stdout']}\nMyPy: {mypy_result['stdout']}",
                'stderr': f"Flake8: {flake8_result['stderr']}\nBlack: {black_result['stderr']}\nMyPy: {mypy_result['stderr']}",
                'success': all_success
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def run_package_tests(self) -> Dict[str, Any]:
        """Run package build and validation tests."""
        self.logger.info("Running package tests...")
        
        # Test package build
        build_result = self.run_command(['python', '-m', 'build'])
        
        # Test package validation
        if build_result['success']:
            validate_result = self.run_command(['python', '-m', 'twine', 'check', 'dist/*'])
        else:
            validate_result = {'returncode': -1, 'stdout': '', 'stderr': 'Build failed', 'success': False}
        
        # Test package installation
        if validate_result['success']:
            # Create temporary directory for installation test
            with tempfile.TemporaryDirectory() as temp_dir:
                install_result = self.run_command([
                    'python', '-m', 'pip', 'install',
                    '--target', temp_dir,
                    'dist/*.whl'
                ])
        else:
            install_result = {'returncode': -1, 'stdout': '', 'stderr': 'Validation failed', 'success': False}
        
        # Combine results
        all_success = all(r['success'] for r in [build_result, validate_result, install_result])
        
        return {
            'name': 'package_tests',
            'result': {
                'returncode': 0 if all_success else 1,
                'stdout': f"Build: {build_result['stdout']}\nValidate: {validate_result['stdout']}\nInstall: {install_result['stdout']}",
                'stderr': f"Build: {build_result['stderr']}\nValidate: {validate_result['stderr']}\nInstall: {install_result['stderr']}",
                'success': all_success
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def run_all_tests(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Run all specified tests."""
        if test_types is None:
            test_types = ['unit', 'integration', 'docker', 'performance', 'security', 'lint', 'package']
        
        self.start_time = datetime.now()
        self.logger.info(f"Starting comprehensive test run: {', '.join(test_types)}")
        
        results = {}
        
        # Run each test type
        for test_type in test_types:
            self.logger.info(f"Running {test_type} tests...")
            
            try:
                if test_type == 'unit':
                    results[test_type] = self.run_unit_tests()
                elif test_type == 'integration':
                    results[test_type] = self.run_integration_tests()
                elif test_type == 'docker':
                    results[test_type] = self.run_docker_tests()
                elif test_type == 'performance':
                    results[test_type] = self.run_performance_tests()
                elif test_type == 'security':
                    results[test_type] = self.run_security_tests()
                elif test_type == 'lint':
                    results[test_type] = self.run_lint_tests()
                elif test_type == 'package':
                    results[test_type] = self.run_package_tests()
                else:
                    self.logger.warning(f"Unknown test type: {test_type}")
                    continue
                
                self.logger.info(f"{test_type} tests completed")
                
            except Exception as e:
                self.logger.error(f"Error running {test_type} tests: {e}")
                results[test_type] = {
                    'name': f'{test_type}_tests',
                    'result': {
                        'returncode': -1,
                        'stdout': '',
                        'stderr': str(e),
                        'success': False
                    },
                    'timestamp': datetime.now().isoformat()
                }
        
        self.end_time = datetime.now()
        self.test_results = results
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.test_results:
            return {"error": "No test results available"}
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results.values() if r['result']['success'])
        failed_tests = total_tests - successful_tests
        
        # Calculate duration
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "successful": successful_tests,
                "failed": failed_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "duration_seconds": duration
            },
            "test_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for test_name, test_result in self.test_results.items():
            if not test_result['result']['success']:
                if test_name == 'unit_tests':
                    recommendations.append("Fix failing unit tests - check test output for details")
                elif test_name == 'integration_tests':
                    recommendations.append("Fix integration test failures - check component interactions")
                elif test_name == 'docker_tests':
                    recommendations.append("Fix Docker build or configuration issues")
                elif test_name == 'performance_tests':
                    recommendations.append("Optimize performance - check benchmark results")
                elif test_name == 'security_tests':
                    recommendations.append("Address security issues - review security test output")
                elif test_name == 'lint_tests':
                    recommendations.append("Fix code style issues - run black and flake8")
                elif test_name == 'package_tests':
                    recommendations.append("Fix package build issues - check build output")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save test report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        report_path = self.log_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Test report saved to {report_path}")
        return report_path


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="DSPy Agent Comprehensive Test Runner")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--test-types", nargs="+", 
                       choices=['unit', 'integration', 'docker', 'performance', 'security', 'lint', 'package'],
                       default=['unit', 'integration', 'docker', 'lint', 'package'],
                       help="Test types to run")
    parser.add_argument("--output", type=str, help="Output file for test report")
    
    args = parser.parse_args()
    
    workspace = Path(args.workspace).resolve()
    
    runner = TestRunner(workspace, args.verbose)
    
    # Run tests
    results = runner.run_all_tests(args.test_types)
    
    # Generate report
    report = runner.generate_report()
    
    # Save report
    if args.output:
        report_path = runner.save_report(report, args.output)
    else:
        report_path = runner.save_report(report)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Successful: {report['summary']['successful']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    if report['summary']['duration_seconds']:
        print(f"Duration: {report['summary']['duration_seconds']:.1f} seconds")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    print(f"\nReport saved to: {report_path}")
    
    # Exit with appropriate code
    if report['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
