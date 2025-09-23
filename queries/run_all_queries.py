#!/usr/bin/env python3
"""
RedDB Query Runner - Run All Analysis Scripts

This script runs all the RedDB analysis scripts and provides a comprehensive
overview of your data landscape, streaming capabilities, and RL training status.

Usage:
    python queries/run_all_queries.py
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class RedDBQueryRunner:
    """Runner for all RedDB analysis scripts"""
    
    def __init__(self):
        self.queries_dir = Path(__file__).parent
        self.scripts = [
            "data_overview.py",
            "streaming_data_analysis.py", 
            "rl_training_analysis.py"
        ]
        self.results = {}
        
    def run_all_queries(self) -> Dict[str, Any]:
        """Run all analysis scripts"""
        print("🚀 Running All RedDB Analysis Scripts...")
        print("=" * 60)
        
        start_time = time.time()
        
        for script in self.scripts:
            script_path = self.queries_dir / script
            if script_path.exists():
                print(f"\n📊 Running {script}...")
                print("-" * 40)
                
                try:
                    # Run the script
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True,
                        text=True,
                        cwd=self.queries_dir
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ {script} completed successfully")
                        self.results[script] = {
                            "status": "success",
                            "output": result.stdout,
                            "error": result.stderr
                        }
                    else:
                        print(f"❌ {script} failed with return code {result.returncode}")
                        print(f"Error: {result.stderr}")
                        self.results[script] = {
                            "status": "failed",
                            "output": result.stdout,
                            "error": result.stderr,
                            "return_code": result.returncode
                        }
                        
                except Exception as e:
                    print(f"❌ {script} failed with exception: {e}")
                    self.results[script] = {
                        "status": "exception",
                        "error": str(e)
                    }
            else:
                print(f"⚠️  Script {script} not found")
                self.results[script] = {
                    "status": "not_found"
                }
        
        execution_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(execution_time)
        
        return {
            "timestamp": time.time(),
            "execution_time": execution_time,
            "scripts_run": len(self.scripts),
            "results": self.results,
            "summary": summary
        }
    
    def _generate_summary(self, execution_time: float) -> Dict[str, Any]:
        """Generate a summary of all query results"""
        print("\n" + "=" * 60)
        print("📋 REDDB ANALYSIS SUMMARY")
        print("=" * 60)
        
        successful_scripts = sum(1 for result in self.results.values() if result["status"] == "success")
        failed_scripts = sum(1 for result in self.results.values() if result["status"] in ["failed", "exception"])
        
        summary = {
            "total_scripts": len(self.scripts),
            "successful_scripts": successful_scripts,
            "failed_scripts": failed_scripts,
            "success_rate": successful_scripts / len(self.scripts) * 100,
            "execution_time": execution_time,
            "status": "success" if failed_scripts == 0 else "partial" if successful_scripts > 0 else "failed"
        }
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   • Total scripts: {summary['total_scripts']}")
        print(f"   • Successful: {summary['successful_scripts']}")
        print(f"   • Failed: {summary['failed_scripts']}")
        print(f"   • Success rate: {summary['success_rate']:.1f}%")
        print(f"   • Execution time: {execution_time:.2f} seconds")
        print(f"   • Overall status: {summary['status'].upper()}")
        
        # Show individual script results
        print(f"\n📋 SCRIPT RESULTS:")
        for script, result in self.results.items():
            status_emoji = "✅" if result["status"] == "success" else "❌" if result["status"] in ["failed", "exception"] else "⚠️"
            print(f"   {status_emoji} {script}: {result['status']}")
        
        # Show generated files
        print(f"\n💾 GENERATED FILES:")
        output_files = [
            "data_overview.json",
            "streaming_analysis.json", 
            "rl_training_analysis.json"
        ]
        
        for output_file in output_files:
            file_path = self.queries_dir / output_file
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"   ✅ {output_file} ({file_size} bytes)")
            else:
                print(f"   ❌ {output_file} (not found)")
        
        return summary
    
    def print_quick_overview(self):
        """Print a quick overview of the RedDB system"""
        print("\n" + "=" * 60)
        print("🔍 REDDB QUICK OVERVIEW")
        print("=" * 60)
        
        try:
            from dspy_agent.db import get_enhanced_data_manager
            
            dm = get_enhanced_data_manager()
            
            # Quick stats
            actions = dm.get_recent_actions(limit=100)
            logs = dm.get_recent_logs(limit=100)
            signatures = dm.get_all_signature_metrics()
            retrieval_events = dm.get_recent_retrieval_events(limit=100)
            training_history = dm.get_training_history(limit=50)
            
            print(f"\n📊 QUICK STATS:")
            print(f"   • Recent actions: {len(actions)}")
            print(f"   • Recent logs: {len(logs)}")
            print(f"   • DSPy signatures: {len(signatures)}")
            print(f"   • Retrieval events: {len(retrieval_events)}")
            print(f"   • Training sessions: {len(training_history)}")
            
            # System health
            health = dm.get_system_health()
            context = dm.get_current_context()
            
            print(f"\n🏥 SYSTEM STATUS:")
            if health:
                print(f"   • System health: Available")
            else:
                print(f"   • System health: Not available")
            
            if context:
                print(f"   • Agent state: {context.agent_state.value}")
                print(f"   • Current task: {context.current_task or 'None'}")
            else:
                print(f"   • Agent state: No active context")
            
            # Cache performance
            cache_stats = dm.get_cache_stats()
            print(f"\n⚡ CACHE PERFORMANCE:")
            print(f"   • Main cache: {cache_stats.get('main_cache', {})}")
            print(f"   • Query cache: {cache_stats.get('query_cache', {})}")
            
        except Exception as e:
            print(f"❌ Error getting quick overview: {e}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save all results to a summary file"""
        import json
        
        output_file = self.queries_dir / "query_results_summary.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results summary saved to: {output_file}")


def main():
    """Main function to run all queries"""
    try:
        runner = RedDBQueryRunner()
        
        # Print quick overview first
        runner.print_quick_overview()
        
        # Run all analysis scripts
        results = runner.run_all_queries()
        
        # Save results
        runner.save_results(results)
        
        # Final status
        if results["summary"]["status"] == "success":
            print(f"\n🎉 All RedDB analysis scripts completed successfully!")
        elif results["summary"]["status"] == "partial":
            print(f"\n⚠️  Some RedDB analysis scripts failed. Check individual results above.")
        else:
            print(f"\n❌ All RedDB analysis scripts failed. Check your RedDB setup.")
        
        print(f"\n⏰ Total execution time: {results['execution_time']:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error running RedDB queries: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
