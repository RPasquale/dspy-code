#!/usr/bin/env python3
"""
RedDB Data Overview Script

This script provides a comprehensive overview of all data types stored in RedDB,
showing the data landscape for the DSPy agent's streaming and RL training system.

Usage:
    python queries/data_overview.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dspy_agent.db import get_enhanced_data_manager, Environment, ActionType, AgentState
from dspy_agent.agentic import get_retrieval_statistics, query_retrieval_events


class RedDBDataExplorer:
    """Explorer for RedDB data landscape"""
    
    def __init__(self):
        self.dm = get_enhanced_data_manager()
        self.storage = self.dm.storage
        self.namespace = "dspy_agent"
        
    def get_data_overview(self) -> Dict[str, Any]:
        """Get comprehensive overview of all data in RedDB"""
        print("🔍 Exploring RedDB Data Landscape...")
        print("=" * 60)
        
        overview = {
            "timestamp": time.time(),
            "data_types": {},
            "streams": {},
            "collections": {},
            "performance_metrics": {},
            "system_health": {}
        }
        
        # 1. Explore Data Types
        overview["data_types"] = self._explore_data_types()
        
        # 2. Explore Streams
        overview["streams"] = self._explore_streams()
        
        # 3. Explore Collections/Registries
        overview["collections"] = self._explore_collections()
        
        # 4. Performance Metrics
        overview["performance_metrics"] = self._explore_performance_metrics()
        
        # 5. System Health
        overview["system_health"] = self._explore_system_health()
        
        return overview
    
    def _explore_data_types(self) -> Dict[str, Any]:
        """Explore different data types stored in RedDB"""
        print("\n📊 DATA TYPES OVERVIEW")
        print("-" * 40)
        
        data_types = {
            "embeddings": {"count": 0, "sources": set()},
            "signatures": {"count": 0, "active": 0, "types": set()},
            "verifiers": {"count": 0, "statuses": set()},
            "actions": {"count": 0, "types": defaultdict(int), "rewards": {"min": float('inf'), "max": 0, "avg": 0}},
            "logs": {"count": 0, "levels": defaultdict(int), "sources": set()},
            "contexts": {"count": 0, "states": set()},
            "patches": {"count": 0, "applied": 0, "confidence_avg": 0},
            "retrieval_events": {"count": 0, "queries": set(), "avg_hits": 0},
            "training_sessions": {"count": 0, "models": set(), "environments": set()}
        }
        
        # Count embeddings (this would need to be implemented based on your storage strategy)
        try:
            # Placeholder - would need actual embedding storage implementation
            data_types["embeddings"]["count"] = 0
            data_types["embeddings"]["sources"] = set()
        except Exception as e:
            print(f"⚠️  Embeddings: {e}")
        
        # Count signatures
        try:
            signatures = self.dm.get_all_signature_metrics()
            data_types["signatures"]["count"] = len(signatures)
            data_types["signatures"]["active"] = sum(1 for s in signatures if s.active)
            data_types["signatures"]["types"] = set(s.signature_type for s in signatures)
            print(f"✅ Signatures: {len(signatures)} total, {data_types['signatures']['active']} active")
        except Exception as e:
            print(f"⚠️  Signatures: {e}")
        
        # Count verifiers
        try:
            verifiers = self.dm.get_all_verifier_metrics()
            data_types["verifiers"]["count"] = len(verifiers)
            data_types["verifiers"]["statuses"] = set(v.status for v in verifiers)
            print(f"✅ Verifiers: {len(verifiers)} total")
        except Exception as e:
            print(f"⚠️  Verifiers: {e}")
        
        # Count actions
        try:
            actions = self.dm.get_recent_actions(limit=1000)
            data_types["actions"]["count"] = len(actions)
            
            if actions:
                rewards = [a.reward for a in actions]
                data_types["actions"]["rewards"]["min"] = min(rewards)
                data_types["actions"]["rewards"]["max"] = max(rewards)
                data_types["actions"]["rewards"]["avg"] = sum(rewards) / len(rewards)
                
                for action in actions:
                    data_types["actions"]["types"][action.action_type.value] += 1
            
            print(f"✅ Actions: {len(actions)} recent actions")
            print(f"   Reward range: {data_types['actions']['rewards']['min']:.3f} - {data_types['actions']['rewards']['max']:.3f}")
        except Exception as e:
            print(f"⚠️  Actions: {e}")
        
        # Count logs
        try:
            logs = self.dm.get_recent_logs(limit=1000)
            data_types["logs"]["count"] = len(logs)
            
            for log in logs:
                data_types["logs"]["levels"][log.level] += 1
                data_types["logs"]["sources"].add(log.source)
            
            print(f"✅ Logs: {len(logs)} recent entries")
            print(f"   Levels: {dict(data_types['logs']['levels'])}")
        except Exception as e:
            print(f"⚠️  Logs: {e}")
        
        # Count contexts
        try:
            context = self.dm.get_current_context()
            if context:
                data_types["contexts"]["count"] = 1
                data_types["contexts"]["states"].add(context.agent_state.value)
                print(f"✅ Context: Current state = {context.agent_state.value}")
            else:
                print("ℹ️  Context: No current context")
        except Exception as e:
            print(f"⚠️  Context: {e}")
        
        # Count patches
        try:
            patches = self.dm.get_patch_history(limit=1000)
            data_types["patches"]["count"] = len(patches)
            data_types["patches"]["applied"] = sum(1 for p in patches if p.applied)
            
            if patches:
                confidences = [p.confidence_score for p in patches]
                data_types["patches"]["confidence_avg"] = sum(confidences) / len(confidences)
            
            print(f"✅ Patches: {len(patches)} total, {data_types['patches']['applied']} applied")
        except Exception as e:
            print(f"⚠️  Patches: {e}")
        
        # Count retrieval events
        try:
            retrieval_events = self.dm.get_recent_retrieval_events(limit=1000)
            data_types["retrieval_events"]["count"] = len(retrieval_events)
            
            if retrieval_events:
                total_hits = sum(len(event.hits) for event in retrieval_events)
                data_types["retrieval_events"]["avg_hits"] = total_hits / len(retrieval_events)
                data_types["retrieval_events"]["queries"] = set(event.query for event in retrieval_events)
            
            print(f"✅ Retrieval Events: {len(retrieval_events)} total")
            print(f"   Average hits per query: {data_types['retrieval_events']['avg_hits']:.1f}")
        except Exception as e:
            print(f"⚠️  Retrieval Events: {e}")
        
        # Count training sessions
        try:
            training_history = self.dm.get_training_history(limit=1000)
            data_types["training_sessions"]["count"] = len(training_history)
            
            for session in training_history:
                data_types["training_sessions"]["models"].add(session.model_type)
                data_types["training_sessions"]["environments"].add(session.environment.value)
            
            print(f"✅ Training Sessions: {len(training_history)} total")
        except Exception as e:
            print(f"⚠️  Training Sessions: {e}")
        
        # Convert sets to lists for JSON serialization
        for data_type, info in data_types.items():
            for key, value in info.items():
                if isinstance(value, set):
                    info[key] = list(value)
                elif isinstance(value, defaultdict):
                    info[key] = dict(value)
        
        return data_types
    
    def _explore_streams(self) -> Dict[str, Any]:
        """Explore time-series streams in RedDB"""
        print("\n🌊 STREAMS OVERVIEW")
        print("-" * 40)
        
        streams = {
            "signature_metrics": {"count": 0, "latest": None},
            "verifier_metrics": {"count": 0, "latest": None},
            "training_history": {"count": 0, "latest": None},
            "rl_actions": {"count": 0, "latest": None},
            "system_logs": {"count": 0, "latest": None},
            "retrieval_events": {"count": 0, "latest": None},
            "context_history": {"count": 0, "latest": None},
            "patch_history": {"count": 0, "latest": None},
            "health_history": {"count": 0, "latest": None}
        }
        
        stream_names = list(streams.keys())
        
        for stream_name in stream_names:
            try:
                count = 0
                latest = None
                
                # Read from stream
                for offset, data in self.storage.read(stream_name, count=1000):
                    count += 1
                    if latest is None:
                        latest = data
                
                streams[stream_name]["count"] = count
                streams[stream_name]["latest"] = latest
                
                print(f"✅ {stream_name}: {count} entries")
                if latest:
                    timestamp = latest.get('timestamp', 'unknown')
                    print(f"   Latest: {timestamp}")
                
            except Exception as e:
                print(f"⚠️  {stream_name}: {e}")
        
        return streams
    
    def _explore_collections(self) -> Dict[str, Any]:
        """Explore collections and registries"""
        print("\n📚 COLLECTIONS & REGISTRIES")
        print("-" * 40)
        
        collections = {
            "signature_registry": {"count": 0, "items": []},
            "verifier_registry": {"count": 0, "items": []},
            "embedding_collection": {"count": 0, "sources": set()},
            "action_collection": {"count": 0, "types": set()},
            "log_collection": {"count": 0, "levels": set()}
        }
        
        # Check registries
        try:
            # Signature registry
            sig_registry_key = f"{self.namespace}:registries:signatures"
            sig_registry = self.storage.get(sig_registry_key) or []
            collections["signature_registry"]["count"] = len(sig_registry)
            collections["signature_registry"]["items"] = sig_registry
            print(f"✅ Signature Registry: {len(sig_registry)} signatures")
        except Exception as e:
            print(f"⚠️  Signature Registry: {e}")
        
        try:
            # Verifier registry
            ver_registry_key = f"{self.namespace}:registries:verifiers"
            ver_registry = self.storage.get(ver_registry_key) or []
            collections["verifier_registry"]["count"] = len(ver_registry)
            collections["verifier_registry"]["items"] = ver_registry
            print(f"✅ Verifier Registry: {len(ver_registry)} verifiers")
        except Exception as e:
            print(f"⚠️  Verifier Registry: {e}")
        
        # Convert sets to lists
        for collection, info in collections.items():
            for key, value in info.items():
                if isinstance(value, set):
                    info[key] = list(value)
        
        return collections
    
    def _explore_performance_metrics(self) -> Dict[str, Any]:
        """Explore performance and analytics"""
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 40)
        
        metrics = {
            "system_performance": {},
            "learning_progress": {},
            "cache_performance": {},
            "error_patterns": {}
        }
        
        try:
            # System performance summary
            perf_summary = self.dm.get_system_performance_summary(hours=24)
            metrics["system_performance"] = perf_summary
            print(f"✅ System Performance: {perf_summary['action_performance']['total_actions']} actions in 24h")
        except Exception as e:
            print(f"⚠️  System Performance: {e}")
        
        try:
            # Learning progress
            learning_progress = self.dm.get_learning_progress(sessions=10)
            metrics["learning_progress"] = learning_progress
            print(f"✅ Learning Progress: {learning_progress.get('sessions_analyzed', 0)} sessions analyzed")
        except Exception as e:
            print(f"⚠️  Learning Progress: {e}")
        
        try:
            # Cache performance
            cache_stats = self.dm.get_cache_stats()
            metrics["cache_performance"] = cache_stats
            print(f"✅ Cache Performance: {cache_stats}")
        except Exception as e:
            print(f"⚠️  Cache Performance: {e}")
        
        try:
            # Error patterns
            from dspy_agent.db.enhanced_storage import get_error_patterns
            error_patterns = get_error_patterns(hours=24)
            metrics["error_patterns"] = error_patterns
            print(f"✅ Error Patterns: {len(error_patterns)} error sources")
        except Exception as e:
            print(f"⚠️  Error Patterns: {e}")
        
        return metrics
    
    def _explore_system_health(self) -> Dict[str, Any]:
        """Explore system health and status"""
        print("\n🏥 SYSTEM HEALTH")
        print("-" * 40)
        
        health = {
            "current_health": {},
            "current_context": {},
            "storage_status": {},
            "environment_info": {}
        }
        
        try:
            # Current system health
            current_health = self.dm.get_system_health()
            health["current_health"] = current_health
            print(f"✅ System Health: {current_health}")
        except Exception as e:
            print(f"⚠️  System Health: {e}")
        
        try:
            # Current context
            current_context = self.dm.get_current_context()
            if current_context:
                health["current_context"] = {
                    "agent_state": current_context.agent_state.value,
                    "current_task": current_context.current_task,
                    "workspace_path": current_context.workspace_path,
                    "active_files_count": len(current_context.active_files),
                    "environment": current_context.environment.value
                }
                print(f"✅ Current Context: {current_context.agent_state.value}")
            else:
                print("ℹ️  Current Context: No active context")
        except Exception as e:
            print(f"⚠️  Current Context: {e}")
        
        try:
            # Storage status
            health["storage_status"] = {
                "namespace": self.namespace,
                "storage_type": type(self.storage).__name__,
                "cache_enabled": hasattr(self.dm, 'cache')
            }
            print(f"✅ Storage Status: {type(self.storage).__name__}")
        except Exception as e:
            print(f"⚠️  Storage Status: {e}")
        
        try:
            # Environment info
            health["environment_info"] = {
                "available_environments": [env.value for env in Environment],
                "available_action_types": [action.value for action in ActionType],
                "available_agent_states": [state.value for state in AgentState]
            }
            print(f"✅ Environment Info: {len(Environment)} environments, {len(ActionType)} action types")
        except Exception as e:
            print(f"⚠️  Environment Info: {e}")
        
        return health
    
    def print_summary(self, overview: Dict[str, Any]):
        """Print a summary of the data overview"""
        print("\n" + "=" * 60)
        print("📋 REDDB DATA LANDSCAPE SUMMARY")
        print("=" * 60)
        
        data_types = overview["data_types"]
        streams = overview["streams"]
        collections = overview["collections"]
        
        # Count total entities
        total_entities = 0
        entity_breakdown = []
        
        # Data types
        for data_type, info in data_types.items():
            count = info.get("count", 0)
            if count > 0:
                total_entities += count
                entity_breakdown.append(f"{count} {data_type.replace('_', ' ')}")
        
        # Streams
        total_stream_entries = sum(stream["count"] for stream in streams.values())
        if total_stream_entries > 0:
            entity_breakdown.append(f"{total_stream_entries} stream entries")
        
        # Collections
        total_collections = sum(collection["count"] for collection in collections.values())
        if total_collections > 0:
            entity_breakdown.append(f"{total_collections} registry items")
        
        print(f"\n🎯 TOTAL DATA ENTITIES: {total_entities + total_stream_entries + total_collections}")
        print(f"📊 BREAKDOWN:")
        for item in entity_breakdown:
            print(f"   • {item}")
        
        # Data type summary
        print(f"\n📈 DATA TYPE SUMMARY:")
        print(f"   • {data_types['signatures']['count']} DSPy Signatures ({data_types['signatures']['active']} active)")
        print(f"   • {data_types['verifiers']['count']} Code Verifiers")
        print(f"   • {data_types['actions']['count']} Agent Actions")
        print(f"   • {data_types['logs']['count']} Log Entries")
        print(f"   • {data_types['patches']['count']} Code Patches")
        print(f"   • {data_types['retrieval_events']['count']} Retrieval Events")
        print(f"   • {data_types['training_sessions']['count']} Training Sessions")
        
        # Stream summary
        active_streams = sum(1 for stream in streams.values() if stream["count"] > 0)
        print(f"\n🌊 STREAM SUMMARY:")
        print(f"   • {active_streams} active streams")
        print(f"   • {total_stream_entries} total stream entries")
        
        # Collection summary
        active_collections = sum(1 for collection in collections.values() if collection["count"] > 0)
        print(f"\n📚 COLLECTION SUMMARY:")
        print(f"   • {active_collections} active collections")
        print(f"   • {total_collections} total registry items")
        
        print(f"\n⏰ Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overview['timestamp']))}")


def main():
    """Main function to run the data overview"""
    try:
        explorer = RedDBDataExplorer()
        overview = explorer.get_data_overview()
        explorer.print_summary(overview)
        
        # Save overview to file
        import json
        output_file = Path(__file__).parent / "data_overview.json"
        with open(output_file, 'w') as f:
            json.dump(overview, f, indent=2, default=str)
        
        print(f"\n💾 Full overview saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error running data overview: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
