#!/usr/bin/env python3
"""
RedDB Streaming Data Analysis Script

This script analyzes the streaming data landscape in RedDB, focusing on
real-time data ingestion, vectorization, and RL training data.

Usage:
    python queries/streaming_data_analysis.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dspy_agent.db import get_enhanced_data_manager, Environment, ActionType, AgentState
from dspy_agent.agentic import get_retrieval_statistics, query_retrieval_events


class StreamingDataAnalyzer:
    """Analyzer for streaming data in RedDB"""
    
    def __init__(self):
        self.dm = get_enhanced_data_manager()
        self.storage = self.dm.storage
        self.namespace = "dspy_agent"
        
    def analyze_streaming_landscape(self) -> Dict[str, Any]:
        """Analyze the streaming data landscape"""
        print("🌊 Analyzing RedDB Streaming Data Landscape...")
        print("=" * 60)
        
        analysis = {
            "timestamp": time.time(),
            "streaming_sources": {},
            "data_ingestion_rates": {},
            "vectorization_status": {},
            "rl_training_data": {},
            "real_time_metrics": {},
            "data_flow_analysis": {}
        }
        
        # 1. Analyze streaming sources
        analysis["streaming_sources"] = self._analyze_streaming_sources()
        
        # 2. Analyze data ingestion rates
        analysis["data_ingestion_rates"] = self._analyze_ingestion_rates()
        
        # 3. Analyze vectorization status
        analysis["vectorization_status"] = self._analyze_vectorization()
        
        # 4. Analyze RL training data
        analysis["rl_training_data"] = self._analyze_rl_training_data()
        
        # 5. Real-time metrics
        analysis["real_time_metrics"] = self._analyze_real_time_metrics()
        
        # 6. Data flow analysis
        analysis["data_flow_analysis"] = self._analyze_data_flow()
        
        return analysis
    
    def _analyze_streaming_sources(self) -> Dict[str, Any]:
        """Analyze different streaming data sources"""
        print("\n📡 STREAMING SOURCES")
        print("-" * 40)
        
        sources = {
            "code_analysis_stream": {"count": 0, "latest": None, "rate_per_hour": 0},
            "action_stream": {"count": 0, "latest": None, "rate_per_hour": 0},
            "log_stream": {"count": 0, "latest": None, "rate_per_hour": 0},
            "retrieval_stream": {"count": 0, "latest": None, "rate_per_hour": 0},
            "training_stream": {"count": 0, "latest": None, "rate_per_hour": 0},
            "patch_stream": {"count": 0, "latest": None, "rate_per_hour": 0},
            "context_stream": {"count": 0, "latest": None, "rate_per_hour": 0},
            "health_stream": {"count": 0, "latest": None, "rate_per_hour": 0}
        }
        
        # Map stream names to their corresponding data
        stream_mapping = {
            "code_analysis_stream": "signature_metrics",
            "action_stream": "rl_actions", 
            "log_stream": "system_logs",
            "retrieval_stream": "retrieval_events",
            "training_stream": "training_history",
            "patch_stream": "patch_history",
            "context_stream": "context_history",
            "health_stream": "health_history"
        }
        
        for source_name, stream_name in stream_mapping.items():
            try:
                count = 0
                timestamps = []
                latest = None
                
                # Read from stream
                for offset, data in self.storage.read(stream_name, count=1000):
                    count += 1
                    if latest is None:
                        latest = data
                    
                    timestamp = data.get('timestamp', 0)
                    if timestamp > 0:
                        timestamps.append(timestamp)
                
                sources[source_name]["count"] = count
                sources[source_name]["latest"] = latest
                
                # Calculate rate per hour
                if timestamps and len(timestamps) > 1:
                    timestamps.sort()
                    time_span = timestamps[-1] - timestamps[0]
                    if time_span > 0:
                        hours = time_span / 3600
                        sources[source_name]["rate_per_hour"] = count / hours
                
                print(f"✅ {source_name}: {count} entries")
                if sources[source_name]["rate_per_hour"] > 0:
                    print(f"   Rate: {sources[source_name]['rate_per_hour']:.1f} entries/hour")
                
            except Exception as e:
                print(f"⚠️  {source_name}: {e}")
        
        return sources
    
    def _analyze_ingestion_rates(self) -> Dict[str, Any]:
        """Analyze data ingestion rates over time"""
        print("\n📊 DATA INGESTION RATES")
        print("-" * 40)
        
        rates = {
            "hourly_rates": {},
            "daily_rates": {},
            "peak_hours": {},
            "ingestion_trends": {}
        }
        
        # Analyze hourly rates for different data types
        current_time = time.time()
        hours_24_ago = current_time - (24 * 3600)
        
        # Get recent data and analyze by hour
        hourly_counts = defaultdict(lambda: defaultdict(int))
        
        try:
            # Analyze actions
            actions = self.dm.get_recent_actions(limit=1000)
            for action in actions:
                if action.timestamp >= hours_24_ago:
                    hour = int(action.timestamp // 3600) * 3600
                    hourly_counts["actions"][hour] += 1
            
            # Analyze logs
            logs = self.dm.get_recent_logs(limit=1000)
            for log in logs:
                if log.timestamp >= hours_24_ago:
                    hour = int(log.timestamp // 3600) * 3600
                    hourly_counts["logs"][hour] += 1
            
            # Analyze retrieval events
            retrieval_events = self.dm.get_recent_retrieval_events(limit=1000)
            for event in retrieval_events:
                if event.timestamp >= hours_24_ago:
                    hour = int(event.timestamp // 3600) * 3600
                    hourly_counts["retrieval"][hour] += 1
            
            # Calculate rates
            for data_type, hour_counts in hourly_counts.items():
                if hour_counts:
                    total_count = sum(hour_counts.values())
                    avg_per_hour = total_count / 24
                    peak_hour = max(hour_counts.items(), key=lambda x: x[1])
                    
                    rates["hourly_rates"][data_type] = {
                        "total_24h": total_count,
                        "avg_per_hour": avg_per_hour,
                        "peak_hour": peak_hour[0],
                        "peak_count": peak_hour[1]
                    }
                    
                    print(f"✅ {data_type}: {total_count} total, {avg_per_hour:.1f}/hour avg")
                    print(f"   Peak: {peak_hour[1]} at {datetime.fromtimestamp(peak_hour[0]).strftime('%H:%M')}")
            
        except Exception as e:
            print(f"⚠️  Ingestion rates: {e}")
        
        return rates
    
    def _analyze_vectorization(self) -> Dict[str, Any]:
        """Analyze vectorization status and effectiveness"""
        print("\n🔢 VECTORIZATION STATUS")
        print("-" * 40)
        
        vectorization = {
            "embedding_sources": {},
            "vector_dimensions": {},
            "similarity_metrics": {},
            "vectorization_effectiveness": {}
        }
        
        try:
            # Analyze retrieval events for vectorization effectiveness
            retrieval_events = self.dm.get_recent_retrieval_events(limit=500)
            
            if retrieval_events:
                total_queries = len(retrieval_events)
                total_hits = sum(len(event.hits) for event in retrieval_events)
                avg_hits_per_query = total_hits / total_queries
                
                # Analyze hit scores
                all_scores = []
                for event in retrieval_events:
                    for hit in event.hits:
                        score = hit.get('score', 0)
                        all_scores.append(score)
                
                if all_scores:
                    avg_score = sum(all_scores) / len(all_scores)
                    high_score_hits = sum(1 for score in all_scores if score > 0.8)
                    vectorization["vectorization_effectiveness"] = {
                        "total_queries": total_queries,
                        "total_hits": total_hits,
                        "avg_hits_per_query": avg_hits_per_query,
                        "avg_score": avg_score,
                        "high_score_ratio": high_score_hits / len(all_scores),
                        "vectorization_quality": "high" if avg_score > 0.7 else "medium" if avg_score > 0.5 else "low"
                    }
                    
                    print(f"✅ Vectorization Effectiveness:")
                    print(f"   Queries: {total_queries}, Hits: {total_hits}")
                    print(f"   Avg hits/query: {avg_hits_per_query:.1f}")
                    print(f"   Avg score: {avg_score:.3f}")
                    print(f"   Quality: {vectorization['vectorization_effectiveness']['vectorization_quality']}")
            
        except Exception as e:
            print(f"⚠️  Vectorization: {e}")
        
        return vectorization
    
    def _analyze_rl_training_data(self) -> Dict[str, Any]:
        """Analyze RL training data quality and trends"""
        print("\n🤖 RL TRAINING DATA")
        print("-" * 40)
        
        rl_data = {
            "action_quality": {},
            "reward_distribution": {},
            "learning_trends": {},
            "training_effectiveness": {}
        }
        
        try:
            # Analyze action quality
            actions = self.dm.get_recent_actions(limit=1000)
            
            if actions:
                rewards = [a.reward for a in actions]
                confidences = [a.confidence for a in actions]
                execution_times = [a.execution_time for a in actions]
                
                # Action type distribution
                action_types = Counter(a.action_type.value for a in actions)
                
                rl_data["action_quality"] = {
                    "total_actions": len(actions),
                    "avg_reward": sum(rewards) / len(rewards),
                    "avg_confidence": sum(confidences) / len(confidences),
                    "avg_execution_time": sum(execution_times) / len(execution_times),
                    "action_type_distribution": dict(action_types),
                    "high_reward_actions": sum(1 for r in rewards if r > 0.8),
                    "high_confidence_actions": sum(1 for c in confidences if c > 0.8)
                }
                
                print(f"✅ Action Quality:")
                print(f"   Total actions: {len(actions)}")
                print(f"   Avg reward: {rl_data['action_quality']['avg_reward']:.3f}")
                print(f"   Avg confidence: {rl_data['action_quality']['avg_confidence']:.3f}")
                print(f"   High reward actions: {rl_data['action_quality']['high_reward_actions']}")
            
            # Analyze training sessions
            training_history = self.dm.get_training_history(limit=50)
            
            if training_history:
                training_accuracies = [t.training_accuracy for t in training_history]
                validation_accuracies = [t.validation_accuracy for t in training_history]
                losses = [t.loss for t in training_history]
                
                rl_data["learning_trends"] = {
                    "sessions_analyzed": len(training_history),
                    "current_training_accuracy": training_accuracies[-1] if training_accuracies else 0,
                    "current_validation_accuracy": validation_accuracies[-1] if validation_accuracies else 0,
                    "current_loss": losses[-1] if losses else 0,
                    "accuracy_trend": "improving" if len(training_accuracies) > 1 and training_accuracies[-1] > training_accuracies[0] else "declining",
                    "loss_trend": "improving" if len(losses) > 1 and losses[-1] < losses[0] else "declining"
                }
                
                print(f"✅ Learning Trends:")
                print(f"   Sessions: {len(training_history)}")
                print(f"   Current accuracy: {rl_data['learning_trends']['current_training_accuracy']:.3f}")
                print(f"   Trend: {rl_data['learning_trends']['accuracy_trend']}")
            
        except Exception as e:
            print(f"⚠️  RL Training Data: {e}")
        
        return rl_data
    
    def _analyze_real_time_metrics(self) -> Dict[str, Any]:
        """Analyze real-time metrics and system performance"""
        print("\n⚡ REAL-TIME METRICS")
        print("-" * 40)
        
        real_time = {
            "current_throughput": {},
            "system_latency": {},
            "data_freshness": {},
            "processing_efficiency": {}
        }
        
        try:
            # Analyze current throughput (last hour)
            current_time = time.time()
            hour_ago = current_time - 3600
            
            # Count recent entries
            recent_actions = [a for a in self.dm.get_recent_actions(limit=1000) if a.timestamp >= hour_ago]
            recent_logs = [l for l in self.dm.get_recent_logs(limit=1000) if l.timestamp >= hour_ago]
            recent_retrievals = [r for r in self.dm.get_recent_retrieval_events(limit=1000) if r.timestamp >= hour_ago]
            
            real_time["current_throughput"] = {
                "actions_per_hour": len(recent_actions),
                "logs_per_hour": len(recent_logs),
                "retrievals_per_hour": len(recent_retrievals),
                "total_throughput": len(recent_actions) + len(recent_logs) + len(recent_retrievals)
            }
            
            print(f"✅ Current Throughput (last hour):")
            print(f"   Actions: {len(recent_actions)}")
            print(f"   Logs: {len(recent_logs)}")
            print(f"   Retrievals: {len(recent_retrievals)}")
            print(f"   Total: {real_time['current_throughput']['total_throughput']}")
            
            # Analyze data freshness
            if recent_actions:
                latest_action_time = max(a.timestamp for a in recent_actions)
                data_age = current_time - latest_action_time
                real_time["data_freshness"] = {
                    "latest_action_age_seconds": data_age,
                    "data_freshness_status": "fresh" if data_age < 300 else "stale" if data_age < 3600 else "very_stale"
                }
                print(f"✅ Data Freshness: {data_age:.1f}s ago ({real_time['data_freshness']['data_freshness_status']})")
            
        except Exception as e:
            print(f"⚠️  Real-time Metrics: {e}")
        
        return real_time
    
    def _analyze_data_flow(self) -> Dict[str, Any]:
        """Analyze data flow patterns and bottlenecks"""
        print("\n🔄 DATA FLOW ANALYSIS")
        print("-" * 40)
        
        flow_analysis = {
            "data_pipeline_stages": {},
            "bottlenecks": {},
            "flow_efficiency": {},
            "streaming_health": {}
        }
        
        try:
            # Analyze data flow through different stages
            stages = {
                "ingestion": 0,
                "processing": 0,
                "vectorization": 0,
                "storage": 0,
                "retrieval": 0
            }
            
            # Estimate based on available data
            actions = self.dm.get_recent_actions(limit=1000)
            logs = self.dm.get_recent_logs(limit=1000)
            retrieval_events = self.dm.get_recent_retrieval_events(limit=1000)
            
            stages["ingestion"] = len(actions) + len(logs)
            stages["processing"] = len(actions)  # Actions represent processed data
            stages["vectorization"] = len(retrieval_events)  # Retrieval events represent vectorized queries
            stages["storage"] = len(actions) + len(logs) + len(retrieval_events)
            stages["retrieval"] = len(retrieval_events)
            
            flow_analysis["data_pipeline_stages"] = stages
            
            # Identify potential bottlenecks
            min_stage = min(stages.values())
            bottlenecks = [stage for stage, count in stages.items() if count < min_stage * 1.5]
            
            flow_analysis["bottlenecks"] = {
                "identified_bottlenecks": bottlenecks,
                "pipeline_balance": "balanced" if len(bottlenecks) == 0 else "imbalanced"
            }
            
            print(f"✅ Data Pipeline Stages:")
            for stage, count in stages.items():
                print(f"   {stage}: {count}")
            
            if bottlenecks:
                print(f"⚠️  Potential bottlenecks: {bottlenecks}")
            else:
                print(f"✅ Pipeline appears balanced")
            
        except Exception as e:
            print(f"⚠️  Data Flow Analysis: {e}")
        
        return flow_analysis
    
    def print_streaming_summary(self, analysis: Dict[str, Any]):
        """Print a summary of the streaming analysis"""
        print("\n" + "=" * 60)
        print("🌊 STREAMING DATA LANDSCAPE SUMMARY")
        print("=" * 60)
        
        sources = analysis["streaming_sources"]
        ingestion = analysis["data_ingestion_rates"]
        rl_data = analysis["rl_training_data"]
        real_time = analysis["real_time_metrics"]
        
        # Calculate totals
        total_stream_entries = sum(source["count"] for source in sources.values())
        total_ingestion_rate = sum(rate.get("total_24h", 0) for rate in ingestion.get("hourly_rates", {}).values())
        
        print(f"\n📊 STREAMING OVERVIEW:")
        print(f"   • {total_stream_entries} total stream entries")
        print(f"   • {total_ingestion_rate} data points ingested in 24h")
        print(f"   • {len([s for s in sources.values() if s['count'] > 0])} active streams")
        
        # Top streaming sources
        active_sources = [(name, info) for name, info in sources.items() if info["count"] > 0]
        active_sources.sort(key=lambda x: x[1]["count"], reverse=True)
        
        print(f"\n🔥 TOP STREAMING SOURCES:")
        for name, info in active_sources[:5]:
            print(f"   • {name}: {info['count']} entries")
        
        # RL Training Status
        if rl_data.get("action_quality"):
            action_quality = rl_data["action_quality"]
            print(f"\n🤖 RL TRAINING STATUS:")
            print(f"   • {action_quality['total_actions']} training actions")
            print(f"   • {action_quality['avg_reward']:.3f} average reward")
            print(f"   • {action_quality['high_reward_actions']} high-reward actions")
        
        # Real-time Performance
        if real_time.get("current_throughput"):
            throughput = real_time["current_throughput"]
            print(f"\n⚡ REAL-TIME PERFORMANCE:")
            print(f"   • {throughput['total_throughput']} events/hour current throughput")
            print(f"   • {throughput['actions_per_hour']} actions/hour")
            print(f"   • {throughput['retrievals_per_hour']} retrievals/hour")
        
        # Data Quality
        vectorization = analysis.get("vectorization_status", {})
        if vectorization.get("vectorization_effectiveness"):
            effectiveness = vectorization["vectorization_effectiveness"]
            print(f"\n🔢 DATA QUALITY:")
            print(f"   • {effectiveness['avg_score']:.3f} average vectorization score")
            print(f"   • {effectiveness['vectorization_quality']} vectorization quality")
            print(f"   • {effectiveness['avg_hits_per_query']:.1f} avg hits per query")
        
        print(f"\n⏰ Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(analysis['timestamp']))}")


def main():
    """Main function to run the streaming analysis"""
    try:
        analyzer = StreamingDataAnalyzer()
        analysis = analyzer.analyze_streaming_landscape()
        analyzer.print_streaming_summary(analysis)
        
        # Save analysis to file
        import json
        output_file = Path(__file__).parent / "streaming_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Full analysis saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error running streaming analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
