from __future__ import annotations

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque

import dspy
from types import SimpleNamespace


TOOLS = [
    "context", "plan", "grep", "extract", "tree", "ls",
    "codectx", "index", "esearch", "emb_index", "emb_search", "knowledge", "vretr", "intel",
    "open", "watch", "sg", "patch", "diff", "git_status",
    "git_add", "git_commit"
]


@dataclass
class ToolResult:
    """Structured result from a tool execution"""
    tool: str
    args: Dict[str, Any]
    result: str
    success: bool
    timestamp: float
    execution_time: float
    score: Optional[float] = None
    feedback: Optional[str] = None


@dataclass
class ChainSummary:
    """Summary of a tool chain execution"""
    query: str
    steps: List[ToolResult]
    total_time: float
    success: bool
    key_findings: List[str]
    next_suggestions: List[str]
    context_for_continuation: str


class SessionMemory:
    """Persistent memory for agent sessions"""
    
    def __init__(self, workspace: Path, max_history: int = 50):
        self.workspace = workspace
        self.memory_file = workspace / '.dspy_session_memory.json'
        self.max_history = max_history
        self.current_chain: List[ToolResult] = []
        self.chain_history: deque = deque(maxlen=max_history)
        self.tool_cache: Dict[str, ToolResult] = {}
        self.query_cache: Dict[str, str] = {}
        self._load_memory()
    
    def _load_memory(self):
        """Load persistent memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.chain_history = deque(data.get('chain_history', []), maxlen=self.max_history)
                    self.tool_cache = data.get('tool_cache', {})
                    self.query_cache = data.get('query_cache', {})
            except Exception:
                pass  # Start fresh if corrupted
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            data = {
                'chain_history': list(self.chain_history),
                'tool_cache': self.tool_cache,
                'query_cache': self.query_cache,
                'last_updated': time.time()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Don't fail if can't save
    
    def add_tool_result(self, result: ToolResult):
        """Add a tool result to current chain"""
        self.current_chain.append(result)
        # Cache tool results for reuse
        cache_key = self._get_cache_key(result.tool, result.args)
        self.tool_cache[cache_key] = result
    
    def finish_chain(self, query: str) -> ChainSummary:
        """Finish current chain and create summary"""
        if not self.current_chain:
            return ChainSummary(
                query=query,
                steps=[],
                total_time=0.0,
                success=False,
                key_findings=[],
                next_suggestions=[],
                context_for_continuation=""
            )
        
        total_time = sum(step.execution_time for step in self.current_chain)
        success = any(step.success for step in self.current_chain)
        
        # Extract key findings from successful steps
        key_findings = []
        for step in self.current_chain:
            if step.success and step.result:
                # Extract key insights (simplified)
                if len(step.result) > 100:
                    key_findings.append(f"{step.tool}: {step.result[:100]}...")
                else:
                    key_findings.append(f"{step.tool}: {step.result}")
        
        # Generate next suggestions based on what was done
        next_suggestions = self._generate_next_suggestions()
        
        # Create context for continuation
        context_for_continuation = self._build_continuation_context(query)
        
        summary = ChainSummary(
            query=query,
            steps=self.current_chain.copy(),
            total_time=total_time,
            success=success,
            key_findings=key_findings,
            next_suggestions=next_suggestions,
            context_for_continuation=context_for_continuation
        )
        
        # Add to history and save
        self.chain_history.append(asdict(summary))
        self.current_chain.clear()
        self._save_memory()
        
        return summary
    
    def _get_cache_key(self, tool: str, args: Dict[str, Any]) -> str:
        """Generate cache key for tool+args combination"""
        key_data = f"{tool}:{json.dumps(args, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, tool: str, args: Dict[str, Any]) -> Optional[ToolResult]:
        """Get cached result if available and recent"""
        cache_key = self._get_cache_key(tool, args)
        if cache_key in self.tool_cache:
            result = self.tool_cache[cache_key]
            # Only use cache if less than 5 minutes old
            if time.time() - result.timestamp < 300:
                return result
        return None
    
    def _generate_next_suggestions(self) -> List[str]:
        """Generate intelligent next step suggestions"""
        suggestions = []
        
        # Analyze what tools were used
        tools_used = [step.tool for step in self.current_chain]
        
        if 'codectx' in tools_used and 'esearch' not in tools_used:
            suggestions.append("Try semantic search with 'esearch' to find specific patterns")
        
        if 'grep' in tools_used and 'intel' not in tools_used:
            suggestions.append("Use 'intel' for deeper analysis of found patterns")
        
        if 'plan' in tools_used:
            suggestions.append("Execute the plan with specific tool commands")
        
        if not any(t in tools_used for t in ['index', 'emb-index']):
            suggestions.append("Build indexes for better search capabilities")
        
        # Default suggestions
        if not suggestions:
            suggestions = [
                "Try 'intel' for comprehensive analysis",
                "Use 'esearch' for semantic code search",
                "Run 'plan' to get structured approach"
            ]
        
        return suggestions
    
    def _build_continuation_context(self, query: str) -> str:
        """Build context string for continuation commands"""
        if not self.current_chain:
            return ""
        
        recent_tools = [step.tool for step in self.current_chain[-3:]]
        recent_results = [step.result[:100] for step in self.current_chain[-2:] if step.result]
        
        context = f"Last query: '{query}'. Recent tools: {', '.join(recent_tools)}. "
        if recent_results:
            context += f"Recent findings: {'; '.join(recent_results)}"
        
        return context
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for a new query"""
        # Check if this is a continuation command
        continuation_words = ['continue', 'more', 'next', 'also', 'and', 'then', 'further']
        if any(word in query.lower() for word in continuation_words):
            if self.chain_history:
                last_chain = self.chain_history[-1]
                return last_chain.get('context_for_continuation', '')
        
        # Check for similar queries in cache
        query_lower = query.lower()
        for cached_query, context in self.query_cache.items():
            if any(word in query_lower for word in cached_query.lower().split()):
                return context
        
        return ""


class OrchestrateToolSig(dspy.Signature):
    """Choose the best CLI tool and arguments for the user's intent.

    Tools: context, plan, grep, extract, tree, ls, codectx, index, esearch, emb_index, emb_search, knowledge, vretr, intel, open, watch, sg, patch, diff, git_status, git_add, git_commit
    Return JSON in args_json with the arguments for that tool.
    Keep choices safe and non-destructive unless explicitly requested by the user.
    """

    query: str = dspy.InputField(desc="User's natural-language request")
    state: str = dspy.InputField(desc="Short environment summary: workspace, logs, last extract, indexes available")

    tool: str = dspy.OutputField(desc=f"One of: {', '.join(TOOLS)} (choose conservatively)")
    args_json: str = dspy.OutputField(desc="JSON object of arguments; omit unknown fields")
    rationale: str = dspy.OutputField(desc="Brief reasoning for the tool choice")


class Orchestrator(dspy.Module):
    def __init__(self, use_cot: bool = True, workspace: Optional[Path] = None):
        super().__init__()
        # Use CoT to justify routing when available
        self.predict = dspy.ChainOfThought(OrchestrateToolSig) if use_cot else dspy.Predict(OrchestrateToolSig)
        
        # Performance and memory enhancements
        self.workspace = workspace
        self.memory: Optional[SessionMemory] = None
        self.prediction_cache: Dict[str, Any] = {}
        self.last_cache_cleanup = time.time()
        
        # Initialize memory if workspace provided
        if workspace:
            self.memory = SessionMemory(workspace)

    def __call__(self, query: str, state: str, memory: Optional[SessionMemory] = None):
        """Enhanced forward with memory, caching, and performance tracking"""
        # Use provided memory or instance memory
        active_memory = memory or self.memory
        
        # Add memory context to state
        if active_memory:
            context = active_memory.get_context_for_query(query)
            if context:
                state = f"{state} | Memory context: {context}"
        
        # Check cache first
        cache_key = self._get_cache_key(query, state)
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            # Only use cache if less than 2 minutes old
            if time.time() - cached_result.get('timestamp', 0) < 120:
                return SimpleNamespace(
                    tool=cached_result['tool'],
                    args_json=cached_result['args_json'],
                    rationale=f"Cached prediction: {cached_result.get('rationale', '')}",
                    cached=True
                )
        
        # Clean cache periodically
        if time.time() - self.last_cache_cleanup > 300:  # 5 minutes
            self._cleanup_cache()
            self.last_cache_cleanup = time.time()
        
        # Predict tool selection with error handling and validation
        start_time = time.time()
        try:
            pred = self.predict(query=query, state=state)
            execution_time = time.time() - start_time
            
            # Cache successful predictions
            if execution_time < 3.0:  # Only cache fast predictions
                self.prediction_cache[cache_key] = {
                    'tool': pred.tool,
                    'args_json': pred.args_json,
                    'rationale': getattr(pred, 'rationale', ''),
                    'timestamp': time.time()
                }
            
        except Exception as e:
            # Sensible default fallback when routing fails
            return SimpleNamespace(
                tool="plan",
                args_json="{}",
                rationale=f"Fallback to 'plan' due to prediction error: {e}",
                cached=False
            )

        tool = (getattr(pred, "tool", None) or "").strip()
        if tool not in TOOLS:
            raise ValueError(f"Predicted tool '{tool}' is not in supported TOOLS: {', '.join(TOOLS)}")

        return pred
    
    def _get_cache_key(self, query: str, state: str) -> str:
        """Generate cache key for prediction"""
        key_data = f"{query}:{state}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        keys_to_remove = []
        for key, value in self.prediction_cache.items():
            if current_time - value.get('timestamp', 0) > 300:  # 5 minutes
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.prediction_cache[key]
    
    def get_memory(self) -> Optional[SessionMemory]:
        """Get the session memory instance"""
        return self.memory
    
    def create_memory(self, workspace: Path) -> SessionMemory:
        """Create a new memory instance"""
        self.memory = SessionMemory(workspace)
        return self.memory
