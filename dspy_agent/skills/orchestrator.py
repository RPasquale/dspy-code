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

from ..db import (
    get_enhanced_data_manager, ActionRecord, LogEntry, 
    Environment, ActionType, AgentState, 
    create_action_record, create_log_entry
)


TOOLS = [
    "context", "plan", "grep", "extract", "tree", "ls",
    "codectx", "index", "esearch", "emb_index", "emb_search", "knowledge", "vretr", "intel",
    "edit", "patch", "run_tests", "lint", "build",
    "open", "watch", "sg", "diff", "git_status", "git_add", "git_commit",
    # Data tools (local RedDB-backed)
    "db_ingest", "db_query", "db_multi"
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
    """Persistent memory for agent sessions with expert-level learning"""
    
    def __init__(self, workspace: Path, max_history: int = 50):
        self.workspace = workspace
        self.memory_file = workspace / '.dspy_session_memory.json'
        self.max_history = max_history
        self.current_chain: List[ToolResult] = []
        self.chain_history: deque = deque(maxlen=max_history)
        self.tool_cache: Dict[str, ToolResult] = {}
        self.query_cache: Dict[str, str] = {}
        
        # Expert-level learning components
        self.expert_patterns: Dict[str, List[Dict]] = {}  # Learned patterns by context
        self.tool_effectiveness: Dict[str, float] = {}    # Tool success rates
        self.context_insights: Dict[str, str] = {}        # Codebase insights
        self.prompt_optimizations: Dict[str, str] = {}    # Optimized prompts by context
        self.action_policies: Dict[str, List[str]] = {}   # Reliable action sequences
        
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
                    
                    # Load expert-level learning data
                    self.expert_patterns = data.get('expert_patterns', {})
                    self.tool_effectiveness = data.get('tool_effectiveness', {})
                    self.context_insights = data.get('context_insights', {})
                    self.prompt_optimizations = data.get('prompt_optimizations', {})
                    self.action_policies = data.get('action_policies', {})
            except Exception:
                pass  # Start fresh if corrupted
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            data = {
                'chain_history': list(self.chain_history),
                'tool_cache': self.tool_cache,
                'query_cache': self.query_cache,
                'expert_patterns': self.expert_patterns,
                'tool_effectiveness': self.tool_effectiveness,
                'context_insights': self.context_insights,
                'prompt_optimizations': self.prompt_optimizations,
                'action_policies': self.action_policies,
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
    
    def learn_expert_pattern(self, context: str, tool_sequence: List[str], success: bool, reward: float):
        """Learn expert patterns from successful tool sequences"""
        if context not in self.expert_patterns:
            self.expert_patterns[context] = []
        
        pattern = {
            "tool_sequence": tool_sequence,
            "success": success,
            "reward": reward,
            "timestamp": time.time(),
            "frequency": 1
        }
        
        # Check if similar pattern exists
        for existing in self.expert_patterns[context]:
            if existing["tool_sequence"] == tool_sequence:
                existing["frequency"] += 1
                existing["reward"] = (existing["reward"] + reward) / 2
                existing["success"] = success if success else existing["success"]
                return
        
        self.expert_patterns[context].append(pattern)
        self._save_memory()
    
    def update_tool_effectiveness(self, tool: str, success: bool):
        """Update tool effectiveness based on usage"""
        if tool not in self.tool_effectiveness:
            self.tool_effectiveness[tool] = 0.5  # Start with neutral
        
        # Simple moving average
        current = self.tool_effectiveness[tool]
        new_value = 1.0 if success else 0.0
        self.tool_effectiveness[tool] = (current * 0.9) + (new_value * 0.1)
        self._save_memory()
    
    def get_best_tool_sequence(self, context: str) -> List[str]:
        """Get the best tool sequence for a given context"""
        if context not in self.expert_patterns:
            return []
        
        # Find the pattern with highest reward and frequency
        best_pattern = None
        best_score = 0.0
        
        for pattern in self.expert_patterns[context]:
            if pattern["success"]:
                score = pattern["reward"] * pattern["frequency"]
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
        
        return best_pattern["tool_sequence"] if best_pattern else []
    
    def get_most_effective_tools(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get the most effective tools based on success rate"""
        return sorted(self.tool_effectiveness.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def add_context_insight(self, context: str, insight: str):
        """Add codebase insight for a context"""
        self.context_insights[context] = insight
        self._save_memory()
    
    def get_context_insight(self, context: str) -> Optional[str]:
        """Get codebase insight for a context"""
        return self.context_insights.get(context)
    
    def optimize_prompt(self, context: str, original_prompt: str, success: bool) -> str:
        """Optimize prompt based on success/failure"""
        if context not in self.prompt_optimizations:
            self.prompt_optimizations[context] = original_prompt
        
        if not success:
            # Enhance prompt for better performance
            enhanced = f"Focus on accuracy and thoroughness. {original_prompt}"
            self.prompt_optimizations[context] = enhanced
        else:
            # Refine successful prompt
            refined = f"Execute efficiently. {original_prompt}"
            self.prompt_optimizations[context] = refined
        
        self._save_memory()
        return self.prompt_optimizations[context]
    
    def get_optimized_prompt(self, context: str, fallback: str) -> str:
        """Get optimized prompt for context"""
        return self.prompt_optimizations.get(context, fallback)
    
    def learn_action_policy(self, context: str, actions: List[str], success: bool):
        """Learn reliable action policies"""
        if context not in self.action_policies:
            self.action_policies[context] = []
        
        if success and actions not in self.action_policies[context]:
            self.action_policies[context].append(actions)
            self._save_memory()
    
    def get_reliable_actions(self, context: str) -> List[str]:
        """Get reliable actions for a context"""
        policies = self.action_policies.get(context, [])
        if policies:
            # Return the most frequently successful policy
            return policies[0]  # Simplified - could be more sophisticated
        return []
    
    def _get_cache_key(self, tool: str, args: Dict[str, Any]) -> str:
        """Generate optimized cache key for tool+args combination"""
        # Optimize key generation by excluding volatile fields
        filtered_args = {k: v for k, v in args.items() 
                        if k not in ['timestamp', 'session_id', 'request_id']}
        key_data = f"{tool}:{json.dumps(filtered_args, sort_keys=True)}"
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
        
        # RedDB integration
        self.data_manager = get_enhanced_data_manager()
        self.session_id = f"orchestrator_{int(time.time())}"
        
        # Initialize memory if workspace provided
        if workspace:
            self.memory = SessionMemory(workspace)
            
        # Optional signature tagging for analytics
        self.signature_name: Optional[str] = None

        # Log orchestrator initialization
        init_log = create_log_entry(
            level="INFO",
            source="skills.orchestrator",
            message=f"Orchestrator initialized with session: {self.session_id}",
            context={
                "session_id": self.session_id,
                "use_cot": use_cot,
                "workspace": str(workspace) if workspace else None
            },
            environment=Environment.DEVELOPMENT
        )
        self.data_manager.log(init_log)

    def set_signature_name(self, name: Optional[str]) -> None:
        """Tag subsequent actions with a signature name for analytics (optional)."""
        self.signature_name = name if (isinstance(name, str) and name.strip()) else None

    def __call__(self, query: str, state: str, memory: Optional[SessionMemory] = None):
        """Enhanced forward with memory, caching, and performance tracking"""
        overall_start_time = time.time()
        
        # Use provided memory or instance memory
        active_memory = memory or self.memory
        
        # Record initial state for action tracking
        initial_state = {
            "query": query,
            "state": state,
            "has_memory": active_memory is not None,
            "cache_size": len(self.prediction_cache)
        }
        if self.signature_name:
            initial_state["signature_name"] = self.signature_name
        
        # Add memory context to state
        enhanced_state = state
        if active_memory:
            context = active_memory.get_context_for_query(query)
            if context:
                enhanced_state = f"{state} | Memory context: {context}"
                initial_state["memory_context_added"] = True
        # Learned policy prompt (if present)
        if self.workspace:
            try:
                pfile = self.workspace / '.dspy_policy_prompt.txt'
                if pfile.exists():
                    text = pfile.read_text()[:400].replace('\n', '; ')
                    enhanced_state = f"{enhanced_state} | Learned policy: {text}"
                    initial_state["policy_learned"] = True
            except Exception:
                pass
        # Policy hints
        if self.workspace:
            try:
                from ..policy import Policy, apply_policy_to_state
                pol = Policy.load(self.workspace)
                if pol is not None:
                    enhanced_state = apply_policy_to_state(query, enhanced_state, pol)
                    initial_state["policy_applied"] = True
            except Exception:
                pass
        
        # Check cache first
        cache_key = self._get_cache_key(query, enhanced_state)
        cache_hit = False
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            # Only use cache if less than 2 minutes old
            if time.time() - cached_result.get('timestamp', 0) < 120:
                cache_hit = True
                execution_time = time.time() - overall_start_time
                
                # Log cache hit
                cache_log = create_log_entry(
                    level="DEBUG",
                    source="skills.orchestrator",
                    message=f"Cache hit for query: {query[:50]}...",
                    context={
                        "session_id": self.session_id,
                        "cache_key": cache_key,
                        "execution_time": execution_time,
                        "tool": cached_result['tool']
                    },
                    environment=Environment.DEVELOPMENT
                )
                self.data_manager.log(cache_log)
                
                # Record cache hit action
                cache_action = create_action_record(
                    action_type=ActionType.TOOL_SELECTION,
                    state_before=initial_state,
                    state_after={"tool": cached_result['tool'], "cached": True, **({"signature_name": self.signature_name} if self.signature_name else {})},
                    parameters={"query": query, "cache_hit": True, **({"signature_name": self.signature_name} if self.signature_name else {})},
                    result={"tool": cached_result['tool'], "args_json": cached_result['args_json'], **({"signature_name": self.signature_name} if self.signature_name else {})},
                    reward=0.9,  # High reward for cache hits
                    confidence=0.95,
                    execution_time=execution_time,
                    environment=Environment.DEVELOPMENT
                )
                self.data_manager.record_action(cache_action)
                
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
        prediction_start_time = time.time()
        success = True
        error_msg = None
        
        try:
            pred = self.predict(query=query, state=enhanced_state)
            prediction_time = time.time() - prediction_start_time
            
            # Cache successful predictions
            if prediction_time < 3.0:  # Only cache fast predictions
                self.prediction_cache[cache_key] = {
                    'tool': pred.tool,
                    'args_json': pred.args_json,
                    'rationale': getattr(pred, 'rationale', ''),
                    'timestamp': time.time()
                }
            
        except Exception as e:
            success = False
            error_msg = str(e)
            prediction_time = time.time() - prediction_start_time
            
            # Log prediction error
            error_log = create_log_entry(
                level="ERROR",
                source="skills.orchestrator",
                message=f"Prediction failed for query: {query[:50]}...",
                context={
                    "session_id": self.session_id,
                    "error": error_msg,
                    "prediction_time": prediction_time
                },
                environment=Environment.DEVELOPMENT
            )
            self.data_manager.log(error_log)
            
            # Create fallback prediction
            pred = SimpleNamespace(
                tool="plan",
                args_json="{}",
                rationale=f"Fallback to 'plan' due to prediction error: {e}",
                cached=False
            )

        # Validate tool selection (with policy enforcement)
        tool = (getattr(pred, "tool", None) or "").strip()
        if self.workspace:
            try:
                from ..policy import Policy, enforce_policy_on_tool
                pol = Policy.load(self.workspace)
                if pol is not None:
                    new_tool, note = enforce_policy_on_tool(query, tool, pol)
                    if new_tool != tool:
                        tool = new_tool
                        error_msg = (error_msg or "") + (f" | {note}" if note else "")
                        setattr(pred, 'tool', tool)
            except Exception:
                pass
        if tool not in TOOLS:
            success = False
            error_msg = f"Invalid tool '{tool}' not in TOOLS"
            
            # Log validation error
            validation_log = create_log_entry(
                level="ERROR",
                source="skills.orchestrator",
                message=f"Invalid tool selected: {tool}",
                context={
                    "session_id": self.session_id,
                    "tool": tool,
                    "valid_tools": TOOLS,
                    "query": query[:100]
                },
                environment=Environment.DEVELOPMENT
            )
            self.data_manager.log(validation_log)
            
            raise ValueError(f"Predicted tool '{tool}' is not in supported TOOLS: {', '.join(TOOLS)}")

        # Calculate final metrics
        total_execution_time = time.time() - overall_start_time
        
        # Determine reward based on performance and success
        reward = 0.8 if success else 0.3
        if total_execution_time < 0.5:
            reward += 0.1  # Bonus for fast execution
        if cache_hit:
            reward += 0.1  # Bonus for cache efficiency
        
        confidence = 0.9 if success else 0.5
        
        # Enhanced expert-level learning with adaptive intelligence
        if active_memory:
            # Update tool effectiveness with weighted learning
            active_memory.update_tool_effectiveness(tool, success)
            
            # Advanced pattern learning with context awareness
            if success and not cache_hit:
                context = self._extract_context_from_query(query)
                tool_sequence = [tool]  # Could be extended to track sequences
                
                # Adaptive reward calculation based on execution time and success
                base_reward = 0.8 if success else 0.3
                time_bonus = max(0, (2.0 - total_execution_time) * 0.1)  # Bonus for fast execution
                reward = min(1.0, base_reward + time_bonus)
                
                active_memory.learn_expert_pattern(context, tool_sequence, success, reward)
                
                # Learn action policy with confidence weighting
                actions = [tool]
                confidence_weight = confidence if hasattr(self, 'confidence') else 0.8
                active_memory.learn_action_policy(context, actions, success)
                
                # Intelligent prompt optimization
                original_prompt = f"Select the best tool for: {query}"
                optimized_prompt = active_memory.optimize_prompt(context, original_prompt, success)
                
                # Enhanced context insights with performance metrics
                if success:
                    insight = f"Tool '{tool}' works well for queries like: {query[:50]}... (execution_time: {total_execution_time:.2f}s, confidence: {confidence:.2f})"
                    active_memory.add_context_insight(context, insight)
                
                # Adaptive learning rate based on performance
                if total_execution_time < 0.5 and success:
                    # High performance - increase learning rate
                    active_memory.learning_rate = min(1.0, getattr(active_memory, 'learning_rate', 0.1) + 0.05)
                elif total_execution_time > 5.0 or not success:
                    # Low performance - decrease learning rate
                    active_memory.learning_rate = max(0.01, getattr(active_memory, 'learning_rate', 0.1) - 0.02)
        
        # Record the orchestration action
        final_state = {
            "tool": tool,
            "args_json": getattr(pred, 'args_json', '{}'),
            "rationale": getattr(pred, 'rationale', ''),
            "success": success,
            "cached": cache_hit
        }
        
        orchestration_action = create_action_record(
            action_type=ActionType.TOOL_SELECTION,
            state_before=initial_state,
            state_after=final_state,
            parameters={
                "query": query,
                "enhanced_state": enhanced_state,
                "cache_checked": True,
                "cache_hit": cache_hit,
                **({"signature_name": self.signature_name} if self.signature_name else {})
            },
            result={
                "tool": tool,
                "args_json": getattr(pred, 'args_json', '{}'),
                "success": success,
                "error": error_msg,
                **({"signature_name": self.signature_name} if self.signature_name else {})
            },
            reward=reward,
            confidence=confidence,
            execution_time=total_execution_time,
            environment=Environment.DEVELOPMENT
        )
        self.data_manager.record_action(orchestration_action)
        
        # Log successful orchestration
        if success:
            success_log = create_log_entry(
                level="INFO",
                source="skills.orchestrator",
                message=f"Successfully orchestrated tool '{tool}' for query: {query[:50]}...",
                context={
                    "session_id": self.session_id,
                    "tool": tool,
                    "execution_time": total_execution_time,
                    "reward": reward,
                    "confidence": confidence,
                    "cache_hit": cache_hit
                },
                environment=Environment.DEVELOPMENT
            )
            self.data_manager.log(success_log)

        return pred
    
    def _extract_context_from_query(self, query: str) -> str:
        """Extract context type from query for expert learning"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['test', 'testing', 'pytest', 'unittest']):
            return 'testing'
        elif any(word in query_lower for word in ['build', 'compile', 'make', 'install']):
            return 'building'
        elif any(word in query_lower for word in ['debug', 'error', 'fix', 'bug']):
            return 'debugging'
        elif any(word in query_lower for word in ['refactor', 'clean', 'improve', 'optimize']):
            return 'refactoring'
        elif any(word in query_lower for word in ['implement', 'add', 'create', 'feature']):
            return 'implementation'
        elif any(word in query_lower for word in ['search', 'find', 'grep', 'look']):
            return 'search'
        elif any(word in query_lower for word in ['analyze', 'understand', 'explain', 'review']):
            return 'analysis'
        else:
            return 'general'
    
    def _get_cache_key(self, query: str, state: str) -> str:
        """Generate cache key for prediction"""
        key_data = f"{query}:{state}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Enhanced cache cleanup with performance optimization"""
        current_time = time.time()
        keys_to_remove = []
        
        # Remove expired entries (older than 5 minutes)
        for key, value in self.prediction_cache.items():
            if current_time - value.get('timestamp', 0) > 300:  # 5 minutes
                keys_to_remove.append(key)
        
        # If cache is still too large, remove least recently used entries
        if len(self.prediction_cache) > 1000:  # Limit cache size
            # Sort by timestamp and remove oldest 20%
            sorted_items = sorted(
                self.prediction_cache.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            items_to_remove = len(sorted_items) // 5  # Remove 20%
            for key, _ in sorted_items[:items_to_remove]:
                if key not in keys_to_remove:
                    keys_to_remove.append(key)
        
        # Remove selected keys
        for key in keys_to_remove:
            del self.prediction_cache[key]
        
        # Log cache cleanup for monitoring
        if keys_to_remove:
            cleanup_log = create_log_entry(
                level="DEBUG",
                source="skills.orchestrator",
                message=f"Cache cleanup: removed {len(keys_to_remove)} entries, {len(self.prediction_cache)} remaining",
                context={
                    "session_id": self.session_id,
                    "cache_size_before": len(self.prediction_cache) + len(keys_to_remove),
                    "cache_size_after": len(self.prediction_cache),
                    "entries_removed": len(keys_to_remove)
                },
                environment=Environment.DEVELOPMENT
            )
            self.data_manager.log(cleanup_log)
    
    def get_memory(self) -> Optional[SessionMemory]:
        """Get the session memory instance"""
        return self.memory
    
    def create_memory(self, workspace: Path) -> SessionMemory:
        """Create a new memory instance"""
        self.memory = SessionMemory(workspace)
        return self.memory
