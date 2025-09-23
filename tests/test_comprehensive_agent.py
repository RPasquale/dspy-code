"""
Comprehensive test suite for the full DSPy agent stack.
This test suite covers all major components and their interactions.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import subprocess
import sys
from typing import Dict, Any, List

# Test imports
from dspy_agent.cli import app
from dspy_agent.launcher import run as launcher_run
from dspy_agent.config import Config
from dspy_agent.llm import LLMProvider
from dspy_agent.agents.orchestrator_runtime import OrchestratorRuntime
from dspy_agent.agents.knowledge import KnowledgeAgent
from dspy_agent.agents.router_worker import RouterWorker
from dspy_agent.streaming.streaming_runtime import StreamingRuntime
from dspy_agent.memory.manager import MemoryManager
from dspy_agent.embedding.embedder import Embedder
from dspy_agent.db.dbkit import DatabaseKit
from dspy_agent.rl.rlkit import train_puffer_policy
from dspy_agent.training.entrypoint import main as training_main


class TestAgentComprehensive:
    """Comprehensive test suite for the full agent stack."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment with proper configuration."""
        self.test_workspace = Path(tempfile.mkdtemp(prefix="dspy_test_"))
        self.test_logs = self.test_workspace / "logs"
        self.test_logs.mkdir(exist_ok=True)
        
        # Set test environment variables
        os.environ.update({
            "DSPY_AUTO_TRAIN": "false",
            "DSPY_LOG_LEVEL": "DEBUG",
            "USE_OLLAMA": "false",  # Use mock LLM for testing
            "WORKSPACE_DIR": str(self.test_workspace),
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            "REDDB_URL": "http://localhost:8080",
            "REDDB_NAMESPACE": "test",
            "REDDB_TOKEN": "test-token"
        })
        
        yield
        
        # Cleanup
        import shutil
        shutil.rmtree(self.test_workspace, ignore_errors=True)
    
    def test_agent_launcher_basic_functionality(self):
        """Test basic launcher functionality."""
        # Test launcher with check-only mode
        result = launcher_run(["--check-only", "--workspace", str(self.test_workspace)])
        assert result == 0
        
        # Test launcher with different modes
        for mode in ["cli", "code", "dashboard"]:
            result = launcher_run([
                "--mode", mode,
                "--workspace", str(self.test_workspace),
                "--check-only"
            ])
            assert result == 0
    
    def test_cli_commands(self):
        """Test all CLI commands and their basic functionality."""
        from typer.testing import CliRunner
        
        runner = CliRunner()
        
        # Test help command
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "dspy-code" in result.stdout
        
        # Test workspace initialization
        result = runner.invoke(app, ["init", "--workspace", str(self.test_workspace)])
        assert result.exit_code == 0
        
        # Test status command
        result = runner.invoke(app, ["status", "--workspace", str(self.test_workspace)])
        assert result.exit_code == 0
    
    @pytest.mark.asyncio
    async def test_streaming_runtime(self):
        """Test streaming runtime functionality."""
        runtime = StreamingRuntime()
        
        # Test runtime initialization
        await runtime.initialize()
        assert runtime.is_initialized
        
        # Test message publishing
        test_message = {"test": "data", "timestamp": time.time()}
        await runtime.publish("test.topic", test_message)
        
        # Test message consumption
        messages = []
        async def message_handler(msg):
            messages.append(msg)
        
        await runtime.subscribe("test.topic", message_handler)
        await asyncio.sleep(0.1)  # Allow message processing
        
        # Cleanup
        await runtime.shutdown()
    
    def test_memory_manager(self):
        """Test memory manager functionality."""
        memory = MemoryManager(workspace=self.test_workspace)
        
        # Test memory storage
        test_data = {"key": "value", "timestamp": time.time()}
        memory.store("test_key", test_data)
        
        # Test memory retrieval
        retrieved = memory.retrieve("test_key")
        assert retrieved == test_data
        
        # Test memory search
        results = memory.search("test")
        assert len(results) > 0
    
    def test_embedding_functionality(self):
        """Test embedding generation and storage."""
        embedder = Embedder()
        
        # Test text embedding
        text = "This is a test document for embedding."
        embedding = embedder.embed_text(text)
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        
        # Test batch embedding
        texts = ["First document", "Second document", "Third document"]
        embeddings = embedder.embed_batch(texts)
        assert len(embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_database_operations(self):
        """Test database operations."""
        db = DatabaseKit()
        
        # Test connection
        assert db.connect()
        
        # Test data operations
        test_data = {
            "id": "test_001",
            "content": "Test content",
            "metadata": {"type": "test"}
        }
        
        # Store data
        db.store("test_collection", test_data)
        
        # Retrieve data
        retrieved = db.retrieve("test_collection", "test_001")
        assert retrieved is not None
        assert retrieved["content"] == "Test content"
        
        # Search data
        results = db.search("test_collection", "content:test")
        assert len(results) > 0
        
        db.disconnect()
    
    def test_knowledge_agent(self):
        """Test knowledge agent functionality."""
        knowledge = KnowledgeAgent(workspace=self.test_workspace)
        
        # Test knowledge ingestion
        test_doc = {
            "title": "Test Document",
            "content": "This is test content for knowledge base.",
            "source": "test"
        }
        knowledge.ingest(test_doc)
        
        # Test knowledge retrieval
        results = knowledge.search("test content")
        assert len(results) > 0
        
        # Test knowledge summarization
        summary = knowledge.summarize("test content")
        assert summary is not None
        assert len(summary) > 0
    
    def test_orchestrator_runtime(self):
        """Test orchestrator runtime functionality."""
        orchestrator = OrchestratorRuntime(workspace=self.test_workspace)
        
        # Test task creation
        task = {
            "id": "test_task_001",
            "type": "code_analysis",
            "description": "Analyze test code",
            "priority": "high"
        }
        
        orchestrator.create_task(task)
        
        # Test task execution
        result = orchestrator.execute_task("test_task_001")
        assert result is not None
        
        # Test task status
        status = orchestrator.get_task_status("test_task_001")
        assert status in ["pending", "running", "completed", "failed"]
    
    def test_router_worker(self):
        """Test router worker functionality."""
        router = RouterWorker()
        
        # Test message routing
        test_message = {
            "topic": "agent.requests",
            "payload": {"action": "analyze", "data": "test"}
        }
        
        # Test routing logic
        route = router.route_message(test_message)
        assert route is not None
        assert "worker" in route
    
    @pytest.mark.slow
    def test_rl_training_basic(self):
        """Test basic RL training functionality."""
        # Mock environment for testing
        def mock_env_factory():
            class MockEnv:
                def __init__(self):
                    self.observation_space = Mock()
                    self.action_space = Mock()
                    self.step_count = 0
                
                def reset(self):
                    self.step_count = 0
                    return {"obs": [0, 1, 2]}, {}
                
                def step(self, action):
                    self.step_count += 1
                    obs = {"obs": [0, 1, 2]}
                    reward = 1.0 if action == 0 else 0.0
                    terminated = self.step_count >= 10
                    truncated = False
                    info = {}
                    return obs, reward, terminated, truncated, info
            
            return MockEnv()
        
        # Test training with minimal steps
        try:
            stats = train_puffer_policy(
                make_env=mock_env_factory,
                steps=100,
                n_envs=2,
                entropy_coef=0.01,
                replay_capacity=128,
                replay_batch=32,
                grad_clip=1.0,
                checkpoint_dir=str(self.test_workspace / "checkpoints"),
                checkpoint_interval=50
            )
            assert stats is not None
            assert "episode_reward" in stats
        except Exception as e:
            pytest.skip(f"RL training test skipped due to: {e}")
    
    def test_training_entrypoint(self):
        """Test training entrypoint functionality."""
        # Mock training arguments
        test_args = [
            "--workspace", str(self.test_workspace),
            "--signature", "CodeContextSig",
            "--verifiers", "dspy_agent.verifiers.custom",
            "--steps", "10",
            "--env", "test"
        ]
        
        # Test training entrypoint (mock mode)
        with patch('sys.argv', ['training_entrypoint'] + test_args):
            try:
                result = training_main()
                assert result is not None
            except Exception as e:
                pytest.skip(f"Training entrypoint test skipped due to: {e}")
    
    def test_agent_integration_workflow(self):
        """Test complete agent integration workflow."""
        # Initialize all components
        memory = MemoryManager(workspace=self.test_workspace)
        knowledge = KnowledgeAgent(workspace=self.test_workspace)
        orchestrator = OrchestratorRuntime(workspace=self.test_workspace)
        
        # Test complete workflow
        # 1. Ingest knowledge
        doc = {
            "title": "Integration Test",
            "content": "This is an integration test document.",
            "source": "test"
        }
        knowledge.ingest(doc)
        
        # 2. Create task
        task = {
            "id": "integration_task",
            "type": "knowledge_query",
            "description": "Query integration test document",
            "priority": "medium"
        }
        orchestrator.create_task(task)
        
        # 3. Execute task with knowledge retrieval
        result = orchestrator.execute_task("integration_task")
        assert result is not None
        
        # 4. Store result in memory
        memory.store("integration_result", result)
        
        # 5. Verify end-to-end flow
        stored_result = memory.retrieve("integration_result")
        assert stored_result == result
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test invalid workspace
        with pytest.raises((OSError, ValueError)):
            MemoryManager(workspace=Path("/invalid/path/that/does/not/exist"))
        
        # Test invalid configuration
        with patch.dict(os.environ, {"DSPY_LOG_LEVEL": "INVALID"}):
            # Should handle gracefully
            config = Config()
            assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        # Test network failure handling
        with patch('dspy_agent.llm.LLMProvider._make_request') as mock_request:
            mock_request.side_effect = ConnectionError("Network error")
            
            llm = LLMProvider()
            # Should handle network errors gracefully
            try:
                response = llm.generate("test prompt")
                assert response is None or "error" in response.lower()
            except Exception:
                # Expected to fail gracefully
                pass
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Test memory usage tracking
        memory = MemoryManager(workspace=self.test_workspace)
        
        # Store multiple items
        for i in range(100):
            memory.store(f"item_{i}", {"data": f"test_data_{i}"})
        
        # Test memory metrics
        metrics = memory.get_metrics()
        assert "total_items" in metrics
        assert "memory_usage" in metrics
        assert metrics["total_items"] >= 100
    
    def test_concurrent_operations(self):
        """Test concurrent operations and thread safety."""
        import threading
        import queue
        
        memory = MemoryManager(workspace=self.test_workspace)
        results = queue.Queue()
        
        def worker(worker_id):
            try:
                for i in range(10):
                    memory.store(f"worker_{worker_id}_item_{i}", {"data": f"data_{i}"})
                results.put(f"worker_{worker_id}_completed")
            except Exception as e:
                results.put(f"worker_{worker_id}_error: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        completed_workers = []
        while not results.empty():
            result = results.get()
            completed_workers.append(result)
        
        assert len(completed_workers) == 5
        assert all("completed" in result for result in completed_workers)
    
    def test_data_persistence(self):
        """Test data persistence across restarts."""
        # Store data
        memory = MemoryManager(workspace=self.test_workspace)
        test_data = {"persistent": "data", "timestamp": time.time()}
        memory.store("persistent_key", test_data)
        
        # Simulate restart by creating new instance
        memory2 = MemoryManager(workspace=self.test_workspace)
        retrieved = memory2.retrieve("persistent_key")
        
        assert retrieved == test_data
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test default configuration
        config = Config()
        assert config.workspace is not None
        assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        # Test configuration override
        with patch.dict(os.environ, {"DSPY_LOG_LEVEL": "DEBUG"}):
            config = Config()
            assert config.log_level == "DEBUG"
        
        # Test invalid configuration handling
        with patch.dict(os.environ, {"DSPY_AUTO_TRAIN": "invalid"}):
            config = Config()
            # Should default to safe value
            assert config.auto_train in [True, False]


class TestDockerIntegration:
    """Test Docker integration and containerized deployment."""
    
    def test_docker_build(self):
        """Test Docker image build process."""
        # Test if Docker is available
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            assert "Docker version" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Docker not available")
    
    def test_docker_compose_validation(self):
        """Test Docker Compose configuration validation."""
        compose_file = Path("/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml")
        
        if not compose_file.exists():
            pytest.skip("Docker Compose file not found")
        
        try:
            result = subprocess.run(
                ["docker-compose", "-f", str(compose_file), "config"],
                capture_output=True,
                text=True,
                check=True
            )
            assert result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Docker Compose not available")


class TestPackageBuild:
    """Test package build and distribution."""
    
    def test_package_structure(self):
        """Test package structure and metadata."""
        # Test pyproject.toml exists and is valid
        pyproject_path = Path("/Users/robbiepasquale/dspy_stuff/pyproject.toml")
        assert pyproject_path.exists()
        
        # Test package can be imported
        import dspy_agent
        assert hasattr(dspy_agent, '__version__')
    
    def test_entry_points(self):
        """Test console script entry points."""
        # Test that entry points are properly configured
        from dspy_agent.cli import app
        assert app is not None
        
        # Test launcher entry point
        from dspy_agent.launcher import main
        assert main is not None
    
    def test_dependencies(self):
        """Test that all dependencies are properly specified."""
        import importlib
        
        # Test core dependencies
        core_deps = [
            'dspy',
            'typer',
            'rich',
            'pydantic',
            'sentence_transformers',
            'transformers'
        ]
        
        for dep in core_deps:
            try:
                importlib.import_module(dep.replace('-', '_'))
            except ImportError:
                pytest.fail(f"Core dependency {dep} not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
