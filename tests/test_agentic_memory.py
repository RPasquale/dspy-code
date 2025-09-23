#!/usr/bin/env python3
"""
Comprehensive tests for dspy_agent.agentic.memory module.

Tests the AgentKnowledgeGraph, retrieval features computation,
and retrieval event logging/loading functionality.
"""

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from dspy_agent.agentic.memory import (
    AgentKnowledgeGraph,
    compute_retrieval_features,
    log_retrieval_event,
    load_retrieval_events,
    query_retrieval_events,
    get_retrieval_statistics,
)


class TestAgentKnowledgeGraph:
    """Test the AgentKnowledgeGraph class functionality."""

    def test_init_empty_graph(self):
        """Test initialization of empty knowledge graph."""
        kg = AgentKnowledgeGraph()
        assert kg.nodes == {}
        assert kg.edges == {}

    def test_add_file_hint(self):
        """Test adding file hints to the knowledge graph."""
        kg = AgentKnowledgeGraph()
        
        # Add a file hint
        kg.add_file_hint("implement feature", "src/main.py", 0.8)
        
        # Check nodes were created
        assert "task::implement feature" in kg.nodes
        assert "file::src/main.py" in kg.nodes
        
        # Check edge was created
        edge = ("task::implement feature", "file::src/main.py")
        assert edge in kg.edges
        assert kg.edges[edge]["weight"] == 0.8
        assert kg.edges[edge]["count"] == 1.0

    def test_add_file_hint_multiple_times(self):
        """Test adding the same file hint multiple times."""
        kg = AgentKnowledgeGraph()
        
        # Add same hint multiple times with different confidences
        kg.add_file_hint("task", "file.py", 0.5)
        kg.add_file_hint("task", "file.py", 0.9)
        kg.add_file_hint("task", "file.py", 0.3)
        
        edge = ("task::task", "file::file.py")
        assert kg.edges[edge]["weight"] == 0.9  # Should keep max weight
        assert kg.edges[edge]["count"] == 3.0   # Should increment count

    def test_add_reference(self):
        """Test adding references between nodes."""
        kg = AgentKnowledgeGraph()
        
        # Add a reference
        kg.add_reference("source_node", "target_node", 0.7)
        
        # Check nodes were created
        assert "source_node" in kg.nodes
        assert "target_node" in kg.nodes
        
        # Check edge was created
        edge = ("source_node", "target_node")
        assert edge in kg.edges
        assert kg.edges[edge]["weight"] == 0.7
        assert kg.edges[edge]["count"] == 1.0

    def test_add_reference_default_weight(self):
        """Test adding reference with default weight."""
        kg = AgentKnowledgeGraph()
        
        kg.add_reference("source", "target")
        
        edge = ("source", "target")
        assert kg.edges[edge]["weight"] == 0.0
        assert kg.edges[edge]["count"] == 1.0

    def test_touch_creates_node(self):
        """Test that _touch creates nodes with default values."""
        kg = AgentKnowledgeGraph()
        
        kg._touch("new_node")
        
        assert "new_node" in kg.nodes
        assert kg.nodes["new_node"]["degree"] == 0.0

    def test_summarise_empty_graph(self):
        """Test summarise on empty graph."""
        kg = AgentKnowledgeGraph()
        summary = kg.summarise()
        
        expected = {
            "kg_nodes": 0.0,
            "kg_edges": 0.0,
            "kg_avg_weight": 0.0,
            "kg_avg_fanout": 0.0,
        }
        assert summary == expected

    def test_summarise_with_data(self):
        """Test summarise with actual graph data."""
        kg = AgentKnowledgeGraph()
        
        # Add some data
        kg.add_file_hint("task1", "file1.py", 0.8)
        kg.add_file_hint("task1", "file2.py", 0.6)
        kg.add_reference("node1", "node2", 0.9)
        
        summary = kg.summarise()
        
        assert summary["kg_nodes"] == 5.0  # task1, file1.py, file2.py, node1, node2
        assert summary["kg_edges"] == 3.0  # 2 file hints + 1 reference
        assert summary["kg_avg_weight"] > 0.0
        assert summary["kg_avg_fanout"] > 0.0

    def test_summarise_avg_weight_calculation(self):
        """Test that average weight is calculated correctly."""
        kg = AgentKnowledgeGraph()
        
        # Add edges with known weights
        kg.add_reference("a", "b", 0.2)
        kg.add_reference("c", "d", 0.8)
        
        summary = kg.summarise()
        assert summary["kg_avg_weight"] == 0.5  # (0.2 + 0.8) / 2

    def test_summarise_avg_fanout_calculation(self):
        """Test that average fanout is calculated correctly."""
        kg = AgentKnowledgeGraph()
        
        # Add multiple edges from same source
        kg.add_reference("source", "target1", 0.5)
        kg.add_reference("source", "target2", 0.3)
        kg.add_reference("other", "target3", 0.7)
        
        summary = kg.summarise()
        # source has 2 edges, other has 1 edge
        # avg_fanout = (2 + 1) / 2 = 1.5
        assert summary["kg_avg_fanout"] == 1.5


class TestComputeRetrievalFeatures:
    """Test the compute_retrieval_features function."""

    def test_compute_features_empty_input(self):
        """Test with empty patches and no retrieval events."""
        workspace = Path("/tmp/test")
        features = compute_retrieval_features(workspace, [], None)
        
        expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert features == expected

    def test_compute_features_with_patches(self):
        """Test feature computation with patch data."""
        workspace = Path("/tmp/test")
        patches = [
            {
                "task": "implement feature",
                "file_hints": "src/main.py,src/utils.py",
                "metrics": {"pass_rate": 0.8}
            },
            {
                "task": "fix bug",
                "file_candidates": ["src/bug.py"],
                "metrics": {"pass_rate": 0.6}
            }
        ]
        
        features = compute_retrieval_features(workspace, patches, None)
        
        # Should have 8 features
        assert len(features) == 8
        # Should have nodes and edges from the patches
        assert features[0] > 0  # kg_nodes
        assert features[1] > 0  # kg_edges
        assert features[4] > 0  # precision (average pass_rate)
        assert features[5] > 0  # coverage (unique files)

    def test_compute_features_with_retrieval_events(self):
        """Test feature computation with retrieval events."""
        workspace = Path("/tmp/test")
        patches = []
        retrieval_events = [
            {
                "query": "find function",
                "hits": [
                    {"path": "src/func.py", "score": 0.9},
                    {"path": "src/helper.py", "score": 0.7}
                ]
            }
        ]
        
        features = compute_retrieval_features(workspace, patches, retrieval_events)
        
        assert len(features) == 8
        assert features[0] > 0  # kg_nodes
        assert features[1] > 0  # kg_edges
        assert features[6] > 0  # avg_score
        assert features[7] == 1.0  # query_count

    def test_compute_features_mixed_data(self):
        """Test with both patches and retrieval events."""
        workspace = Path("/tmp/test")
        patches = [
            {
                "task": "test task",
                "file_hints": "test.py",
                "metrics": {"pass_rate": 0.5}
            }
        ]
        retrieval_events = [
            {
                "query": "test query",
                "hits": [{"path": "test.py", "score": 0.8}]
            }
        ]
        
        features = compute_retrieval_features(workspace, patches, retrieval_events)
        
        assert len(features) == 8
        # Should combine data from both sources
        assert features[0] > 0  # kg_nodes
        assert features[1] > 0  # kg_edges

    def test_compute_features_error_handling(self):
        """Test error handling in feature computation."""
        workspace = Path("/tmp/test")
        patches = [
            {
                "task": "test",
                "file_hints": "file.py",
                "metrics": {"pass_rate": "invalid"}  # Invalid pass_rate
            }
        ]
        
        features = compute_retrieval_features(workspace, patches, None)
        
        # Should handle errors gracefully
        assert len(features) == 8
        assert features[4] == 0.0  # precision should be 0 due to error

    def test_compute_features_different_hint_formats(self):
        """Test with different file hint formats."""
        workspace = Path("/tmp/test")
        patches = [
            {
                "task": "task1",
                "file_hints": "file1.py,file2.py",  # String format
                "metrics": {"pass_rate": 0.5}
            },
            {
                "task": "task2",
                "file_candidates": ["file3.py", "file4.py"],  # List format
                "metrics": {"pass_rate": 0.7}
            }
        ]
        
        features = compute_retrieval_features(workspace, patches, None)
        
        assert len(features) == 8
        assert features[5] == 4.0  # coverage should be 4 unique files


class TestLogRetrievalEvent:
    """Test the log_retrieval_event function."""

    def test_log_retrieval_event_basic(self):
        """Test basic retrieval event logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            query = "test query"
            hits = [
                {"path": "file1.py", "score": 0.9, "source": "semantic"},
                {"path": "file2.py", "score": 0.7, "source": "keyword"}
            ]
            
            log_retrieval_event(workspace, query, hits)
            
            # Check that log file was created
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            assert log_path.exists()
            
            # Check log content
            with open(log_path) as f:
                lines = f.readlines()
                assert len(lines) == 1
                
                record = json.loads(lines[0])
                assert record["query"] == query
                assert len(record["hits"]) == 2
                assert record["hits"][0]["path"] == "file1.py"
                assert record["hits"][0]["score"] == 0.9

    def test_log_retrieval_event_with_limit(self):
        """Test retrieval event logging with hit limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            query = "test query"
            hits = [
                {"path": f"file{i}.py", "score": 0.5}
                for i in range(25)  # More than default limit of 20
            ]
            
            log_retrieval_event(workspace, query, hits, limit=5)
            
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            with open(log_path) as f:
                record = json.loads(f.read())
                assert len(record["hits"]) == 5  # Should be limited to 5

    def test_log_retrieval_event_invalid_hits(self):
        """Test logging with invalid hit data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            query = "test query"
            hits = [
                {"path": "valid.py", "score": 0.8},
                {"path": "", "score": 0.5},  # Empty path
                {"score": 0.3},  # Missing path
                {"path": "another.py", "score": 0.6}
            ]
            
            log_retrieval_event(workspace, query, hits)
            
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            with open(log_path) as f:
                record = json.loads(f.read())
                # Should only include valid hits
                assert len(record["hits"]) == 2
                assert record["hits"][0]["path"] == "valid.py"
                assert record["hits"][1]["path"] == "another.py"

    def test_log_retrieval_event_reddb_integration(self):
        """Test that RedDB integration doesn't break file logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            query = "test query"
            hits = [{"path": "test.py", "score": 0.8}]
            
            # Test that file logging still works even if RedDB fails
            log_retrieval_event(workspace, query, hits)
            
            # Verify file logging worked
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            assert log_path.exists()
            
            with open(log_path) as f:
                record = json.loads(f.read())
                assert record["query"] == query
                assert len(record["hits"]) == 1
                assert record["hits"][0]["path"] == "test.py"

    def test_log_retrieval_event_reddb_error_handling(self):
        """Test that RedDB errors don't break file logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            query = "test query"
            hits = [{"path": "test.py", "score": 0.8}]
            
            # Mock RedDB to raise an exception
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager', side_effect=Exception("RedDB error")):
                log_retrieval_event(workspace, query, hits)
                
                # File logging should still work
                log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
                assert log_path.exists()
                
                with open(log_path) as f:
                    record = json.loads(f.read())
                    assert record["query"] == query

    def test_log_retrieval_event_file_error_handling(self):
        """Test error handling when file writing fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            query = "test query"
            hits = [{"path": "test.py", "score": 0.8}]
            
            # Make the directory read-only to cause file write error
            agentic_dir = workspace / ".dspy_agentic"
            agentic_dir.mkdir()
            agentic_dir.chmod(0o444)  # Read-only
            
            # Should not raise exception
            log_retrieval_event(workspace, query, hits)
            
            # Restore permissions for cleanup
            agentic_dir.chmod(0o755)


class TestLoadRetrievalEvents:
    """Test the load_retrieval_events function."""

    def test_load_retrieval_events_no_file(self):
        """Test loading when no log file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            # Mock RedDB to return empty list to ensure test isolation
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm:
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                mock_manager.get_recent_retrieval_events.return_value = []
                
                events = load_retrieval_events(workspace)
                assert events == []

    def test_load_retrieval_events_empty_file(self):
        """Test loading from empty log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            log_path.parent.mkdir()
            log_path.touch()
            
            # Mock RedDB to return empty list to ensure test isolation
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm:
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                mock_manager.get_recent_retrieval_events.return_value = []
                
                events = load_retrieval_events(workspace)
                assert events == []

    def test_load_retrieval_events_valid_data(self):
        """Test loading valid retrieval events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            log_path.parent.mkdir()
            
            # Write test data
            test_events = [
                {
                    "timestamp": time.time(),
                    "event_id": str(uuid.uuid4()),
                    "query": "query1",
                    "hits": [{"path": "file1.py", "score": 0.8}]
                },
                {
                    "timestamp": time.time(),
                    "event_id": str(uuid.uuid4()),
                    "query": "query2",
                    "hits": [{"path": "file2.py", "score": 0.6}]
                }
            ]
            
            with open(log_path, 'w') as f:
                for event in test_events:
                    f.write(json.dumps(event) + '\n')
            
            # Mock RedDB to return empty list to ensure test isolation
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm:
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                mock_manager.get_recent_retrieval_events.return_value = []
                
                events = load_retrieval_events(workspace)
                assert len(events) == 2
                # Events are sorted by timestamp (newest first), so query2 comes first
                assert events[0]["query"] == "query2"
                assert events[1]["query"] == "query1"

    def test_load_retrieval_events_with_limit(self):
        """Test loading with limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            log_path.parent.mkdir()
            
            # Write 10 events
            with open(log_path, 'w') as f:
                for i in range(10):
                    event = {
                        "timestamp": time.time(),
                        "event_id": str(uuid.uuid4()),
                        "query": f"query{i}",
                        "hits": []
                    }
                    f.write(json.dumps(event) + '\n')
            
            # Mock RedDB to return empty list to ensure test isolation
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm:
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                mock_manager.get_recent_retrieval_events.return_value = []
                
                # Load with limit of 5
                events = load_retrieval_events(workspace, limit=5)
                assert len(events) == 5
                # Should get the last 5 events, sorted by timestamp (newest first)
                assert events[0]["query"] == "query9"
                assert events[4]["query"] == "query5"

    def test_load_retrieval_events_invalid_json(self):
        """Test loading with invalid JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            log_path.parent.mkdir()
            
            # Write mix of valid and invalid JSON
            with open(log_path, 'w') as f:
                f.write('{"valid": "json"}\n')
                f.write('invalid json line\n')
                f.write('{"another": "valid"}\n')
                f.write('\n')  # Empty line
            
            # Mock RedDB to return empty list to ensure test isolation
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm:
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                mock_manager.get_recent_retrieval_events.return_value = []
                
                events = load_retrieval_events(workspace)
                assert len(events) == 2  # Only valid JSON should be loaded

    def test_load_retrieval_events_file_read_error(self):
        """Test error handling when file read fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            log_path.parent.mkdir()
            log_path.touch()
            
            # Make file unreadable
            log_path.chmod(0o000)
            
            # Mock RedDB to return empty list to ensure test isolation
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm:
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                mock_manager.get_recent_retrieval_events.return_value = []
                
                events = load_retrieval_events(workspace)
                assert events == []  # Should return empty list on error
            
            # Restore permissions for cleanup
            log_path.chmod(0o644)


class TestIntegration:
    """Integration tests for the agentic memory module."""

    def test_full_workflow(self):
        """Test the complete workflow from logging to loading to feature computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Mock RedDB to ensure test isolation
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm, \
                 patch('dspy_agent.agentic.memory.create_log_entry') as mock_log_entry, \
                 patch('dspy_agent.agentic.memory.create_retrieval_event') as mock_create_event, \
                 patch('dspy_agent.agentic.memory.Environment') as mock_env, \
                 patch('os.getenv', return_value='development'):
                
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                mock_log_entry.return_value = Mock()
                mock_create_event.return_value = Mock()
                mock_env.DEVELOPMENT = "DEVELOPMENT"
                mock_env.__getitem__ = Mock(return_value="DEVELOPMENT")
                
                # Log some retrieval events
                log_retrieval_event(workspace, "query1", [
                    {"path": "file1.py", "score": 0.9},
                    {"path": "file2.py", "score": 0.7}
                ])
                log_retrieval_event(workspace, "query2", [
                    {"path": "file3.py", "score": 0.8}
                ])
                
                # Mock RedDB to return the events we just logged
                mock_events = [
                    {
                        "event_id": "test-1",
                        "timestamp": time.time(),
                        "query": "query1",
                        "hits": [{"path": "file1.py", "score": 0.9}, {"path": "file2.py", "score": 0.7}]
                    },
                    {
                        "event_id": "test-2", 
                        "timestamp": time.time(),
                        "query": "query2",
                        "hits": [{"path": "file3.py", "score": 0.8}]
                    }
                ]
                mock_manager.get_recent_retrieval_events.return_value = mock_events
                
                # Load the events - should get only the mocked RedDB events
                # Use limit=2 to prevent loading from file
                events = load_retrieval_events(workspace, limit=2)
                assert len(events) == 2
                # Verify we got the mocked events, not the file events
                # Events are sorted by timestamp (newest first)
                event_ids = {event["event_id"] for event in events}
                assert event_ids == {"test-1", "test-2"}
                
                # Create some patch data
                patches = [
                    {
                        "task": "implement feature",
                        "file_hints": "file1.py,file2.py",
                        "metrics": {"pass_rate": 0.8}
                    }
                ]
                
                # Compute features
                features = compute_retrieval_features(workspace, patches, events)
                
                # Verify features are computed correctly
                assert len(features) == 8
                assert features[0] > 0  # Should have nodes
                assert features[1] > 0  # Should have edges
                assert features[4] > 0  # Should have precision
                assert features[5] > 0  # Should have coverage

    def test_knowledge_graph_consistency(self):
        """Test that knowledge graph maintains consistency across operations."""
        kg = AgentKnowledgeGraph()
        
        # Add various types of data
        kg.add_file_hint("task1", "file1.py", 0.8)
        kg.add_file_hint("task1", "file2.py", 0.6)
        kg.add_reference("query::search", "file::file1.py", 0.9)
        kg.add_reference("query::search", "file::file3.py", 0.7)
        
        summary = kg.summarise()
        
        # Verify consistency
        assert summary["kg_nodes"] == 5.0  # task1, file1.py, file2.py, query::search, file::file3.py
        assert summary["kg_edges"] == 4.0  # 2 file hints + 2 references
        
        # Verify node degrees are tracked
        task_node = kg.nodes["task::task1"]
        assert task_node["degree"] == 0.0  # Default degree
        
        # Verify edge weights are correct
        edge1 = ("task::task1", "file::file1.py")
        assert kg.edges[edge1]["weight"] == 0.8
        assert kg.edges[edge1]["count"] == 1.0


class TestRedDBIntegration:
    """Test RedDB integration for retrieval events."""

    def test_log_retrieval_event_reddb_success(self):
        """Test successful RedDB integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            query = "test query"
            hits = [{"path": "test.py", "score": 0.8}]
            
            # Mock RedDB components
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm, \
                 patch('dspy_agent.agentic.memory.create_log_entry') as mock_log_entry, \
                 patch('dspy_agent.agentic.memory.create_retrieval_event') as mock_create_event, \
                 patch('dspy_agent.agentic.memory.Environment') as mock_env, \
                 patch('os.getenv', return_value='development'):
                
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                mock_log_entry.return_value = Mock()
                mock_create_event.return_value = Mock()
                
                # Mock Environment enum
                mock_env.DEVELOPMENT = "DEVELOPMENT"
                mock_env.__getitem__ = Mock(return_value="DEVELOPMENT")
                
                log_retrieval_event(workspace, query, hits)
                
                # Verify RedDB calls were made
                mock_dm.assert_called_once()
                mock_create_event.assert_called_once()
                mock_manager.record_retrieval_event.assert_called_once()
                mock_manager.log.assert_called_once()

    def test_load_retrieval_events_reddb_priority(self):
        """Test that RedDB events are loaded with priority over file events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Mock RedDB to return some events
            reddb_events = [
                {
                    "event_id": "reddb-1",
                    "timestamp": 1000.0,
                    "query": "reddb query",
                    "hits": [{"path": "reddb.py", "score": 0.9}]
                }
            ]
            
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager') as mock_dm:
                mock_manager = Mock()
                mock_dm.return_value = mock_manager
                # Mock the get_recent_retrieval_events method
                mock_manager.get_recent_retrieval_events.return_value = reddb_events
                
                events = load_retrieval_events(workspace, limit=10)
                
                # Should get RedDB events
                assert len(events) == 1
                assert events[0]["event_id"] == "reddb-1"
                assert events[0]["query"] == "reddb query"

    def test_load_retrieval_events_fallback_to_file(self):
        """Test fallback to file when RedDB fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Create file with events
            log_path = workspace / ".dspy_agentic" / "retrieval.jsonl"
            log_path.parent.mkdir()
            
            file_event = {
                "event_id": "file-1",
                "timestamp": 1000.0,
                "query": "file query",
                "hits": [{"path": "file.py", "score": 0.8}]
            }
            
            with open(log_path, 'w') as f:
                f.write(json.dumps(file_event) + '\n')
            
            # Mock RedDB to fail
            with patch('dspy_agent.agentic.memory.get_enhanced_data_manager', side_effect=Exception("RedDB error")):
                events = load_retrieval_events(workspace, limit=10)
                
                # Should fallback to file events
                assert len(events) == 1
                assert events[0]["event_id"] == "file-1"
                assert events[0]["query"] == "file query"

    def test_query_retrieval_events_filtering(self):
        """Test query filtering functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Mock events with different characteristics
            mock_events = [
                {
                    "event_id": "1",
                    "timestamp": 1000.0,
                    "query": "search for function",
                    "hits": [{"path": "func.py", "score": 0.9}]
                },
                {
                    "event_id": "2", 
                    "timestamp": 2000.0,
                    "query": "find variable",
                    "hits": [{"path": "var.py", "score": 0.3}]
                },
                {
                    "event_id": "3",
                    "timestamp": 3000.0,
                    "query": "search for class",
                    "hits": [{"path": "class.py", "score": 0.8}]
                }
            ]
            
            with patch('dspy_agent.agentic.memory.load_retrieval_events', return_value=mock_events):
                # Test query filter
                filtered = query_retrieval_events(workspace, query_filter="function")
                assert len(filtered) == 1
                assert filtered[0]["query"] == "search for function"
                
                # Test min score filter
                filtered = query_retrieval_events(workspace, min_score=0.5)
                assert len(filtered) == 2  # Only events with score >= 0.5
                
                # Test timestamp filter
                filtered = query_retrieval_events(workspace, since_timestamp=1500.0)
                assert len(filtered) == 2  # Only events after timestamp 1500

    def test_get_retrieval_statistics(self):
        """Test retrieval statistics calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Mock events for statistics
            mock_events = [
                {
                    "query": "query1",
                    "hits": [
                        {"path": "file1.py", "score": 0.8},
                        {"path": "file2.py", "score": 0.6}
                    ]
                },
                {
                    "query": "query2", 
                    "hits": [
                        {"path": "file3.py", "score": 0.9}
                    ]
                }
            ]
            
            with patch('dspy_agent.agentic.memory.load_retrieval_events', return_value=mock_events):
                stats = get_retrieval_statistics(workspace)
                
                assert stats["total_events"] == 2.0
                assert stats["avg_hits_per_query"] == 1.5  # (2 + 1) / 2
                assert stats["avg_score"] == pytest.approx(0.766, rel=1e-2)  # (0.8 + 0.6 + 0.9) / 3
                assert stats["unique_queries"] == 2.0
                assert stats["unique_files"] == 3.0

    def test_get_retrieval_statistics_empty(self):
        """Test statistics with no events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            with patch('dspy_agent.agentic.memory.load_retrieval_events', return_value=[]):
                stats = get_retrieval_statistics(workspace)
                
                assert stats["total_events"] == 0.0
                assert stats["avg_hits_per_query"] == 0.0
                assert stats["avg_score"] == 0.0
                assert stats["unique_queries"] == 0.0
                assert stats["unique_files"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
