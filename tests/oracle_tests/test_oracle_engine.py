"""
Unit tests for Oracle Engine
Tests multi-timeline forks, cause-effect chains, parallel predictions, and quantum randomness
"""

import unittest
import torch
import numpy as np
import tempfile
import json
import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.oracle_engine import (
    OracleEngine, 
    TimelineBranch, 
    CauseEffectGraph, 
    CauseEffectNode,
    PredictionType,
    QuantumRandomness
)

class TestTimelineBranch(unittest.TestCase):
    """Test TimelineBranch functionality"""
    
    def setUp(self):
        self.branch = TimelineBranch("test_branch", 0.8)
    
    def test_branch_creation(self):
        """Test timeline branch creation"""
        self.assertEqual(self.branch.branch_id, "test_branch")
        self.assertEqual(self.branch.probability, 0.8)
        self.assertIsNotNone(self.branch.divergence_point)
        self.assertEqual(len(self.branch.events), 0)
        self.assertEqual(len(self.branch.sub_branches), 0)
    
    def test_add_event(self):
        """Test adding events to timeline branch"""
        event = {"type": "test", "data": "test_data"}
        self.branch.add_event(event)
        
        self.assertEqual(len(self.branch.events), 1)
        self.assertEqual(self.branch.events[0]["event"], event)
    
    def test_create_sub_branch(self):
        """Test creating sub-branches"""
        sub_branch = self.branch.create_sub_branch("sub_branch", 0.5, 2.0)
        
        self.assertEqual(len(self.branch.sub_branches), 1)
        self.assertEqual(sub_branch.parent_branch, "test_branch")
        self.assertEqual(sub_branch.weight, 2.0)
        self.assertEqual(self.branch.fork_count, 1)
    
    def test_branch_probability(self):
        """Test branch probability calculation"""
        # Create sub-branches
        sub1 = self.branch.create_sub_branch("sub1", 0.6, 1.5)
        sub2 = self.branch.create_sub_branch("sub2", 0.4, 0.8)
        
        # Calculate total probability
        total_prob = self.branch.get_branch_probability()
        expected_prob = 0.8 * 0.6 * 1.5 * 0.4 * 0.8
        self.assertAlmostEqual(total_prob, expected_prob, places=6)
    
    def test_branch_depth(self):
        """Test branch depth calculation"""
        # Create nested branches
        sub1 = self.branch.create_sub_branch("sub1")
        sub2 = sub1.create_sub_branch("sub2")
        sub3 = sub2.create_sub_branch("sub3")
        
        self.assertEqual(self.branch.get_branch_depth(), 3)
        self.assertEqual(sub1.get_branch_depth(), 2)
        self.assertEqual(sub2.get_branch_depth(), 1)
        self.assertEqual(sub3.get_branch_depth(), 0)

class TestCauseEffectGraph(unittest.TestCase):
    """Test CauseEffectGraph functionality"""
    
    def setUp(self):
        self.graph = CauseEffectGraph()
    
    def test_add_node(self):
        """Test adding nodes to the graph"""
        node = CauseEffectNode(
            node_id="test_node",
            event="test_event",
            timestamp=datetime.now(),
            probability=0.8,
            confidence=0.9
        )
        
        self.graph.add_node(node)
        self.assertIn("test_node", self.graph.nodes)
        self.assertIn("test_node", self.graph.graph.nodes)
    
    def test_add_cause_effect(self):
        """Test adding cause-effect relationships"""
        # Create nodes
        cause_node = CauseEffectNode(
            node_id="cause",
            event="cause_event",
            timestamp=datetime.now(),
            probability=1.0,
            confidence=1.0
        )
        
        effect_node = CauseEffectNode(
            node_id="effect",
            event="effect_event",
            timestamp=datetime.now(),
            probability=0.8,
            confidence=0.9
        )
        
        self.graph.add_node(cause_node)
        self.graph.add_node(effect_node)
        
        # Add relationship
        self.graph.add_cause_effect("cause", "effect", 0.9)
        
        self.assertIn(("cause", "effect"), self.graph.graph.edges)
        self.assertEqual(self.graph.edge_weights[("cause", "effect")], 0.9)
    
    def test_get_causal_chain(self):
        """Test getting causal chains"""
        # Create a chain: A -> B -> C
        nodes = []
        for i, name in enumerate(["A", "B", "C"]):
            node = CauseEffectNode(
                node_id=name,
                event=f"event_{name}",
                timestamp=datetime.now(),
                probability=0.8,
                confidence=0.9
            )
            self.graph.add_node(node)
            nodes.append(node)
        
        # Add relationships
        self.graph.add_cause_effect("A", "B")
        self.graph.add_cause_effect("B", "C")
        
        # Get causal chain
        chain = self.graph.get_causal_chain("A", max_depth=5)
        self.assertEqual(chain, ["A", "B", "C"])
    
    def test_get_root_causes(self):
        """Test finding root causes"""
        # Create nodes: A -> B -> C, D -> B
        for name in ["A", "B", "C", "D"]:
            node = CauseEffectNode(
                node_id=name,
                event=f"event_{name}",
                timestamp=datetime.now(),
                probability=0.8,
                confidence=0.9
            )
            self.graph.add_node(node)
        
        # Add relationships
        self.graph.add_cause_effect("A", "B")
        self.graph.add_cause_effect("B", "C")
        self.graph.add_cause_effect("D", "B")
        
        # Find root causes for C
        root_causes = self.graph.get_root_causes("C")
        self.assertIn("A", root_causes)
        self.assertIn("D", root_causes)

class TestQuantumRandomness(unittest.TestCase):
    """Test QuantumRandomness functionality"""
    
    def setUp(self):
        self.quantum_rand = QuantumRandomness(use_real_quantum=False)
    
    def test_mock_quantum_seed_generation(self):
        """Test mock quantum seed generation"""
        seed1 = self.quantum_rand.generate_quantum_seed()
        seed2 = self.quantum_rand.generate_quantum_seed()
        
        self.assertIsInstance(seed1, int)
        self.assertIsInstance(seed2, int)
        self.assertNotEqual(seed1, seed2)  # Should be different
        self.assertEqual(len(self.quantum_rand.seed_history), 2)
    
    def test_seed_history(self):
        """Test seed history tracking"""
        seeds = []
        for _ in range(5):
            seed = self.quantum_rand.generate_quantum_seed()
            seeds.append(seed)
        
        history = self.quantum_rand.get_seed_history()
        self.assertEqual(len(history), 5)
        self.assertEqual(history, seeds)
    
    @patch('core.oracle_engine.QuantumRandomness._generate_real_quantum_seed')
    def test_real_quantum_fallback(self, mock_real_quantum):
        """Test fallback to mock quantum when real quantum fails"""
        mock_real_quantum.side_effect = Exception("Quantum backend unavailable")
        
        quantum_rand = QuantumRandomness(use_real_quantum=True)
        seed = quantum_rand.generate_quantum_seed()
        
        self.assertIsInstance(seed, int)
        self.assertEqual(len(quantum_rand.seed_history), 1)

class TestOracleEngine(unittest.TestCase):
    """Test OracleEngine functionality"""
    
    def setUp(self):
        self.oracle = OracleEngine(
            prediction_horizon=100,
            confidence_threshold=0.8,
            max_parallel_predictions=5,
            use_quantum_randomness=False
        )
    
    def test_oracle_initialization(self):
        """Test Oracle Engine initialization"""
        self.assertIsNotNone(self.oracle.oracle_network)
        self.assertIsNotNone(self.oracle.quantum_randomness)
        self.assertIsNotNone(self.oracle.cause_effect_graph)
        self.assertGreater(len(self.oracle.timelines), 0)
        self.assertIsNotNone(self.oracle.prediction_lock)
        self.assertIsNotNone(self.oracle.timeline_lock)
    
    def test_timeline_fork_creation(self):
        """Test creating timeline forks"""
        fork = self.oracle.create_timeline_fork(
            parent_timeline="main",
            fork_id="test_fork",
            probability=0.6,
            weight=1.5
        )
        
        self.assertIsNotNone(fork)
        self.assertIn("test_fork", self.oracle.timelines)
        self.assertEqual(fork.parent_branch, "main")
        self.assertEqual(fork.weight, 1.5)
    
    def test_event_prediction(self):
        """Test basic event prediction"""
        prediction = self.oracle.predict_event(
            "Test event prediction",
            PredictionType.OMNI
        )
        
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.target_event, "Test event prediction")
        self.assertEqual(prediction.prediction_type, PredictionType.OMNI)
        self.assertGreaterEqual(prediction.probability, 0.0)
        self.assertLessEqual(prediction.probability, 1.0)
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)
        self.assertIn(prediction.timeline, self.oracle.timelines)
    
    def test_quantum_prediction(self):
        """Test quantum prediction with seed"""
        prediction = self.oracle.predict_event(
            "Quantum test event",
            PredictionType.QUANTUM,
            use_quantum_seed=True
        )
        
        self.assertIsNotNone(prediction.quantum_seed)
        self.assertEqual(prediction.prediction_type, PredictionType.QUANTUM)
    
    def test_parallel_predictions(self):
        """Test parallel event predictions"""
        events = [
            ("Event 1", PredictionType.TEMPORAL),
            ("Event 2", PredictionType.PROBABILISTIC),
            ("Event 3", PredictionType.CAUSAL)
        ]
        
        predictions = self.oracle.predict_parallel_events(events)
        
        self.assertEqual(len(predictions), 3)
        for i, (event_desc, pred_type) in enumerate(events):
            self.assertEqual(predictions[i].target_event, event_desc)
            self.assertEqual(predictions[i].prediction_type, pred_type)
    
    def test_cause_effect_relationship(self):
        """Test adding cause-effect relationships"""
        success = self.oracle.add_cause_effect_relationship(
            "Rain causes wet ground",
            "Wet ground causes slippery surface",
            weight=0.9
        )
        
        self.assertTrue(success)
        self.assertGreater(len(self.oracle.cause_effect_graph.nodes), 0)
        self.assertGreater(len(self.oracle.cause_effect_graph.graph.edges), 0)
    
    def test_causal_chain_prediction(self):
        """Test causal chain prediction"""
        chain = self.oracle.predict_causal_chain(
            "Initial event",
            chain_length=3
        )
        
        self.assertEqual(len(chain), 3)
        for prediction in chain:
            self.assertEqual(prediction.prediction_type, PredictionType.CAUSAL)
    
    def test_multiple_timeline_predictions(self):
        """Test predictions across multiple timelines"""
        predictions = self.oracle.predict_multiple_timelines(
            "Multi-timeline event",
            num_timelines=3
        )
        
        self.assertEqual(len(predictions), 3)
        timelines = set(pred.timeline for pred in predictions)
        self.assertEqual(len(timelines), 3)  # Should be different timelines
    
    def test_probability_distribution_analysis(self):
        """Test probability distribution analysis"""
        analysis = self.oracle.analyze_probability_distribution(
            "Distribution test event",
            num_samples=10
        )
        
        self.assertIn("mean_probability", analysis)
        self.assertIn("std_probability", analysis)
        self.assertIn("mean_confidence", analysis)
        self.assertIn("std_confidence", analysis)
        self.assertIn("distribution", analysis)
    
    def test_quantum_predictions(self):
        """Test quantum superposition predictions"""
        predictions = self.oracle.quantum_predict(
            "Quantum superposition event",
            superposition_states=3
        )
        
        self.assertEqual(len(predictions), 3)
        for prediction in predictions:
            self.assertEqual(prediction.prediction_type, PredictionType.QUANTUM)
            self.assertIsNotNone(prediction.quantum_seed)
    
    def test_fractal_predictions(self):
        """Test fractal predictions"""
        predictions = self.oracle.fractal_predict(
            "Fractal event",
            fractal_depth=3
        )
        
        self.assertEqual(len(predictions), 3)
        for prediction in predictions:
            self.assertEqual(prediction.prediction_type, PredictionType.FRACTAL)
    
    def test_oracle_insights(self):
        """Test getting oracle insights"""
        # Make some predictions first
        self.oracle.predict_event("Test event 1")
        self.oracle.predict_event("Test event 2")
        
        insights = self.oracle.get_oracle_insights()
        
        self.assertIn("total_predictions", insights)
        self.assertIn("timeline_count", insights)
        self.assertIn("quantum_seeds_generated", insights)
        self.assertIn("cause_effect_nodes", insights)
        self.assertIn("average_confidence", insights)
        self.assertIn("average_probability", insights)
        self.assertIn("timeline_branches", insights)
    
    def test_prediction_save_load(self):
        """Test saving and loading predictions"""
        # Make some predictions
        self.oracle.predict_event("Save test event 1")
        self.oracle.predict_event("Save test event 2")
        
        # Save predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_path = f.name
        
        try:
            self.oracle.save_predictions(save_path)
            
            # Clear predictions
            original_count = len(self.oracle.prediction_history)
            self.oracle.prediction_history.clear()
            
            # Load predictions
            self.oracle.load_predictions(save_path)
            
            self.assertEqual(len(self.oracle.prediction_history), original_count)
            
        finally:
            os.unlink(save_path)
    
    def test_thread_safety(self):
        """Test thread safety of predictions"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_prediction():
            try:
                prediction = self.oracle.predict_event(f"Thread test {threading.current_thread().name}")
                results.append(prediction)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)
        
        # All predictions should have unique IDs
        prediction_ids = [pred.prediction_id for pred in results]
        self.assertEqual(len(prediction_ids), len(set(prediction_ids)))

if __name__ == '__main__':
    unittest.main() 