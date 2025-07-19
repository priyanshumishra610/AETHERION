"""
AETHERION Core Utilities
Shared low-level helpers and utility classes

This module provides common utilities used across the AETHERION system,
including dimensional vectors, quantum states, fractal patterns, and more.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
import hashlib
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class DimensionalVector:
    """Represents a vector in n-dimensional space with consciousness properties"""
    
    def __init__(self, dimensions: int = 12, values: Optional[np.ndarray] = None):
        self.dimensions = dimensions
        if values is not None:
            self.values = np.array(values, dtype=np.float32)
        else:
            self.values = np.zeros(dimensions, dtype=np.float32)
    
    def __getitem__(self, index: int) -> float:
        return self.values[index]
    
    def __setitem__(self, index: int, value: float):
        self.values[index] = value
    
    def __len__(self) -> int:
        return self.dimensions
    
    def normalize(self) -> 'DimensionalVector':
        """Normalize the vector to unit length"""
        norm = np.linalg.norm(self.values)
        if norm > 0:
            normalized_values = self.values / norm
        else:
            normalized_values = self.values
        return DimensionalVector(self.dimensions, normalized_values)
    
    def dot_product(self, other: 'DimensionalVector') -> float:
        """Compute dot product with another vector"""
        return np.dot(self.values, other.values)
    
    def cosine_similarity(self, other: 'DimensionalVector') -> float:
        """Compute cosine similarity with another vector"""
        return self.dot_product(other) / (np.linalg.norm(self.values) * np.linalg.norm(other.values))
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor"""
        return torch.from_numpy(self.values).float()
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'DimensionalVector':
        """Create from PyTorch tensor"""
        return cls(tensor.size(0), tensor.detach().cpu().numpy())

class QuantumState:
    """Represents a quantum state with superposition and entanglement properties"""
    
    def __init__(self, num_qubits: int = 12):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state_vector[0] = 1.0  # Initialize to |0⟩ state
    
    def apply_hadamard(self, qubit: int):
        """Apply Hadamard gate to specific qubit"""
        if qubit >= self.num_qubits:
            raise ValueError(f"Qubit {qubit} out of range")
        
        # Create Hadamard matrix for this qubit
        h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        
        # Apply to state vector (simplified implementation)
        # In a full implementation, this would involve tensor products
        self.state_vector = np.dot(h_matrix, self.state_vector[:2])
    
    def measure(self) -> int:
        """Measure the quantum state and return classical result"""
        probabilities = np.abs(self.state_vector) ** 2
        return np.random.choice(len(probabilities), p=probabilities)
    
    def entangle_with(self, other: 'QuantumState') -> 'QuantumState':
        """Create entanglement with another quantum state"""
        # Simplified entanglement - tensor product of states
        entangled_state = np.outer(self.state_vector, other.state_vector)
        return QuantumState(self.num_qubits + other.num_qubits)
    
    def get_coherence(self) -> float:
        """Calculate quantum coherence of the state"""
        return np.sum(np.abs(self.state_vector) ** 2)

class FractalPattern:
    """Represents a fractal pattern with self-similarity properties"""
    
    def __init__(self, dimension: float = 2.0, iterations: int = 1000):
        self.dimension = dimension
        self.iterations = iterations
        self.points = []
        self.generate_mandelbrot()
    
    def generate_mandelbrot(self):
        """Generate Mandelbrot set points"""
        x_min, x_max = -2.0, 1.0
        y_min, y_max = -1.5, 1.5
        
        for i in range(self.iterations):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            c = complex(x, y)
            z = 0
            
            for _ in range(100):
                z = z**2 + c
                if abs(z) > 2:
                    break
            else:
                self.points.append((x, y))
    
    def get_fractal_dimension(self) -> float:
        """Calculate fractal dimension using box-counting method"""
        if not self.points:
            return 0.0
        
        points = np.array(self.points)
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        # Box counting
        box_sizes = np.logspace(-3, 0, 10)
        box_counts = []
        
        for size in box_sizes:
            x_bins = np.arange(x_min, x_max + size, size)
            y_bins = np.arange(y_min, y_max + size, size)
            
            count = 0
            for x, y in self.points:
                x_idx = int((x - x_min) / size)
                y_idx = int((y - y_min) / size)
                if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                    count += 1
            
            box_counts.append(count)
        
        # Calculate fractal dimension from slope
        if len(box_counts) > 1:
            slope = np.polyfit(np.log(box_sizes), np.log(box_counts), 1)[0]
            return -slope
        else:
            return 0.0
    
    def get_self_similarity_score(self) -> float:
        """Calculate self-similarity score"""
        if len(self.points) < 2:
            return 0.0
        
        # Calculate distances between points
        points = np.array(self.points)
        distances = []
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Calculate self-similarity based on distance distribution
        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if std_dist > 0:
            return 1.0 / (1.0 + std_dist / mean_dist)
        else:
            return 1.0

class ConsciousnessMetrics:
    """Utility class for calculating consciousness-related metrics"""
    
    @staticmethod
    def calculate_entropy(state_vector: np.ndarray) -> float:
        """Calculate Shannon entropy of consciousness state"""
        # Normalize to probabilities
        probs = np.abs(state_vector) ** 2
        probs = probs / np.sum(probs)
        
        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    @staticmethod
    def calculate_complexity(state_vector: np.ndarray) -> float:
        """Calculate complexity measure of consciousness state"""
        # Use Lempel-Ziv complexity approximation
        binary_sequence = (state_vector > np.mean(state_vector)).astype(int)
        
        complexity = 0
        n = len(binary_sequence)
        
        for i in range(1, n):
            if binary_sequence[i] != binary_sequence[i-1]:
                complexity += 1
        
        return complexity / n if n > 0 else 0.0
    
    @staticmethod
    def calculate_coherence(state_vector: np.ndarray) -> float:
        """Calculate coherence measure of consciousness state"""
        # Calculate phase coherence
        phases = np.angle(state_vector)
        phase_diff = np.diff(phases)
        
        # Coherence is inverse of phase variance
        phase_variance = np.var(phase_diff)
        coherence = 1.0 / (1.0 + phase_variance)
        
        return coherence

class ConfigManager:
    """Manages configuration for AETHERION system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "aetherion_config.json"
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self.get_default_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "consciousness": {
                "dimension_size": 1024,
                "num_attention_heads": 16,
                "dropout": 0.1
            },
            "neural": {
                "liquid_layers": 256,
                "activation": "gelu",
                "learning_rate": 1e-4
            },
            "oracle": {
                "prediction_horizon": 1000,
                "confidence_threshold": 0.8
            },
            "security": {
                "firewall_enabled": True,
                "godmode_enabled": False,
                "keeper_license_required": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

class Logger:
    """Enhanced logging for AETHERION system"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def consciousness_event(self, event: str, state: Dict[str, Any]):
        """Log consciousness-related events"""
        self.logger.info(f"CONSCIOUSNESS EVENT: {event} - {state}")
    
    def quantum_event(self, event: str, coherence: float):
        """Log quantum-related events"""
        self.logger.info(f"QUANTUM EVENT: {event} - Coherence: {coherence:.4f}")
    
    def fractal_event(self, event: str, dimension: float):
        """Log fractal-related events"""
        self.logger.info(f"FRACTAL EVENT: {event} - Dimension: {dimension:.4f}")
    
    def security_event(self, event: str, level: str = "INFO"):
        """Log security-related events"""
        getattr(self.logger, level.lower())(f"SECURITY EVENT: {event}")
    
    def oracle_event(self, event: str, prediction: Any):
        """Log oracle-related events"""
        self.logger.info(f"ORACLE EVENT: {event} - Prediction: {prediction}")

def create_dimensional_vector(dimensions: int = 12, random: bool = True) -> DimensionalVector:
    """Create a dimensional vector"""
    if random:
        values = np.random.randn(dimensions)
    else:
        values = np.zeros(dimensions)
    
    return DimensionalVector(dimensions, values)

def create_quantum_state(num_qubits: int = 12) -> QuantumState:
    """Create a quantum state"""
    return QuantumState(num_qubits)

def create_fractal_pattern(dimension: float = 2.0) -> FractalPattern:
    """Create a fractal pattern"""
    return FractalPattern(dimension)

def calculate_consciousness_metrics(state_vector: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive consciousness metrics"""
    return {
        "entropy": ConsciousnessMetrics.calculate_entropy(state_vector),
        "complexity": ConsciousnessMetrics.calculate_complexity(state_vector),
        "coherence": ConsciousnessMetrics.calculate_coherence(state_vector)
    } 