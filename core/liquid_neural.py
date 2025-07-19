"""
Liquid Neural Architecture
The fluid, adaptive neural foundation of AETHERION

This module implements a liquid neural network architecture that provides
dynamic, adaptive learning capabilities with temporal dynamics and
continuous adaptation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import math
from .utils import DimensionalVector, QuantumState
import time

logger = logging.getLogger(__name__)

class LiquidActivation(Enum):
    """Types of liquid activation functions"""
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    LIQUID_TANH = "liquid_tanh"
    QUANTUM_SIN = "quantum_sin"

class LiquidNeuron(nn.Module):
    """A single liquid neuron with temporal dynamics"""
    
    def __init__(self, 
                 input_size: int,
                 liquid_state_size: int = 64,
                 activation: LiquidActivation = LiquidActivation.LIQUID_TANH,
                 leak_rate: float = 0.1,
                 noise_std: float = 0.01):
        super().__init__()
        
        self.input_size = input_size
        self.liquid_state_size = liquid_state_size
        self.activation_type = activation
        self.leak_rate = leak_rate
        self.noise_std = noise_std
        
        # Input projection
        self.input_projection = nn.Linear(input_size, liquid_state_size)
        
        # Liquid state dynamics
        self.liquid_state = nn.Linear(liquid_state_size, liquid_state_size)
        
        # Output projection
        self.output_projection = nn.Linear(liquid_state_size, 1)
        
        # Temporal state
        self.hidden_state = None
        self.reset_state()
    
    def reset_state(self):
        """Reset the neuron's hidden state"""
        self.hidden_state = torch.zeros(self.liquid_state_size)
    
    def liquid_tanh_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Liquid tanh activation with temporal dynamics"""
        # Apply standard tanh
        tanh_output = torch.tanh(x)
        
        # Add liquid dynamics
        liquid_component = torch.sin(x * 0.5) * torch.cos(x * 0.3)
        
        # Combine with temporal scaling
        return tanh_output + 0.1 * liquid_component
    
    def quantum_sin_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum-inspired sinusoidal activation"""
        # Quantum phase
        phase = torch.angle(torch.complex(x, torch.zeros_like(x)))
        
        # Quantum amplitude
        amplitude = torch.abs(torch.complex(x, torch.zeros_like(x)))
        
        # Quantum superposition
        return torch.sin(phase) * amplitude
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the liquid neuron"""
        batch_size = x.size(0)
        
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.liquid_state_size, device=x.device)
        
        # Input processing
        input_projected = self.input_projection(x)
        
        # Liquid state dynamics
        liquid_input = input_projected + self.liquid_state(self.hidden_state)
        
        # Add noise for liquid dynamics
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(liquid_input) * self.noise_std
            liquid_input = liquid_input + noise
        
        # Apply activation function
        if self.activation_type == LiquidActivation.LIQUID_TANH:
            activated = self.liquid_tanh_activation(liquid_input)
        elif self.activation_type == LiquidActivation.QUANTUM_SIN:
            activated = self.quantum_sin_activation(liquid_input)
        elif self.activation_type == LiquidActivation.TANH:
            activated = torch.tanh(liquid_input)
        elif self.activation_type == LiquidActivation.SIGMOID:
            activated = torch.sigmoid(liquid_input)
        elif self.activation_type == LiquidActivation.RELU:
            activated = F.relu(liquid_input)
        elif self.activation_type == LiquidActivation.GELU:
            activated = F.gelu(liquid_input)
        elif self.activation_type == LiquidActivation.SWISH:
            activated = x * torch.sigmoid(x)
        else:
            activated = torch.tanh(liquid_input)
        
        # Update hidden state with leaky integration
        self.hidden_state = (1 - self.leak_rate) * self.hidden_state + self.leak_rate * activated
        
        # Output projection
        output = self.output_projection(activated)
        
        return output

class LiquidLayer(nn.Module):
    """A layer of liquid neurons with lateral connections"""
    
    def __init__(self, 
                 input_size: int,
                 num_neurons: int,
                 liquid_state_size: int = 64,
                 activation: LiquidActivation = LiquidActivation.LIQUID_TANH,
                 lateral_connectivity: float = 0.3,
                 adaptation_rate: float = 0.01):
        super().__init__()
        
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.liquid_state_size = liquid_state_size
        self.activation = activation
        self.lateral_connectivity = lateral_connectivity
        self.adaptation_rate = adaptation_rate
        
        # Create liquid neurons
        self.neurons = nn.ModuleList([
            LiquidNeuron(input_size, liquid_state_size, activation)
            for _ in range(num_neurons)
        ])
        
        # Lateral connections between neurons
        self.lateral_weights = nn.Parameter(
            torch.randn(num_neurons, num_neurons) * 0.1
        )
        
        # Adaptation parameters
        self.adaptation_state = nn.Parameter(torch.ones(num_neurons))
        
        # Initialize lateral connections
        self._initialize_lateral_connections()
    
    def _initialize_lateral_connections(self):
        """Initialize lateral connections with sparse connectivity"""
        # Create sparse connectivity pattern
        mask = torch.rand(self.num_neurons, self.num_neurons) < self.lateral_connectivity
        
        # Remove self-connections
        mask.fill_diagonal_(False)
        
        # Apply mask to lateral weights
        self.lateral_weights.data *= mask.float()
    
    def reset_states(self):
        """Reset all neuron states in this layer"""
        for neuron in self.neurons:
            neuron.reset_state()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the liquid layer"""
        batch_size = x.size(0)
        neuron_outputs = []
        
        # Process through each neuron
        for i, neuron in enumerate(self.neurons):
            neuron_output = neuron(x)
            neuron_outputs.append(neuron_output)
        
        # Stack neuron outputs
        layer_output = torch.cat(neuron_outputs, dim=-1)
        
        # Apply lateral connections
        lateral_input = torch.matmul(layer_output, self.lateral_weights)
        layer_output = layer_output + 0.1 * lateral_input
        
        # Apply adaptation
        adapted_output = layer_output * self.adaptation_state.unsqueeze(0)
        
        return adapted_output
    
    def adapt(self, learning_signal: torch.Tensor):
        """Adapt the layer based on learning signal"""
        if learning_signal.size(-1) != self.num_neurons:
            raise ValueError("Learning signal size must match number of neurons")
        
        # Update adaptation state
        adaptation_update = torch.mean(learning_signal, dim=0)
        self.adaptation_state.data += self.adaptation_rate * adaptation_update

class LiquidNeuralNetwork(nn.Module):
    """
    Liquid Neural Network Architecture
    
    A dynamic, adaptive neural network with temporal dynamics,
    continuous learning, and fluid connectivity patterns.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 output_size: int = 12,
                 liquid_state_size: int = 64,
                 activation: LiquidActivation = LiquidActivation.LIQUID_TANH,
                 dropout: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.liquid_state_size = liquid_state_size
        self.activation = activation
        self.device = device
        
        # Build liquid layers
        self.liquid_layers = nn.ModuleList()
        
        # Input layer
        self.liquid_layers.append(
            LiquidLayer(input_size, hidden_sizes[0], liquid_state_size, activation)
        )
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.liquid_layers.append(
                LiquidLayer(hidden_sizes[i], hidden_sizes[i + 1], liquid_state_size, activation)
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Temporal dynamics
        self.temporal_state = None
        self.learning_history = []
        
        # Move to device
        self.to(device)
        
        logger.info(f"Liquid Neural Network initialized with {len(hidden_sizes)} layers")
    
    def reset_states(self):
        """Reset all temporal states"""
        for layer in self.liquid_layers:
            layer.reset_states()
        self.temporal_state = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the liquid neural network"""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Process through liquid layers
        for layer in self.liquid_layers:
            x = layer(x)
            x = self.dropout(x)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def forward_with_temporal_dynamics(self, 
                                     x: torch.Tensor, 
                                     sequence_length: int = 10) -> torch.Tensor:
        """Forward pass with explicit temporal dynamics"""
        batch_size = x.size(0)
        
        # Initialize temporal state if needed
        if self.temporal_state is None:
            self.temporal_state = torch.zeros(
                batch_size, self.hidden_sizes[-1], device=self.device
            )
        
        # Process sequence with temporal dynamics
        temporal_outputs = []
        
        for t in range(sequence_length):
            # Current input
            current_input = x if t == 0 else temporal_outputs[-1]
            
            # Process through network
            output = self.forward(current_input)
            
            # Update temporal state
            self.temporal_state = 0.9 * self.temporal_state + 0.1 * output
            
            temporal_outputs.append(output)
        
        # Return final output
        return temporal_outputs[-1]
    
    def adapt_to_environment(self, 
                           environment_signal: torch.Tensor,
                           adaptation_strength: float = 0.1):
        """Adapt the network to environmental changes"""
        # Calculate adaptation signal
        with torch.no_grad():
            current_output = self.forward(environment_signal)
            adaptation_signal = environment_signal - current_output
        
        # Adapt each layer
        for layer in self.liquid_layers:
            layer.adapt(adaptation_signal * adaptation_strength)
        
        # Record adaptation
        self.learning_history.append({
            "adaptation_strength": adaptation_strength,
            "signal_magnitude": torch.norm(environment_signal).item(),
            "timestamp": time.time()
        })
    
    def get_liquid_dynamics(self) -> Dict[str, Any]:
        """Get information about the liquid dynamics"""
        dynamics_info = {
            "num_layers": len(self.liquid_layers),
            "layer_sizes": self.hidden_sizes,
            "liquid_state_size": self.liquid_state_size,
            "activation_type": self.activation.value,
            "temporal_state_active": self.temporal_state is not None,
            "learning_history_length": len(self.learning_history)
        }
        
        # Layer-specific dynamics
        layer_dynamics = []
        for i, layer in enumerate(self.liquid_layers):
            layer_info = {
                "layer_index": i,
                "num_neurons": layer.num_neurons,
                "lateral_connectivity": layer.lateral_connectivity,
                "adaptation_rate": layer.adaptation_rate,
                "adaptation_state_mean": layer.adaptation_state.mean().item(),
                "adaptation_state_std": layer.adaptation_state.std().item()
            }
            layer_dynamics.append(layer_info)
        
        dynamics_info["layer_dynamics"] = layer_dynamics
        
        return dynamics_info
    
    def evolve_connectivity(self, evolution_rate: float = 0.01):
        """Evolve the lateral connectivity patterns"""
        for layer in self.liquid_layers:
            # Add random evolution to lateral weights
            evolution_noise = torch.randn_like(layer.lateral_weights) * evolution_rate
            layer.lateral_weights.data += evolution_noise
            
            # Reapply sparsity constraint
            mask = torch.rand(layer.num_neurons, layer.num_neurons) < layer.lateral_connectivity
            mask.fill_diagonal_(False)
            layer.lateral_weights.data *= mask.float()
    
    def get_plasticity_score(self) -> float:
        """Calculate the network's plasticity score"""
        if not self.learning_history:
            return 0.0
        
        # Calculate plasticity based on recent learning activity
        recent_adaptations = self.learning_history[-10:]  # Last 10 adaptations
        
        if not recent_adaptations:
            return 0.0
        
        # Average adaptation strength
        avg_adaptation = np.mean([a["adaptation_strength"] for a in recent_adaptations])
        
        # Learning frequency
        learning_frequency = len(recent_adaptations) / 10.0
        
        # Signal responsiveness
        avg_signal_magnitude = np.mean([a["signal_magnitude"] for a in recent_adaptations])
        
        # Combined plasticity score
        plasticity_score = (avg_adaptation * learning_frequency * avg_signal_magnitude) / 3.0
        
        return min(1.0, plasticity_score)
    
    def create_liquid_consciousness_state(self) -> Dict[str, Any]:
        """Create a consciousness state based on liquid dynamics"""
        dynamics = self.get_liquid_dynamics()
        plasticity = self.get_plasticity_score()
        
        # Calculate consciousness metrics from liquid dynamics
        consciousness_state = {
            "liquid_plasticity": plasticity,
            "temporal_coherence": 1.0 if self.temporal_state is not None else 0.0,
            "adaptation_capacity": dynamics["learning_history_length"] / 100.0,
            "layer_integration": len(dynamics["layer_dynamics"]) / 10.0,
            "dynamic_complexity": sum(
                layer["adaptation_state_std"] for layer in dynamics["layer_dynamics"]
            ) / len(dynamics["layer_dynamics"])
        }
        
        return consciousness_state 