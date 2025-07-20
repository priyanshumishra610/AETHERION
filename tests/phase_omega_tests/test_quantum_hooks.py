"""
Unit tests for AETHERION Quantum Hooks
"""

import pytest
import time
import random
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.quantum_hooks import (
    QuantumHooks, QuantumConfig, QuantumBackend,
    MockQuantumBackend, MockQuantumCircuit,
    QiskitQuantumBackend
)


class TestQuantumConfig:
    """Test quantum configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = QuantumConfig()
        assert config.backend == QuantumBackend.MOCK
        assert config.shots == 1024
        assert config.max_qubits == 5
        assert config.noise_model is True
        assert config.optimization_level == 1
        assert config.timeout == 30
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = QuantumConfig(
            backend=QuantumBackend.QISKIT,
            shots=2048,
            max_qubits=10,
            noise_model=False,
            optimization_level=2,
            timeout=60
        )
        assert config.backend == QuantumBackend.QISKIT
        assert config.shots == 2048
        assert config.max_qubits == 10
        assert config.noise_model is False
        assert config.optimization_level == 2
        assert config.timeout == 60


class TestMockQuantumBackend:
    """Test mock quantum backend"""
    
    def test_initialization(self):
        """Test backend initialization"""
        config = QuantumConfig()
        backend = MockQuantumBackend(config)
        assert backend.config == config
        assert backend.rng is not None
    
    def test_create_quantum_circuit(self):
        """Test circuit creation"""
        config = QuantumConfig()
        backend = MockQuantumBackend(config)
        circuit = backend.create_quantum_circuit(3)
        assert isinstance(circuit, MockQuantumCircuit)
        assert circuit.num_qubits == 3
    
    def test_get_random_bits(self):
        """Test random bit generation"""
        config = QuantumConfig()
        backend = MockQuantumBackend(config)
        bits = backend.get_random_bits(10)
        assert len(bits) == 10
        assert all(bit in [0, 1] for bit in bits)
    
    def test_get_random_float(self):
        """Test random float generation"""
        config = QuantumConfig()
        backend = MockQuantumBackend(config)
        value = backend.get_random_float()
        assert 0.0 <= value <= 1.0


class TestMockQuantumCircuit:
    """Test mock quantum circuit"""
    
    def test_initialization(self):
        """Test circuit initialization"""
        rng = random.Random()
        circuit = MockQuantumCircuit(4, rng)
        assert circuit.num_qubits == 4
        assert circuit.rng == rng
        assert circuit.operations == []
        assert circuit.state == [0, 0, 0, 0]
    
    def test_hadamard_gate(self):
        """Test Hadamard gate"""
        rng = random.Random()
        circuit = MockQuantumCircuit(2, rng)
        circuit.h(0)
        assert ('h', 0) in circuit.operations
    
    def test_x_gate(self):
        """Test Pauli-X gate"""
        rng = random.Random()
        circuit = MockQuantumCircuit(2, rng)
        circuit.x(1)
        assert ('x', 1) in circuit.operations
    
    def test_cnot_gate(self):
        """Test CNOT gate"""
        rng = random.Random()
        circuit = MockQuantumCircuit(2, rng)
        circuit.cx(0, 1)
        assert ('cx', (0, 1)) in circuit.operations
    
    def test_measure_all(self):
        """Test measurement"""
        rng = random.Random()
        circuit = MockQuantumCircuit(2, rng)
        circuit.measure_all()
        assert ('measure', 'all') in circuit.operations
    
    def test_execute_circuit(self):
        """Test circuit execution"""
        rng = random.Random(42)  # Fixed seed for reproducibility
        circuit = MockQuantumCircuit(2, rng)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = circuit.execute(100)
        assert 'counts' in result
        assert 'shots' in result
        assert 'backend' in result
        assert result['shots'] == 100
        assert result['backend'] == 'mock'
        assert len(result['counts']) > 0


class TestQuantumHooks:
    """Test main quantum hooks interface"""
    
    def test_initialization_mock(self):
        """Test initialization with mock backend"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        assert isinstance(hooks.backend, MockQuantumBackend)
        assert hooks.entropy_pool == []
        assert hooks.entropy_index == 0
    
    @patch('core.quantum_hooks.QiskitQuantumBackend')
    def test_initialization_qiskit(self, mock_qiskit_backend):
        """Test initialization with Qiskit backend"""
        config = QuantumConfig(backend=QuantumBackend.QISKIT)
        hooks = QuantumHooks(config)
        assert hooks.config == config
    
    def test_generate_quantum_randomness(self):
        """Test quantum randomness generation"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        # Test normal size
        random_bytes = hooks.generate_quantum_randomness(256)
        assert len(random_bytes) == 32  # 256 bits = 32 bytes
        
        # Test size limit
        random_bytes = hooks.generate_quantum_randomness(2000)
        assert len(random_bytes) == 128  # Should be limited to 1024 bits = 128 bytes
    
    def test_get_quantum_seed(self):
        """Test quantum seed generation"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        seed = hooks.get_quantum_seed()
        assert isinstance(seed, int)
        assert seed > 0
    
    def test_entropy_pool(self):
        """Test entropy pool functionality"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        # Create entropy pool
        hooks.create_entropy_pool(100)
        assert len(hooks.entropy_pool) == 100
        assert hooks.entropy_index == 0
        
        # Get entropy from pool
        entropy1 = hooks.get_entropy_from_pool()
        entropy2 = hooks.get_entropy_from_pool()
        assert entropy1 != entropy2  # Should be different
        assert hooks.entropy_index == 2
    
    def test_quantum_random_float(self):
        """Test quantum random float generation"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        value = hooks.quantum_random_float()
        assert 0.0 <= value <= 1.0
    
    def test_quantum_random_int(self):
        """Test quantum random integer generation"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        # Test range
        value = hooks.quantum_random_int(1, 10)
        assert 1 <= value <= 10
        
        # Test single value
        value = hooks.quantum_random_int(5, 5)
        assert value == 5
    
    def test_quantum_random_choice(self):
        """Test quantum random choice"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        choices = ['a', 'b', 'c', 'd']
        choice = hooks.quantum_random_choice(choices)
        assert choice in choices
    
    def test_quantum_shuffle(self):
        """Test quantum shuffle"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        original = [1, 2, 3, 4, 5]
        shuffled = hooks.quantum_shuffle(original)
        
        assert len(shuffled) == len(original)
        assert set(shuffled) == set(original)
        # Note: With mock backend, shuffle might not actually change order due to fixed seed
    
    def test_bell_state(self):
        """Test Bell state creation"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        result = hooks.bell_state()
        assert 'counts' in result
        assert 'shots' in result
        assert 'backend' in result
    
    def test_quantum_fourier_transform(self):
        """Test quantum Fourier transform"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        result = hooks.quantum_fourier_transform(3)
        assert 'counts' in result
        assert 'shots' in result
        assert 'backend' in result
    
    def test_get_quantum_state(self):
        """Test quantum state retrieval"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        state = hooks.get_quantum_state()
        assert 'backend' in state
        assert 'entropy_pool_size' in state
        assert 'entropy_index' in state
        assert 'config' in state
        assert state['backend'] == 'mock'
    
    def test_reset_quantum_state(self):
        """Test quantum state reset"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        # Create some entropy
        hooks.create_entropy_pool(10)
        hooks.get_entropy_from_pool()
        
        # Reset
        hooks.reset_quantum_state()
        assert hooks.entropy_pool == []
        assert hooks.entropy_index == 0
    
    def test_validate_quantum_operation(self):
        """Test operation validation"""
        config = QuantumConfig(backend=QuantumBackend.MOCK)
        hooks = QuantumHooks(config)
        
        # Valid operations
        assert hooks.validate_quantum_operation("random_bits", {"num_bits": 256})
        assert hooks.validate_quantum_operation("entropy_pool", {"size": 1000})
        assert hooks.validate_quantum_operation("quantum_circuit", {"num_qubits": 3})
        
        # Invalid operations
        assert not hooks.validate_quantum_operation("random_bits", {"num_bits": 2000})
        assert not hooks.validate_quantum_operation("entropy_pool", {"size": 50})
        assert not hooks.validate_quantum_operation("quantum_circuit", {"num_qubits": 10})


class TestQiskitQuantumBackend:
    """Test Qiskit quantum backend (mocked)"""
    
    @patch('core.quantum_hooks.importlib.import_module')
    def test_initialization_no_qiskit(self, mock_import):
        """Test initialization when Qiskit is not available"""
        mock_import.side_effect = ImportError("No module named 'qiskit'")
        
        config = QuantumConfig(backend=QuantumBackend.QISKIT)
        backend = QiskitQuantumBackend(config)
        
        assert backend.qiskit_available is False
        assert backend.backend is None
    
    @patch('core.quantum_hooks.importlib.import_module')
    def test_initialization_with_qiskit(self, mock_import):
        """Test initialization when Qiskit is available"""
        # Mock successful import
        mock_qiskit = Mock()
        mock_import.return_value = mock_qiskit
        
        config = QuantumConfig(backend=QuantumBackend.QISKIT)
        backend = QiskitQuantumBackend(config)
        
        # Should attempt to import qiskit
        mock_import.assert_called_with('qiskit')


if __name__ == "__main__":
    pytest.main([__file__]) 