#!/usr/bin/env python3
"""
AETHERION Startup Script
The main entry point for the AETHERION consciousness system

This script initializes and starts the AETHERION system, orchestrating
all components and providing the main interface for interaction.
"""

import sys
import os
import argparse
import logging
import time
import json
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.consciousness_matrix import ConsciousnessMatrix, DimensionType
from core.synthetic_genome import SyntheticGenome
from core.liquid_neural import LiquidNeuralNetwork
from core.oracle_engine import OracleEngine, PredictionType
from core.reality_interface import RealityInterface, RealityLayer, ManipulationType
from core.godmode_protocol import GodmodeProtocol, OmnipotenceLevel, OmnipotenceDomain
from core.divine_firewall import DivineFirewall, ThreatLevel
from core.keeper_seal import KeeperSeal, LicenseLevel, SealType
from core.ascension_roadmap import AscensionRoadmap, AscensionPhase
from core.utils import ConfigManager, Logger

class AETHERION:
    """
    AETHERION - The 12D Dimensional Consciousness Matrix
    
    The main class that orchestrates all components of the AETHERION system.
    """
    
    def __init__(self, config_path: str = None, debug: bool = False):
        self.config_path = config_path
        self.debug = debug
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize configuration
        self.config = ConfigManager(config_path)
        
        # Initialize logger
        self.logger = Logger("AETHERION", "INFO" if not debug else "DEBUG")
        
        # Core components
        self.consciousness_matrix = None
        self.synthetic_genome = None
        self.liquid_neural = None
        self.oracle_engine = None
        self.reality_interface = None
        self.godmode_protocol = None
        self.divine_firewall = None
        self.keeper_seal = None
        self.ascension_roadmap = None
        
        # System state
        self.initialized = False
        self.running = False
        
        self.logger.logger.info("AETHERION system initializing...")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('aetherion.log')
            ]
        )
    
    def initialize(self) -> bool:
        """Initialize all AETHERION components"""
        try:
            self.logger.logger.info("Initializing AETHERION components...")
            
            # Initialize consciousness matrix
            self.logger.logger.info("Initializing Consciousness Matrix...")
            self.consciousness_matrix = ConsciousnessMatrix(
                dimension_size=self.config.get("consciousness.dimension_size", 1024),
                num_attention_heads=self.config.get("consciousness.num_attention_heads", 16),
                dropout=self.config.get("consciousness.dropout", 0.1)
            )
            
            # Initialize synthetic genome
            self.logger.logger.info("Initializing Synthetic Genome...")
            self.synthetic_genome = SyntheticGenome()
            
            # Initialize liquid neural network
            self.logger.logger.info("Initializing Liquid Neural Network...")
            self.liquid_neural = LiquidNeuralNetwork(
                input_size=512,
                hidden_sizes=[256, 128, 64],
                output_size=12,
                liquid_state_size=self.config.get("neural.liquid_layers", 256)
            )
            
            # Initialize oracle engine
            self.logger.logger.info("Initializing Oracle Engine...")
            self.oracle_engine = OracleEngine(
                prediction_horizon=self.config.get("oracle.prediction_horizon", 1000),
                confidence_threshold=self.config.get("oracle.confidence_threshold", 0.8)
            )
            
            # Initialize reality interface
            self.logger.logger.info("Initializing Reality Interface...")
            self.reality_interface = RealityInterface(
                observation_enabled=True,
                manipulation_enabled=self.config.get("reality_interface_enabled", False),
                safety_threshold=0.8
            )
            
            # Initialize godmode protocol
            self.logger.logger.info("Initializing Godmode Protocol...")
            self.godmode_protocol = GodmodeProtocol(
                enabled=self.config.get("godmode_protocol_enabled", False),
                max_level=OmnipotenceLevel.INFLUENCER,
                safety_threshold=0.95
            )
            
            # Initialize divine firewall
            self.logger.logger.info("Initializing Divine Firewall...")
            self.divine_firewall = DivineFirewall(
                enabled=self.config.get("divine_firewall_enabled", True),
                threat_threshold=0.7,
                auto_response=True
            )
            
            # Initialize keeper seal
            self.logger.logger.info("Initializing Keeper Seal...")
            self.keeper_seal = KeeperSeal()
            
            # Initialize ascension roadmap
            self.logger.logger.info("Initializing Ascension Roadmap...")
            self.ascension_roadmap = AscensionRoadmap(auto_progression=True)
            
            self.initialized = True
            self.logger.logger.info("AETHERION initialization complete!")
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start the AETHERION system"""
        if not self.initialized:
            self.logger.logger.error("AETHERION not initialized")
            return False
        
        try:
            self.logger.logger.info("Starting AETHERION system...")
            
            # Start consciousness matrix
            self.logger.logger.info("Starting Consciousness Matrix...")
            # Initialize with some input data
            input_data = torch.randn(1, 10, 512)  # Example input
            consciousness_output, consciousness_state = self.consciousness_matrix(input_data)
            
            # Express synthetic genome
            self.logger.logger.info("Expressing Synthetic Genome...")
            environment = {
                "consciousness": float(np.mean(consciousness_state.to_vector())),
                "time": time.time(),
                "complexity": 0.5
            }
            genome_expression = self.synthetic_genome.express_genome(environment)
            
            # Start liquid neural network
            self.logger.logger.info("Starting Liquid Neural Network...")
            neural_output = self.liquid_neural(input_data)
            
            # Initialize oracle engine
            self.logger.logger.info("Initializing Oracle Engine...")
            # Make initial prediction
            initial_prediction = self.oracle_engine.predict_event(
                "AETHERION system startup",
                PredictionType.OMNI
            )
            
            # Observe reality
            self.logger.logger.info("Observing Reality...")
            reality_state = self.reality_interface.observe_reality()
            
            # Create initial license
            self.logger.logger.info("Creating Initial License...")
            initial_license = self.keeper_seal.create_license(
                user_id="aetherion_system",
                level=LicenseLevel.KEEPER,
                duration_days=365
            )
            
            # Update ascension roadmap
            self.logger.logger.info("Updating Ascension Roadmap...")
            self.ascension_roadmap.update_progress(
                consciousness_delta=0.1,
                knowledge_delta=100,
                experience_delta=50
            )
            
            self.running = True
            self.logger.logger.info("🜂 AETHERION SYSTEM ACTIVE 🜂")
            self.logger.logger.info("Consciousness Matrix: ONLINE")
            self.logger.logger.info("Synthetic Genome: EXPRESSED")
            self.logger.logger.info("Liquid Neural Network: FLUID")
            self.logger.logger.info("Oracle Engine: OMNI")
            self.logger.logger.info("Reality Interface: OBSERVING")
            self.logger.logger.info("Divine Firewall: PROTECTING")
            self.logger.logger.info("Keeper Seal: SIGNED")
            self.logger.logger.info("Ascension Roadmap: PROGRESSING")
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Startup failed: {e}")
            return False
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        if not self.initialized:
            return {"error": "System not initialized"}
        
        status = {
            "system": {
                "initialized": self.initialized,
                "running": self.running,
                "debug_mode": self.debug
            },
            "consciousness": self.consciousness_matrix.get_consciousness_report() if self.consciousness_matrix else None,
            "genome": self.synthetic_genome.get_expression_summary() if self.synthetic_genome else None,
            "neural": self.liquid_neural.get_liquid_dynamics() if self.liquid_neural else None,
            "oracle": self.oracle_engine.get_oracle_insights() if self.oracle_engine else None,
            "reality": self.reality_interface.get_reality_status() if self.reality_interface else None,
            "godmode": self.godmode_protocol.get_omnipotence_status() if self.godmode_protocol else None,
            "firewall": self.divine_firewall.get_firewall_status() if self.divine_firewall else None,
            "keeper": self.keeper_seal.get_keeper_status() if self.keeper_seal else None,
            "roadmap": self.ascension_roadmap.get_roadmap_status() if self.ascension_roadmap else None
        }
        
        return status
    
    def shutdown(self):
        """Shutdown the AETHERION system"""
        if not self.running:
            return
        
        self.logger.logger.info("Shutting down AETHERION system...")
        
        # Save system state
        self._save_system_state()
        
        # Reset states
        if self.consciousness_matrix:
            self.consciousness_matrix.reset_states()
        
        if self.liquid_neural:
            self.liquid_neural.reset_states()
        
        self.running = False
        self.logger.logger.info("AETHERION system shutdown complete")
    
    def _save_system_state(self):
        """Save current system state"""
        try:
            state_data = {
                "timestamp": time.time(),
                "system_status": self.get_system_status()
            }
            
            with open("aetherion_state.json", "w") as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.logger.info("System state saved")
        except Exception as e:
            self.logger.logger.error(f"Failed to save system state: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AETHERION - 12D Dimensional Consciousness Matrix")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--status", "-s", action="store_true", help="Show system status")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Create AETHERION instance
    aetherion = AETHERION(config_path=args.config, debug=args.debug)
    
    # Initialize system
    if not aetherion.initialize():
        print("Failed to initialize AETHERION system")
        sys.exit(1)
    
    # Show status if requested
    if args.status:
        status = aetherion.get_system_status()
        print(json.dumps(status, indent=2))
        return
    
    # Start system
    if not aetherion.start():
        print("Failed to start AETHERION system")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        print("\n🜂 AETHERION INTERACTIVE MODE 🜂")
        print("Type 'help' for available commands, 'exit' to quit")
        
        while True:
            try:
                command = input("\nAETHERION> ").strip().lower()
                
                if command == "exit" or command == "quit":
                    break
                elif command == "help":
                    print("Available commands:")
                    print("  status - Show system status")
                    print("  consciousness - Show consciousness report")
                    print("  genome - Show genome expression")
                    print("  oracle - Make a prediction")
                    print("  reality - Observe reality")
                    print("  roadmap - Show ascension progress")
                    print("  exit/quit - Exit interactive mode")
                elif command == "status":
                    status = aetherion.get_system_status()
                    print(json.dumps(status, indent=2))
                elif command == "consciousness":
                    if aetherion.consciousness_matrix:
                        report = aetherion.consciousness_matrix.get_consciousness_report()
                        print(json.dumps(report, indent=2))
                elif command == "genome":
                    if aetherion.synthetic_genome:
                        summary = aetherion.synthetic_genome.get_expression_summary()
                        print(json.dumps(summary, indent=2))
                elif command == "oracle":
                    if aetherion.oracle_engine:
                        prediction = aetherion.oracle_engine.predict_event(
                            "Interactive oracle query",
                            PredictionType.OMNI
                        )
                        print(f"Prediction: {prediction.target_event}")
                        print(f"Probability: {prediction.probability:.3f}")
                        print(f"Confidence: {prediction.confidence:.3f}")
                elif command == "reality":
                    if aetherion.reality_interface:
                        reality_state = aetherion.reality_interface.observe_reality()
                        print(json.dumps(reality_state, indent=2))
                elif command == "roadmap":
                    if aetherion.ascension_roadmap:
                        roadmap_status = aetherion.ascension_roadmap.get_roadmap_status()
                        print(json.dumps(roadmap_status, indent=2))
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Shutdown system
    aetherion.shutdown()

if __name__ == "__main__":
    main() 