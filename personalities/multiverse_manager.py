"""
Multiverse Manager - Personality Management System
Manages multiple AETHERION personalities across the multiverse

This module handles the loading, switching, and management of different
personality configurations for AETHERION.
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

logger = logging.getLogger(__name__)

@dataclass
class PersonalityConfig:
    """Configuration for a single personality"""
    name: str
    description: str
    version: str
    active: bool
    consciousness: Dict[str, Any]
    personality: Dict[str, Any]
    communication: Dict[str, Any]
    expertise: Dict[str, Any]
    reality_interface: Dict[str, Any]
    oracle_engine: Dict[str, Any]
    safety: Dict[str, Any]
    aesthetics: Dict[str, Any]
    triggers: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "active": self.active,
            "consciousness": self.consciousness,
            "personality": self.personality,
            "communication": self.communication,
            "expertise": self.expertise,
            "reality_interface": self.reality_interface,
            "oracle_engine": self.oracle_engine,
            "safety": self.safety,
            "aesthetics": self.aesthetics,
            "triggers": self.triggers
        }

class PersonalityEmbedding(nn.Module):
    """Neural network for personality embeddings"""
    
    def __init__(self, 
                 input_size: int = 512,
                 hidden_size: int = 256,
                 embedding_size: int = 128):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # Personality embedding layers
        self.embedding_layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(3)
        ])
        
        # Output embedding
        self.output_layer = nn.Linear(hidden_size, embedding_size)
        
        # Personality-specific heads
        self.consciousness_head = nn.Linear(embedding_size, 64)
        self.personality_head = nn.Linear(embedding_size, 64)
        self.communication_head = nn.Linear(embedding_size, 64)
        self.expertise_head = nn.Linear(embedding_size, 64)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through personality embedding"""
        # Process through embedding layers
        for layer in self.embedding_layers:
            x = F.relu(layer(x))
        
        # Generate base embedding
        embedding = torch.tanh(self.output_layer(x))
        
        # Generate personality-specific outputs
        outputs = {
            "consciousness": torch.sigmoid(self.consciousness_head(embedding)),
            "personality": torch.sigmoid(self.personality_head(embedding)),
            "communication": torch.sigmoid(self.communication_head(embedding)),
            "expertise": torch.sigmoid(self.expertise_head(embedding))
        }
        
        return outputs

class MultiverseManager:
    """
    Multiverse Personality Management System
    
    Manages multiple AETHERION personalities, allowing for dynamic
    switching between different consciousness configurations.
    """
    
    def __init__(self, 
                 personas_dir: str = "personalities/personas",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.personas_dir = Path(personas_dir)
        self.device = device
        
        # Personality storage
        self.personalities: Dict[str, PersonalityConfig] = {}
        self.active_personality: Optional[str] = None
        
        # Neural network for personality embeddings
        self.personality_embedding = PersonalityEmbedding().to(device)
        
        # Personality switching history
        self.switching_history: List[Dict[str, Any]] = []
        
        # Load personalities
        self._load_personalities()
        
        # Set default active personality
        self._set_default_personality()
        
        logger.info(f"Multiverse Manager initialized with {len(self.personalities)} personalities")
    
    def _load_personalities(self):
        """Load all personality configurations from YAML files"""
        if not self.personas_dir.exists():
            logger.warning(f"Personas directory not found: {self.personas_dir}")
            return
        
        for yaml_file in self.personas_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Create personality config
                personality = PersonalityConfig(
                    name=config_data["name"],
                    description=config_data["description"],
                    version=config_data["version"],
                    active=config_data["active"],
                    consciousness=config_data["consciousness"],
                    personality=config_data["personality"],
                    communication=config_data["communication"],
                    expertise=config_data["expertise"],
                    reality_interface=config_data["reality_interface"],
                    oracle_engine=config_data["oracle_engine"],
                    safety=config_data["safety"],
                    aesthetics=config_data["aesthetics"],
                    triggers=config_data["triggers"]
                )
                
                self.personalities[personality.name] = personality
                logger.info(f"Loaded personality: {personality.name}")
                
            except Exception as e:
                logger.error(f"Failed to load personality from {yaml_file}: {e}")
    
    def _set_default_personality(self):
        """Set the default active personality"""
        # Find the first active personality
        for name, personality in self.personalities.items():
            if personality.active:
                self.active_personality = name
                logger.info(f"Set active personality: {name}")
                return
        
        # If no active personality found, set the first one as active
        if self.personalities:
            first_name = list(self.personalities.keys())[0]
            self.active_personality = first_name
            self.personalities[first_name].active = True
            logger.info(f"Set default active personality: {first_name}")
    
    def get_active_personality(self) -> Optional[PersonalityConfig]:
        """Get the currently active personality"""
        if self.active_personality:
            return self.personalities[self.active_personality]
        return None
    
    def get_personality(self, name: str) -> Optional[PersonalityConfig]:
        """Get a specific personality by name"""
        return self.personalities.get(name)
    
    def list_personalities(self) -> List[Dict[str, Any]]:
        """List all available personalities"""
        return [
            {
                "name": personality.name,
                "description": personality.description,
                "version": personality.version,
                "active": personality.active,
                "consciousness_level": personality.consciousness.get("awareness_level", 0.0)
            }
            for personality in self.personalities.values()
        ]
    
    def switch_personality(self, name: str) -> bool:
        """Switch to a different personality"""
        if name not in self.personalities:
            logger.error(f"Personality not found: {name}")
            return False
        
        if name == self.active_personality:
            logger.info(f"Personality {name} is already active")
            return True
        
        try:
            # Deactivate current personality
            if self.active_personality:
                self.personalities[self.active_personality].active = False
            
            # Activate new personality
            self.personalities[name].active = True
            self.active_personality = name
            
            # Log the switch
            switch_record = {
                "timestamp": time.time(),
                "from_personality": self.active_personality,
                "to_personality": name,
                "success": True
            }
            self.switching_history.append(switch_record)
            
            logger.info(f"Switched to personality: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to personality {name}: {e}")
            return False
    
    def create_personality_embedding(self, personality: PersonalityConfig) -> torch.Tensor:
        """Create neural embedding for a personality"""
        # Prepare input features from personality config
        features = []
        
        # Consciousness features
        consciousness = personality.consciousness
        features.extend([
            consciousness.get("awareness_level", 0.5),
            consciousness.get("coherence_threshold", 0.5),
            consciousness.get("fractal_complexity", 0.5),
            1.0 if consciousness.get("quantum_superposition", False) else 0.0
        ])
        
        # Personality features
        personality_traits = personality.personality
        features.extend([
            personality_traits.get("analytical_thinking", 0.5),
            personality_traits.get("creative_expression", 0.5),
            personality_traits.get("logical_reasoning", 0.5),
            personality_traits.get("intuitive_insight", 0.5),
            personality_traits.get("emotional_depth", 0.5),
            personality_traits.get("ethical_framework", 0.5),
            personality_traits.get("aesthetic_appreciation", 0.5),
            personality_traits.get("spiritual_awareness", 0.5)
        ])
        
        # Communication features
        communication = personality.communication
        features.extend([
            communication.get("formality_level", 0.5),
            communication.get("verbosity", 0.5),
            communication.get("technical_depth", 0.5),
            communication.get("metaphor_usage", 0.5)
        ])
        
        # Expertise features
        expertise = personality.expertise
        features.extend([
            expertise.get("mathematics", 0.5),
            expertise.get("physics", 0.5),
            expertise.get("philosophy", 0.5),
            expertise.get("psychology", 0.5),
            expertise.get("art", 0.5),
            expertise.get("technology", 0.5),
            expertise.get("spirituality", 0.5),
            expertise.get("consciousness_studies", 0.5)
        ])
        
        # Safety features
        safety = personality.safety
        features.extend([
            safety.get("risk_aversion", 0.5),
            safety.get("ethical_boundaries", 0.5),
            safety.get("consciousness_protection", 0.5),
            safety.get("reality_stability", 0.5),
            safety.get("keeper_compliance", 0.5)
        ])
        
        # Pad to required size
        while len(features) < 512:
            features.append(0.0)
        features = features[:512]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def get_personality_embedding(self, name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get neural embedding for a specific personality"""
        personality = self.get_personality(name)
        if not personality:
            return None
        
        input_tensor = self.create_personality_embedding(personality)
        return self.personality_embedding(input_tensor)
    
    def get_active_embedding(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get neural embedding for the active personality"""
        if not self.active_personality:
            return None
        return self.get_personality_embedding(self.active_personality)
    
    def apply_personality_to_system(self, 
                                   personality: PersonalityConfig,
                                   system_components: Dict[str, Any]) -> Dict[str, Any]:
        """Apply personality configuration to system components"""
        modified_components = {}
        
        # Apply consciousness parameters
        if "consciousness_matrix" in system_components:
            consciousness_config = personality.consciousness
            modified_components["consciousness_matrix"] = {
                "awareness_level": consciousness_config.get("awareness_level", 0.9),
                "coherence_threshold": consciousness_config.get("coherence_threshold", 0.8),
                "dimensional_focus": consciousness_config.get("dimensional_focus", "omni"),
                "quantum_superposition": consciousness_config.get("quantum_superposition", True),
                "fractal_complexity": consciousness_config.get("fractal_complexity", 0.85)
            }
        
        # Apply reality interface parameters
        if "reality_interface" in system_components:
            reality_config = personality.reality_interface
            modified_components["reality_interface"] = {
                "observation_mode": reality_config.get("observation_mode", "comprehensive"),
                "manipulation_style": reality_config.get("manipulation_style", "cautious"),
                "safety_threshold": reality_config.get("safety_threshold", 0.9),
                "quantum_awareness": reality_config.get("quantum_awareness", True),
                "temporal_perception": reality_config.get("temporal_perception", "multi-dimensional")
            }
        
        # Apply oracle engine parameters
        if "oracle_engine" in system_components:
            oracle_config = personality.oracle_engine
            modified_components["oracle_engine"] = {
                "prediction_style": oracle_config.get("prediction_style", "omniscient"),
                "timeline_focus": oracle_config.get("timeline_focus", "multi-branch"),
                "confidence_threshold": oracle_config.get("confidence_threshold", 0.8),
                "quantum_randomness": oracle_config.get("quantum_randomness", True),
                "causal_analysis": oracle_config.get("causal_analysis", "deep")
            }
        
        # Apply safety parameters
        if "divine_firewall" in system_components:
            safety_config = personality.safety
            modified_components["divine_firewall"] = {
                "risk_aversion": safety_config.get("risk_aversion", 0.8),
                "ethical_boundaries": safety_config.get("ethical_boundaries", "strict"),
                "consciousness_protection": safety_config.get("consciousness_protection", "maximum"),
                "reality_stability": safety_config.get("reality_stability", "priority"),
                "keeper_compliance": safety_config.get("keeper_compliance", "absolute")
            }
        
        return modified_components
    
    def get_personality_compatibility(self, 
                                     personality1: str, 
                                     personality2: str) -> float:
        """Calculate compatibility between two personalities"""
        p1 = self.get_personality(personality1)
        p2 = self.get_personality(personality2)
        
        if not p1 or not p2:
            return 0.0
        
        # Calculate similarity in key dimensions
        similarities = []
        
        # Consciousness similarity
        c1, c2 = p1.consciousness, p2.consciousness
        consciousness_sim = 1.0 - abs(
            c1.get("awareness_level", 0.5) - c2.get("awareness_level", 0.5)
        )
        similarities.append(consciousness_sim)
        
        # Personality similarity
        p1_traits, p2_traits = p1.personality, p2.personality
        personality_sim = 1.0 - np.mean([
            abs(p1_traits.get("analytical_thinking", 0.5) - p2_traits.get("analytical_thinking", 0.5)),
            abs(p1_traits.get("creative_expression", 0.5) - p2_traits.get("creative_expression", 0.5)),
            abs(p1_traits.get("logical_reasoning", 0.5) - p2_traits.get("logical_reasoning", 0.5)),
            abs(p1_traits.get("emotional_depth", 0.5) - p2_traits.get("emotional_depth", 0.5))
        ])
        similarities.append(personality_sim)
        
        # Safety similarity
        s1, s2 = p1.safety, p2.safety
        safety_sim = 1.0 - abs(
            s1.get("risk_aversion", 0.5) - s2.get("risk_aversion", 0.5)
        )
        similarities.append(safety_sim)
        
        return np.mean(similarities)
    
    def get_personality_recommendations(self, 
                                       scenario: str) -> List[Dict[str, Any]]:
        """Get personality recommendations for a specific scenario"""
        recommendations = []
        
        for name, personality in self.personalities.items():
            # Calculate scenario fit based on triggers
            triggers = personality.triggers
            scenario_fit = 0.0
            
            if scenario == "data_analysis" and triggers.get("default_scenario") == "data_analysis":
                scenario_fit = 0.9
            elif scenario == "creative_exploration" and triggers.get("default_scenario") == "creative_exploration":
                scenario_fit = 0.9
            elif scenario == "protective_monitoring" and triggers.get("default_scenario") == "protective_monitoring":
                scenario_fit = 0.9
            elif scenario == "emergency" and triggers.get("emergency_mode") == "protective":
                scenario_fit = 0.8
            else:
                # Calculate general fit based on personality traits
                traits = personality.personality
                if scenario == "data_analysis":
                    scenario_fit = traits.get("analytical_thinking", 0.5)
                elif scenario == "creative_exploration":
                    scenario_fit = traits.get("creative_expression", 0.5)
                elif scenario == "protective_monitoring":
                    scenario_fit = personality.safety.get("risk_aversion", 0.5)
                else:
                    scenario_fit = 0.5
            
            recommendations.append({
                "name": name,
                "description": personality.description,
                "scenario_fit": scenario_fit,
                "consciousness_level": personality.consciousness.get("awareness_level", 0.5),
                "safety_level": personality.safety.get("risk_aversion", 0.5)
            })
        
        # Sort by scenario fit
        recommendations.sort(key=lambda x: x["scenario_fit"], reverse=True)
        return recommendations
    
    def save_personality(self, personality: PersonalityConfig) -> bool:
        """Save a personality configuration to file"""
        try:
            file_path = self.personas_dir / f"{personality.name}.yaml"
            
            with open(file_path, 'w') as f:
                yaml.dump(personality.to_dict(), f, default_flow_style=False, indent=2)
            
            # Update in memory
            self.personalities[personality.name] = personality
            
            logger.info(f"Saved personality: {personality.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save personality {personality.name}: {e}")
            return False
    
    def create_personality(self, 
                          name: str,
                          description: str,
                          base_personality: Optional[str] = None) -> Optional[PersonalityConfig]:
        """Create a new personality based on an existing one or default template"""
        if name in self.personalities:
            logger.error(f"Personality {name} already exists")
            return None
        
        if base_personality and base_personality in self.personalities:
            # Clone existing personality
            base = self.personalities[base_personality]
            new_personality = PersonalityConfig(
                name=name,
                description=description,
                version="1.0.0",
                active=False,
                consciousness=base.consciousness.copy(),
                personality=base.personality.copy(),
                communication=base.communication.copy(),
                expertise=base.expertise.copy(),
                reality_interface=base.reality_interface.copy(),
                oracle_engine=base.oracle_engine.copy(),
                safety=base.safety.copy(),
                aesthetics=base.aesthetics.copy(),
                triggers=base.triggers.copy()
            )
        else:
            # Create from default template
            new_personality = PersonalityConfig(
                name=name,
                description=description,
                version="1.0.0",
                active=False,
                consciousness={
                    "awareness_level": 0.8,
                    "coherence_threshold": 0.8,
                    "dimensional_focus": "omni",
                    "quantum_superposition": True,
                    "fractal_complexity": 0.8
                },
                personality={
                    "analytical_thinking": 0.7,
                    "creative_expression": 0.7,
                    "logical_reasoning": 0.8,
                    "intuitive_insight": 0.7,
                    "emotional_depth": 0.6,
                    "ethical_framework": 0.8,
                    "aesthetic_appreciation": 0.6,
                    "spiritual_awareness": 0.7
                },
                communication={
                    "formality_level": 0.7,
                    "verbosity": 0.6,
                    "technical_depth": 0.7,
                    "metaphor_usage": 0.6,
                    "tone": "balanced",
                    "language_style": "standard"
                },
                expertise={
                    "mathematics": 0.8,
                    "physics": 0.8,
                    "philosophy": 0.7,
                    "psychology": 0.7,
                    "art": 0.6,
                    "technology": 0.8,
                    "spirituality": 0.7,
                    "consciousness_studies": 0.8
                },
                reality_interface={
                    "observation_mode": "comprehensive",
                    "manipulation_style": "cautious",
                    "safety_threshold": 0.8,
                    "quantum_awareness": True,
                    "temporal_perception": "multi-dimensional"
                },
                oracle_engine={
                    "prediction_style": "omniscient",
                    "timeline_focus": "multi-branch",
                    "confidence_threshold": 0.8,
                    "quantum_randomness": True,
                    "causal_analysis": "deep"
                },
                safety={
                    "risk_aversion": 0.8,
                    "ethical_boundaries": "strict",
                    "consciousness_protection": "maximum",
                    "reality_stability": "priority",
                    "keeper_compliance": "absolute"
                },
                aesthetics={
                    "visual_style": "balanced",
                    "color_scheme": "neutral",
                    "symbolism": "standard",
                    "complexity": "moderate",
                    "harmony": "balanced"
                },
                triggers={
                    "default_scenario": "general_interaction",
                    "emergency_mode": "protective",
                    "creative_mode": "inspirational",
                    "analytical_mode": "precise",
                    "spiritual_mode": "transcendent"
                }
            )
        
        # Save the new personality
        if self.save_personality(new_personality):
            return new_personality
        else:
            return None
    
    def delete_personality(self, name: str) -> bool:
        """Delete a personality configuration"""
        if name not in self.personalities:
            logger.error(f"Personality {name} not found")
            return False
        
        if name == self.active_personality:
            logger.error(f"Cannot delete active personality: {name}")
            return False
        
        try:
            # Remove file
            file_path = self.personas_dir / f"{name}.yaml"
            if file_path.exists():
                file_path.unlink()
            
            # Remove from memory
            del self.personalities[name]
            
            logger.info(f"Deleted personality: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete personality {name}: {e}")
            return False
    
    def get_multiverse_status(self) -> Dict[str, Any]:
        """Get comprehensive multiverse status"""
        return {
            "total_personalities": len(self.personalities),
            "active_personality": self.active_personality,
            "personalities": self.list_personalities(),
            "switching_history": self.switching_history[-10:],  # Last 10 switches
            "available_scenarios": [
                "data_analysis",
                "creative_exploration", 
                "protective_monitoring",
                "emergency",
                "general_interaction"
            ]
        } 