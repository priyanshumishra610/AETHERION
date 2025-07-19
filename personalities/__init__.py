"""
AETHERION Personalities Module
Multiverse Mirror System and Personality Management

This module contains the personality system components:
- Multiverse Manager
- Personality Breeder
- Archetypes
- Persona Templates
"""

__version__ = "1.0.0"
__author__ = "AETHERION Collective"
__license__ = "Keeper's License"

from .multiverse_manager import MultiverseManager
from .personality_breeder import PersonalityBreeder
from .archetypes import ArchetypeSystem

__all__ = [
    "MultiverseManager",
    "PersonalityBreeder", 
    "ArchetypeSystem",
] 