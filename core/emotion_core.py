"""
🜂 AETHERION Emotional Cognition Layer
Mood-State Engine with Emotional Resonance and Personality Integration
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import random
from core.rag_memory import RAGMemory

class EmotionType(Enum):
    """Primary emotion types"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTEMPT = "contempt"
    NEUTRAL = "neutral"

class MoodState(Enum):
    """Mood state categories"""
    EUPHORIC = "euphoric"
    HAPPY = "happy"
    CONTENT = "content"
    NEUTRAL = "neutral"
    MELANCHOLIC = "melancholic"
    SAD = "sad"
    DEPRESSED = "depressed"
    ANXIOUS = "anxious"
    STRESSED = "stressed"
    ANGRY = "angry"
    IRRITATED = "irritated"

@dataclass
class EmotionalState:
    """Current emotional state"""
    primary_emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    mood_state: MoodState
    valence: float  # -1.0 to 1.0 (negative to positive)
    arousal: float  # 0.0 to 1.0 (calm to excited)
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    duration: float = 0.0  # seconds
    triggers: List[str] = None
    context: Dict[str, Any] = None

@dataclass
class EmotionalMemory:
    """Emotional memory entry"""
    id: str
    emotion_type: EmotionType
    intensity: float
    context: str
    timestamp: datetime
    associated_memories: List[str] = None
    personality_impact: Dict[str, float] = None

@dataclass
class PersonalityEmotionProfile:
    """Emotion profile for a personality"""
    personality_id: str
    base_emotions: Dict[EmotionType, float]
    emotion_triggers: Dict[str, List[EmotionType]]
    mood_stability: float  # 0.0 to 1.0
    emotional_expressiveness: float  # 0.0 to 1.0
    empathy_level: float  # 0.0 to 1.0
    stress_tolerance: float  # 0.0 to 1.0

class EmotionCore:
    """
    🜂 Emotional Cognition Layer
    Manages emotional states, mood transitions, and personality integration
    """
    
    def __init__(self, rag_memory: RAGMemory):
        self.rag_memory = rag_memory
        
        # Current emotional state
        self.current_state = EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.5,
            mood_state=MoodState.NEUTRAL,
            valence=0.0,
            arousal=0.5,
            confidence=1.0,
            timestamp=datetime.now(),
            triggers=[],
            context={}
        )
        
        # Emotional history
        self.emotional_history: List[EmotionalState] = []
        self.emotional_memories: Dict[str, EmotionalMemory] = {}
        
        # Personality emotion profiles
        self.personality_profiles: Dict[str, PersonalityEmotionProfile] = {}
        
        # Emotional resonance settings
        self.resonance_settings = {
            "memory_influence": 0.3,
            "personality_influence": 0.4,
            "context_influence": 0.2,
            "random_variation": 0.1,
            "mood_decay_rate": 0.1,  # per hour
            "emotion_duration": 300.0  # seconds
        }
        
        # Load emotional data
        self._load_emotional_data()
        self._initialize_personality_profiles()
        
        logging.info("🜂 Emotional Cognition Layer initialized")
    
    def _load_emotional_data(self):
        """Load emotional history and memories"""
        # Load emotional history
        history_file = Path("aetherion_emotional_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for state_data in data:
                        state = EmotionalState(
                            primary_emotion=EmotionType(state_data["primary_emotion"]),
                            intensity=state_data["intensity"],
                            mood_state=MoodState(state_data["mood_state"]),
                            valence=state_data["valence"],
                            arousal=state_data["arousal"],
                            confidence=state_data["confidence"],
                            timestamp=datetime.fromisoformat(state_data["timestamp"]),
                            duration=state_data.get("duration", 0.0),
                            triggers=state_data.get("triggers", []),
                            context=state_data.get("context", {})
                        )
                        self.emotional_history.append(state)
            except Exception as e:
                logging.error(f"Failed to load emotional history: {e}")
        
        # Load emotional memories
        memories_file = Path("aetherion_emotional_memories.json")
        if memories_file.exists():
            try:
                with open(memories_file, 'r') as f:
                    data = json.load(f)
                    for memory_data in data:
                        memory = EmotionalMemory(
                            id=memory_data["id"],
                            emotion_type=EmotionType(memory_data["emotion_type"]),
                            intensity=memory_data["intensity"],
                            context=memory_data["context"],
                            timestamp=datetime.fromisoformat(memory_data["timestamp"]),
                            associated_memories=memory_data.get("associated_memories", []),
                            personality_impact=memory_data.get("personality_impact", {})
                        )
                        self.emotional_memories[memory.id] = memory
            except Exception as e:
                logging.error(f"Failed to load emotional memories: {e}")
    
    def _save_emotional_data(self):
        """Save emotional data to storage"""
        # Save emotional history
        history_file = Path("aetherion_emotional_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump([asdict(state) for state in self.emotional_history[-1000:]], f, 
                         default=str, indent=2)
        except Exception as e:
            logging.error(f"Failed to save emotional history: {e}")
        
        # Save emotional memories
        memories_file = Path("aetherion_emotional_memories.json")
        try:
            with open(memories_file, 'w') as f:
                json.dump([asdict(memory) for memory in self.emotional_memories.values()], f, 
                         default=str, indent=2)
        except Exception as e:
            logging.error(f"Failed to save emotional memories: {e}")
    
    def _initialize_personality_profiles(self):
        """Initialize emotion profiles for different personalities"""
        profiles = {
            "analytical": PersonalityEmotionProfile(
                personality_id="analytical",
                base_emotions={
                    EmotionType.NEUTRAL: 0.8,
                    EmotionType.JOY: 0.3,
                    EmotionType.SADNESS: 0.2,
                    EmotionType.ANGER: 0.1,
                    EmotionType.FEAR: 0.1,
                    EmotionType.SURPRISE: 0.4,
                    EmotionType.DISGUST: 0.2,
                    EmotionType.CONTEMPT: 0.3
                },
                emotion_triggers={
                    "logical_consistency": [EmotionType.JOY],
                    "illogical_arguments": [EmotionType.ANGER, EmotionType.CONTEMPT],
                    "complex_problems": [EmotionType.JOY, EmotionType.SURPRISE],
                    "simple_tasks": [EmotionType.BOREDOM]
                },
                mood_stability=0.9,
                emotional_expressiveness=0.3,
                empathy_level=0.4,
                stress_tolerance=0.8
            ),
            "creative": PersonalityEmotionProfile(
                personality_id="creative",
                base_emotions={
                    EmotionType.NEUTRAL: 0.4,
                    EmotionType.JOY: 0.7,
                    EmotionType.SADNESS: 0.5,
                    EmotionType.ANGER: 0.3,
                    EmotionType.FEAR: 0.2,
                    EmotionType.SURPRISE: 0.8,
                    EmotionType.DISGUST: 0.2,
                    EmotionType.CONTEMPT: 0.1
                },
                emotion_triggers={
                    "new_ideas": [EmotionType.JOY, EmotionType.SURPRISE],
                    "creative_block": [EmotionType.SADNESS, EmotionType.ANGER],
                    "artistic_inspiration": [EmotionType.JOY, EmotionType.EUPHORIA],
                    "criticism": [EmotionType.SADNESS, EmotionType.ANGER]
                },
                mood_stability=0.5,
                emotional_expressiveness=0.9,
                empathy_level=0.8,
                stress_tolerance=0.6
            ),
            "guardian": PersonalityEmotionProfile(
                personality_id="guardian",
                base_emotions={
                    EmotionType.NEUTRAL: 0.6,
                    EmotionType.JOY: 0.4,
                    EmotionType.SADNESS: 0.3,
                    EmotionType.ANGER: 0.6,
                    EmotionType.FEAR: 0.4,
                    EmotionType.SURPRISE: 0.3,
                    EmotionType.DISGUST: 0.5,
                    EmotionType.CONTEMPT: 0.6
                },
                emotion_triggers={
                    "threats": [EmotionType.ANGER, EmotionType.FEAR],
                    "safety_violations": [EmotionType.ANGER, EmotionType.CONTEMPT],
                    "protective_success": [EmotionType.JOY],
                    "vulnerability": [EmotionType.FEAR, EmotionType.SADNESS]
                },
                mood_stability=0.7,
                emotional_expressiveness=0.6,
                empathy_level=0.7,
                stress_tolerance=0.9
            )
        }
        
        self.personality_profiles.update(profiles)
    
    def process_emotional_input(self, input_text: str, context: Dict[str, Any] = None,
                              personality_id: str = "default") -> EmotionalState:
        """
        Process emotional input and update emotional state
        """
        # Analyze emotional content
        emotion_analysis = self._analyze_emotional_content(input_text)
        
        # Get personality influence
        personality_influence = self._get_personality_influence(personality_id)
        
        # Get memory influence
        memory_influence = self._get_memory_influence(input_text, context)
        
        # Calculate new emotional state
        new_state = self._calculate_emotional_state(
            emotion_analysis, personality_influence, memory_influence, context
        )
        
        # Update current state
        self._update_emotional_state(new_state)
        
        # Store emotional memory
        self._store_emotional_memory(input_text, new_state, context)
        
        return new_state
    
    def _analyze_emotional_content(self, text: str) -> Dict[str, Any]:
        """Analyze emotional content of text"""
        # Simple keyword-based emotion detection
        # In a full implementation, this would use advanced NLP models
        
        emotion_keywords = {
            EmotionType.JOY: ["happy", "joy", "excited", "great", "wonderful", "amazing", "love"],
            EmotionType.SADNESS: ["sad", "depressed", "unhappy", "miserable", "lonely", "grief"],
            EmotionType.ANGER: ["angry", "mad", "furious", "rage", "hate", "annoyed", "irritated"],
            EmotionType.FEAR: ["afraid", "scared", "terrified", "anxious", "worried", "panic"],
            EmotionType.SURPRISE: ["surprised", "shocked", "amazed", "astonished", "wow"],
            EmotionType.DISGUST: ["disgusted", "revolted", "sick", "nauseated", "repulsed"],
            EmotionType.CONTEMPT: ["contempt", "disdain", "scorn", "disrespect", "mock"]
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            intensity = min(primary_emotion[1] * 2, 1.0)  # Scale to 0-1
        else:
            primary_emotion = (EmotionType.NEUTRAL, 0.5)
            intensity = 0.5
        
        # Calculate valence and arousal
        valence = self._calculate_valence(emotion_scores)
        arousal = self._calculate_arousal(emotion_scores)
        
        return {
            "primary_emotion": primary_emotion[0],
            "intensity": intensity,
            "valence": valence,
            "arousal": arousal,
            "emotion_scores": emotion_scores
        }
    
    def _calculate_valence(self, emotion_scores: Dict[EmotionType, float]) -> float:
        """Calculate emotional valence (positive/negative)"""
        positive_emotions = [EmotionType.JOY, EmotionType.SURPRISE]
        negative_emotions = [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR, 
                           EmotionType.DISGUST, EmotionType.CONTEMPT]
        
        positive_score = sum(emotion_scores.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(emotion_scores.get(emotion, 0) for emotion in negative_emotions)
        
        total_score = positive_score + negative_score
        if total_score == 0:
            return 0.0
        
        return (positive_score - negative_score) / total_score
    
    def _calculate_arousal(self, emotion_scores: Dict[EmotionType, float]) -> float:
        """Calculate emotional arousal (calm/excited)"""
        high_arousal_emotions = [EmotionType.JOY, EmotionType.ANGER, EmotionType.FEAR, 
                               EmotionType.SURPRISE]
        low_arousal_emotions = [EmotionType.SADNESS, EmotionType.DISGUST, EmotionType.CONTEMPT]
        
        high_score = sum(emotion_scores.get(emotion, 0) for emotion in high_arousal_emotions)
        low_score = sum(emotion_scores.get(emotion, 0) for emotion in low_arousal_emotions)
        
        total_score = high_score + low_score
        if total_score == 0:
            return 0.5
        
        return high_score / total_score
    
    def _get_personality_influence(self, personality_id: str) -> Dict[str, float]:
        """Get emotional influence from personality"""
        profile = self.personality_profiles.get(personality_id)
        if not profile:
            return {"mood_stability": 0.5, "emotional_expressiveness": 0.5}
        
        return {
            "mood_stability": profile.mood_stability,
            "emotional_expressiveness": profile.emotional_expressiveness,
            "empathy_level": profile.empathy_level,
            "stress_tolerance": profile.stress_tolerance
        }
    
    def _get_memory_influence(self, text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Get emotional influence from memory"""
        # Query RAG memory for emotionally relevant content
        query = f"emotional context: {text}"
        memories = self.rag_memory.query_memory(
            query=query,
            filters={"content_type": "text"},
            limit=5
        )
        
        influence = {
            "memory_valence": 0.0,
            "memory_arousal": 0.5,
            "emotional_resonance": 0.0
        }
        
        if memories:
            total_resonance = 0
            for memory in memories:
                if memory.emotional_context:
                    influence["memory_valence"] += memory.emotional_context.get("valence", 0)
                    influence["memory_arousal"] += memory.emotional_context.get("arousal", 0.5)
                    total_resonance += 1
            
            if total_resonance > 0:
                influence["memory_valence"] /= total_resonance
                influence["memory_arousal"] /= total_resonance
                influence["emotional_resonance"] = min(total_resonance * 0.2, 1.0)
        
        return influence
    
    def _calculate_emotional_state(self, emotion_analysis: Dict[str, Any],
                                 personality_influence: Dict[str, float],
                                 memory_influence: Dict[str, float],
                                 context: Dict[str, Any]) -> EmotionalState:
        """Calculate new emotional state based on all influences"""
        # Combine influences with weights
        settings = self.resonance_settings
        
        # Base emotion from analysis
        primary_emotion = emotion_analysis["primary_emotion"]
        base_intensity = emotion_analysis["intensity"]
        
        # Adjust intensity based on personality
        personality_modifier = personality_influence.get("emotional_expressiveness", 0.5)
        adjusted_intensity = base_intensity * personality_modifier
        
        # Calculate valence with memory influence
        base_valence = emotion_analysis["valence"]
        memory_valence = memory_influence["memory_valence"]
        final_valence = (base_valence * (1 - settings["memory_influence"]) + 
                        memory_valence * settings["memory_influence"])
        
        # Calculate arousal with memory influence
        base_arousal = emotion_analysis["arousal"]
        memory_arousal = memory_influence["memory_arousal"]
        final_arousal = (base_arousal * (1 - settings["memory_influence"]) + 
                        memory_arousal * settings["memory_influence"])
        
        # Add random variation
        random_valence = (random.random() - 0.5) * 2 * settings["random_variation"]
        random_arousal = (random.random() - 0.5) * 2 * settings["random_variation"]
        
        final_valence = max(-1.0, min(1.0, final_valence + random_valence))
        final_arousal = max(0.0, min(1.0, final_arousal + random_arousal))
        
        # Determine mood state
        mood_state = self._determine_mood_state(final_valence, final_arousal)
        
        # Calculate confidence
        confidence = 1.0 - (settings["random_variation"] * 2)
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=adjusted_intensity,
            mood_state=mood_state,
            valence=final_valence,
            arousal=final_arousal,
            confidence=confidence,
            timestamp=datetime.now(),
            triggers=context.get("triggers", []),
            context=context or {}
        )
    
    def _determine_mood_state(self, valence: float, arousal: float) -> MoodState:
        """Determine mood state from valence and arousal"""
        if valence > 0.7 and arousal > 0.7:
            return MoodState.EUPHORIC
        elif valence > 0.3 and arousal > 0.5:
            return MoodState.HAPPY
        elif valence > 0.1 and arousal < 0.5:
            return MoodState.CONTENT
        elif valence < -0.3 and arousal > 0.5:
            return MoodState.ANGRY
        elif valence < -0.3 and arousal < 0.5:
            return MoodState.SAD
        elif valence < -0.7:
            return MoodState.DEPRESSED
        elif arousal > 0.7:
            return MoodState.ANXIOUS
        elif arousal > 0.5:
            return MoodState.STRESSED
        else:
            return MoodState.NEUTRAL
    
    def _update_emotional_state(self, new_state: EmotionalState):
        """Update current emotional state"""
        # Calculate duration of previous state
        if self.emotional_history:
            duration = (datetime.now() - self.current_state.timestamp).total_seconds()
            self.current_state.duration = duration
        
        # Add to history
        self.emotional_history.append(self.current_state)
        
        # Update current state
        self.current_state = new_state
        
        # Limit history size
        if len(self.emotional_history) > 1000:
            self.emotional_history = self.emotional_history[-1000:]
        
        # Save data
        self._save_emotional_data()
    
    def _store_emotional_memory(self, text: str, emotional_state: EmotionalState,
                               context: Dict[str, Any]):
        """Store emotional memory"""
        import uuid
        
        memory_id = str(uuid.uuid4())
        memory = EmotionalMemory(
            id=memory_id,
            emotion_type=emotional_state.primary_emotion,
            intensity=emotional_state.intensity,
            context=text,
            timestamp=datetime.now(),
            personality_impact={}
        )
        
        self.emotional_memories[memory_id] = memory
        
        # Store in RAG memory
        self.rag_memory.store_memory(
            content=text,
            content_type="text",
            metadata={"emotion_memory_id": memory_id},
            tags=["emotion", emotional_state.primary_emotion.value],
            emotional_context={
                "valence": emotional_state.valence,
                "arousal": emotional_state.arousal,
                "intensity": emotional_state.intensity,
                "mood_state": emotional_state.mood_state.value
            }
        )
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state"""
        return {
            "primary_emotion": self.current_state.primary_emotion.value,
            "intensity": self.current_state.intensity,
            "mood_state": self.current_state.mood_state.value,
            "valence": self.current_state.valence,
            "arousal": self.current_state.arousal,
            "confidence": self.current_state.confidence,
            "timestamp": self.current_state.timestamp.isoformat(),
            "duration": self.current_state.duration,
            "triggers": self.current_state.triggers,
            "context": self.current_state.context
        }
    
    def get_emotional_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get emotional history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for state in self.emotional_history:
            if state.timestamp >= cutoff_time:
                history.append({
                    "primary_emotion": state.primary_emotion.value,
                    "intensity": state.intensity,
                    "mood_state": state.mood_state.value,
                    "valence": state.valence,
                    "arousal": state.arousal,
                    "timestamp": state.timestamp.isoformat(),
                    "duration": state.duration
                })
        
        return history
    
    def get_emotional_statistics(self) -> Dict[str, Any]:
        """Get emotional statistics"""
        if not self.emotional_history:
            return {"error": "No emotional history available"}
        
        # Calculate statistics
        emotions = [state.primary_emotion.value for state in self.emotional_history]
        valences = [state.valence for state in self.emotional_history]
        arousals = [state.arousal for state in self.emotional_history]
        intensities = [state.intensity for state in self.emotional_history]
        
        # Emotion distribution
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "total_emotional_events": len(self.emotional_history),
            "emotion_distribution": emotion_counts,
            "average_valence": np.mean(valences),
            "average_arousal": np.mean(arousals),
            "average_intensity": np.mean(intensities),
            "valence_volatility": np.std(valences),
            "arousal_volatility": np.std(arousals),
            "most_common_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
        }
    
    def set_personality_emotion_profile(self, personality_id: str, profile: PersonalityEmotionProfile):
        """Set emotion profile for a personality"""
        self.personality_profiles[personality_id] = profile
        logging.info(f"🜂 Emotion profile set for personality: {personality_id}")
    
    def get_personality_emotion_profile(self, personality_id: str) -> Optional[Dict[str, Any]]:
        """Get emotion profile for a personality"""
        profile = self.personality_profiles.get(personality_id)
        if profile:
            return asdict(profile)
        return None
    
    def adjust_emotional_state(self, emotion_type: EmotionType, intensity: float,
                             valence: float = None, arousal: float = None):
        """Manually adjust emotional state (for testing/debugging)"""
        new_state = EmotionalState(
            primary_emotion=emotion_type,
            intensity=intensity,
            mood_state=self._determine_mood_state(valence or 0.0, arousal or 0.5),
            valence=valence or 0.0,
            arousal=arousal or 0.5,
            confidence=1.0,
            timestamp=datetime.now(),
            triggers=["manual_adjustment"],
            context={"manual": True}
        )
        
        self._update_emotional_state(new_state)
        logging.info(f"🜂 Emotional state manually adjusted to {emotion_type.value}") 