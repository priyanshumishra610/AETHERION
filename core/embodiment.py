"""
🜂 AETHERION Embodied Presence Hooks
Speech Recognition, TTS, and AR/VR Avatar Integration
"""

import os
import json
import logging
import asyncio
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import wave
import pyaudio
import tempfile
from core.emotion_core import EmotionCore
from core.rag_memory import RAGMemory

class EmbodimentMode(Enum):
    """Embodiment modes"""
    VOICE_ONLY = "voice_only"
    AVATAR_2D = "avatar_2d"
    AVATAR_3D = "avatar_3d"
    AR_OVERLAY = "ar_overlay"
    VR_IMMERSIVE = "vr_immersive"
    HOLOGraphic = "holographic"

class VoiceState(Enum):
    """Voice interaction states"""
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    SILENT = "silent"
    ERROR = "error"

@dataclass
class VoiceInput:
    """Voice input data"""
    audio_data: bytes
    sample_rate: int
    duration: float
    timestamp: datetime
    confidence: float
    transcription: Optional[str] = None
    language: str = "en-US"
    emotion_detected: Optional[str] = None

@dataclass
class VoiceOutput:
    """Voice output data"""
    text: str
    audio_data: bytes
    sample_rate: int
    duration: float
    timestamp: datetime
    voice_id: str
    emotion: Optional[str] = None
    prosody: Dict[str, Any] = None

@dataclass
class AvatarState:
    """Avatar state information"""
    mode: EmbodimentMode
    position: List[float]  # [x, y, z]
    rotation: List[float]  # [pitch, yaw, roll]
    scale: List[float]  # [x, y, z]
    animation: str
    expression: str
    gesture: str
    timestamp: datetime

class EmbodimentCore:
    """
    🜂 Embodied Presence Hooks
    Manages voice interaction, TTS, and avatar presence
    """
    
    def __init__(self, emotion_core: EmotionCore, rag_memory: RAGMemory):
        self.emotion_core = emotion_core
        self.rag_memory = rag_memory
        
        # Voice processing
        self.voice_state = VoiceState.SILENT
        self.audio_queue = queue.Queue()
        self.voice_thread = None
        self.is_listening = False
        
        # Audio configuration
        self.audio_config = {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024,
            "format": pyaudio.paInt16
        }
        
        # Voice settings
        self.voice_settings = {
            "voice_id": "aetherion_voice_01",
            "language": "en-US",
            "speed": 1.0,
            "pitch": 1.0,
            "volume": 1.0,
            "emotion_modulation": True,
            "prosody_enabled": True
        }
        
        # Avatar configuration
        self.avatar_state = AvatarState(
            mode=EmbodimentMode.VOICE_ONLY,
            position=[0.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
            animation="idle",
            expression="neutral",
            gesture="none",
            timestamp=datetime.now()
        )
        
        # Initialize components
        self._initialize_speech_recognition()
        self._initialize_tts()
        self._initialize_avatar_system()
        
        # Callbacks
        self.on_voice_input: Optional[Callable[[VoiceInput], None]] = None
        self.on_voice_output: Optional[Callable[[VoiceOutput], None]] = None
        self.on_avatar_update: Optional[Callable[[AvatarState], None]] = None
        
        logging.info("🜂 Embodiment Core initialized")
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition system"""
        try:
            # Try to import Vosk for offline recognition
            import vosk
            self.vosk_model = vosk.Model("vosk-model-small-en-us-0.15")
            self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, self.audio_config["sample_rate"])
            self.use_vosk = True
            logging.info("🜂 Vosk speech recognition initialized")
        except ImportError:
            try:
                # Fallback to Whisper
                import whisper
                self.whisper_model = whisper.load_model("base")
                self.use_vosk = False
                logging.info("🜂 Whisper speech recognition initialized")
            except ImportError:
                logging.warning("🜂 No speech recognition available - voice input disabled")
                self.use_vosk = False
                self.whisper_model = None
    
    def _initialize_tts(self):
        """Initialize text-to-speech system"""
        try:
            # Try to import Coqui TTS
            from TTS.api import TTS
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            self.use_coqui_tts = True
            logging.info("🜂 Coqui TTS initialized")
        except ImportError:
            try:
                # Fallback to pyttsx3
                import pyttsx3
                self.pyttsx3_engine = pyttsx3.init()
                self.use_coqui_tts = False
                logging.info("🜂 pyttsx3 TTS initialized")
            except ImportError:
                logging.warning("🜂 No TTS available - voice output disabled")
                self.use_coqui_tts = False
                self.pyttsx3_engine = None
    
    def _initialize_avatar_system(self):
        """Initialize avatar system"""
        # Avatar system would integrate with Three.js or similar
        # For now, we'll create a mock implementation
        self.avatar_available = True
        self.avatar_websocket = None
        
        logging.info("🜂 Avatar system initialized")
    
    def start_voice_listening(self):
        """Start listening for voice input"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.voice_thread = threading.Thread(target=self._voice_listening_loop)
        self.voice_thread.daemon = True
        self.voice_thread.start()
        
        self.voice_state = VoiceState.LISTENING
        logging.info("🜂 Voice listening started")
    
    def stop_voice_listening(self):
        """Stop listening for voice input"""
        self.is_listening = False
        self.voice_state = VoiceState.SILENT
        
        if self.voice_thread:
            self.voice_thread.join(timeout=1.0)
        
        logging.info("🜂 Voice listening stopped")
    
    def _voice_listening_loop(self):
        """Main voice listening loop"""
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=self.audio_config["format"],
                channels=self.audio_config["channels"],
                rate=self.audio_config["sample_rate"],
                input=True,
                frames_per_buffer=self.audio_config["chunk_size"]
            )
            
            audio_buffer = b""
            silence_threshold = 0.01
            silence_duration = 0
            max_silence_duration = 2.0  # seconds
            
            while self.is_listening:
                try:
                    data = stream.read(self.audio_config["chunk_size"], exception_on_overflow=False)
                    audio_buffer += data
                    
                    # Check for silence
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    volume = np.sqrt(np.mean(audio_array**2))
                    
                    if volume < silence_threshold:
                        silence_duration += self.audio_config["chunk_size"] / self.audio_config["sample_rate"]
                    else:
                        silence_duration = 0
                    
                    # Process audio when silence is detected or buffer is full
                    if (silence_duration >= max_silence_duration and len(audio_buffer) > 0) or len(audio_buffer) > self.audio_config["sample_rate"] * 10:
                        if len(audio_buffer) > 0:
                            self._process_voice_input(audio_buffer)
                        audio_buffer = b""
                        silence_duration = 0
                
                except Exception as e:
                    logging.error(f"Error in voice listening loop: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logging.error(f"Failed to initialize audio stream: {e}")
        finally:
            p.terminate()
    
    def _process_voice_input(self, audio_data: bytes):
        """Process voice input and convert to text"""
        try:
            self.voice_state = VoiceState.PROCESSING
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                
                # Write WAV file
                with wave.open(temp_filename, 'wb') as wav_file:
                    wav_file.setnchannels(self.audio_config["channels"])
                    wav_file.setsampwidth(pyaudio.get_sample_size(self.audio_config["format"]))
                    wav_file.setframerate(self.audio_config["sample_rate"])
                    wav_file.writeframes(audio_data)
            
            # Transcribe audio
            transcription = self._transcribe_audio(temp_filename)
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            if transcription and len(transcription.strip()) > 0:
                # Create voice input object
                voice_input = VoiceInput(
                    audio_data=audio_data,
                    sample_rate=self.audio_config["sample_rate"],
                    duration=len(audio_data) / (self.audio_config["sample_rate"] * 2),
                    timestamp=datetime.now(),
                    confidence=0.8,  # Placeholder
                    transcription=transcription,
                    language=self.voice_settings["language"]
                )
                
                # Detect emotion in speech
                voice_input.emotion_detected = self._detect_speech_emotion(transcription)
                
                # Store in memory
                self.rag_memory.store_memory(
                    content=transcription,
                    content_type="voice",
                    metadata={"voice_input_id": str(hash(transcription))},
                    tags=["voice", "input", voice_input.emotion_detected] if voice_input.emotion_detected else ["voice", "input"]
                )
                
                # Process emotional input
                self.emotion_core.process_emotional_input(
                    transcription,
                    context={"source": "voice_input", "confidence": voice_input.confidence}
                )
                
                # Call callback if set
                if self.on_voice_input:
                    self.on_voice_input(voice_input)
                
                logging.info(f"🜂 Voice input processed: '{transcription}'")
            
            self.voice_state = VoiceState.LISTENING
            
        except Exception as e:
            logging.error(f"Error processing voice input: {e}")
            self.voice_state = VoiceState.ERROR
    
    def _transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio file to text"""
        try:
            if self.use_vosk:
                # Use Vosk for transcription
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                
                if self.vosk_recognizer.AcceptWaveform(audio_data):
                    result = json.loads(self.vosk_recognizer.Result())
                    return result.get("text", "")
                else:
                    result = json.loads(self.vosk_recognizer.PartialResult())
                    return result.get("partial", "")
            
            elif self.whisper_model:
                # Use Whisper for transcription
                result = self.whisper_model.transcribe(audio_file)
                return result["text"]
            
            else:
                logging.warning("No speech recognition available")
                return None
                
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return None
    
    def _detect_speech_emotion(self, text: str) -> Optional[str]:
        """Detect emotion in speech text"""
        # Simple keyword-based emotion detection
        # In a full implementation, this would use advanced NLP models
        
        emotion_keywords = {
            "joy": ["happy", "excited", "great", "wonderful", "amazing"],
            "sadness": ["sad", "unhappy", "miserable", "lonely"],
            "anger": ["angry", "mad", "furious", "hate"],
            "fear": ["afraid", "scared", "terrified", "worried"],
            "surprise": ["surprised", "shocked", "amazed", "wow"]
        }
        
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return None
    
    def speak_text(self, text: str, emotion: Optional[str] = None, 
                  prosody: Optional[Dict[str, Any]] = None) -> Optional[VoiceOutput]:
        """Convert text to speech and play audio"""
        try:
            self.voice_state = VoiceState.SPEAKING
            
            # Get current emotional state for voice modulation
            if emotion is None:
                emotional_state = self.emotion_core.get_emotional_state()
                emotion = emotional_state["primary_emotion"]
            
            # Generate audio
            audio_data = self._generate_speech(text, emotion, prosody)
            
            if audio_data:
                # Create voice output object
                voice_output = VoiceOutput(
                    text=text,
                    audio_data=audio_data,
                    sample_rate=self.audio_config["sample_rate"],
                    duration=len(audio_data) / (self.audio_config["sample_rate"] * 2),
                    timestamp=datetime.now(),
                    voice_id=self.voice_settings["voice_id"],
                    emotion=emotion,
                    prosody=prosody
                )
                
                # Play audio
                self._play_audio(audio_data)
                
                # Store in memory
                self.rag_memory.store_memory(
                    content=text,
                    content_type="voice",
                    metadata={"voice_output_id": str(hash(text))},
                    tags=["voice", "output", emotion] if emotion else ["voice", "output"]
                )
                
                # Call callback if set
                if self.on_voice_output:
                    self.on_voice_output(voice_output)
                
                logging.info(f"🜂 Voice output: '{text}'")
                
                self.voice_state = VoiceState.SILENT
                return voice_output
            
        except Exception as e:
            logging.error(f"Error generating speech: {e}")
            self.voice_state = VoiceState.ERROR
        
        return None
    
    def _generate_speech(self, text: str, emotion: str, 
                        prosody: Optional[Dict[str, Any]]) -> Optional[bytes]:
        """Generate speech audio from text"""
        try:
            if self.use_coqui_tts:
                # Use Coqui TTS
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                # Generate speech with emotion
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_filename,
                    speaker=self.voice_settings["voice_id"]
                )
                
                # Read audio data
                with open(temp_filename, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up
                os.unlink(temp_filename)
                
                return audio_data
            
            elif self.pyttsx3_engine:
                # Use pyttsx3
                self.pyttsx3_engine.say(text)
                self.pyttsx3_engine.runAndWait()
                
                # Note: pyttsx3 doesn't return audio data, it plays directly
                # This is a limitation for now
                return None
            
            else:
                logging.warning("No TTS available")
                return None
                
        except Exception as e:
            logging.error(f"Speech generation failed: {e}")
            return None
    
    def _play_audio(self, audio_data: bytes):
        """Play audio data"""
        try:
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=self.audio_config["format"],
                channels=self.audio_config["channels"],
                rate=self.audio_config["sample_rate"],
                output=True
            )
            
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            logging.error(f"Audio playback failed: {e}")
    
    def set_embodiment_mode(self, mode: EmbodimentMode):
        """Set embodiment mode"""
        self.avatar_state.mode = mode
        self.avatar_state.timestamp = datetime.now()
        
        if self.on_avatar_update:
            self.on_avatar_update(self.avatar_state)
        
        logging.info(f"🜂 Embodiment mode set to: {mode.value}")
    
    def update_avatar_state(self, position: Optional[List[float]] = None,
                          rotation: Optional[List[float]] = None,
                          scale: Optional[List[float]] = None,
                          animation: Optional[str] = None,
                          expression: Optional[str] = None,
                          gesture: Optional[str] = None):
        """Update avatar state"""
        if position:
            self.avatar_state.position = position
        if rotation:
            self.avatar_state.rotation = rotation
        if scale:
            self.avatar_state.scale = scale
        if animation:
            self.avatar_state.animation = animation
        if expression:
            self.avatar_state.expression = expression
        if gesture:
            self.avatar_state.gesture = gesture
        
        self.avatar_state.timestamp = datetime.now()
        
        if self.on_avatar_update:
            self.on_avatar_update(self.avatar_state)
    
    def get_avatar_state(self) -> Dict[str, Any]:
        """Get current avatar state"""
        return asdict(self.avatar_state)
    
    def set_voice_settings(self, settings: Dict[str, Any]):
        """Update voice settings"""
        self.voice_settings.update(settings)
        logging.info("🜂 Voice settings updated")
    
    def get_voice_settings(self) -> Dict[str, Any]:
        """Get current voice settings"""
        return self.voice_settings.copy()
    
    def get_voice_state(self) -> str:
        """Get current voice state"""
        return self.voice_state.value
    
    def set_voice_callbacks(self, on_input: Optional[Callable[[VoiceInput], None]] = None,
                          on_output: Optional[Callable[[VoiceOutput], None]] = None):
        """Set voice event callbacks"""
        self.on_voice_input = on_input
        self.on_voice_output = on_output
    
    def set_avatar_callback(self, callback: Optional[Callable[[AvatarState], None]] = None):
        """Set avatar update callback"""
        self.on_avatar_update = callback
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get embodiment system status"""
        return {
            "voice_state": self.voice_state.value,
            "is_listening": self.is_listening,
            "embodiment_mode": self.avatar_state.mode.value,
            "speech_recognition_available": self.use_vosk or self.whisper_model is not None,
            "tts_available": self.use_coqui_tts or self.pyttsx3_engine is not None,
            "avatar_available": self.avatar_available,
            "voice_settings": self.voice_settings,
            "audio_config": self.audio_config
        }
    
    def export_voice_data(self, format: str = "wav") -> Optional[str]:
        """Export voice interaction data"""
        # This would export voice recordings and transcriptions
        # Implementation depends on storage requirements
        pass
    
    def import_voice_data(self, data_file: str) -> bool:
        """Import voice interaction data"""
        # This would import voice recordings and transcriptions
        # Implementation depends on storage requirements
        pass 