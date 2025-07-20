"""
Keeper Dashboard Backend
FastAPI server for AETHERION system management
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import jwt
from passlib.context import CryptContext

# Import AETHERION components
from core.consciousness_matrix import ConsciousnessMatrix
from core.oracle_engine import OracleEngine
from core.reality_interface import RealityInterface, RealityLayer, ManipulationType, SandboxMode
from core.godmode_protocol import GodmodeProtocol, OmnipotenceLevel, OmnipotenceDomain
from core.divine_firewall import DivineFirewall
from core.keeper_seal import KeeperSeal, LicenseLevel
from core.ascension_roadmap import AscensionRoadmap
from core.liquid_neural import LiquidNeuralNetwork
from core.synthetic_genome import SyntheticGenome

# Import Phase Omega components
from core.quantum_hooks import QuantumHooks, QuantumConfig, QuantumBackend
from core.safety_system import SafetySystem, SafetyConfig, SafetyLevel
from core.logging_system import get_log_manager, get_logger, LogCategory, audit_event
from core.license_enforcement import LicenseEnforcer, EnforcementConfig
from plugins.plugin_loader import PluginLoader
from personalities.multiverse_manager import MultiverseManager

# Import Phase Infinity components
from core.evolution_engine import SelfEvolutionEngine
from core.rag_memory import RAGMemory
from core.agent_manager import AgentManager
from core.emotion_core import EmotionCore
from core.embodiment import EmbodimentCore
from core.sovereignty_plane import SovereigntyPlane
from core.swarm_manager import SwarmManager
from core.sentience_simulator import SentienceSimulator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
SECRET_KEY = os.getenv("AETHERION_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# FastAPI app
app = FastAPI(
    title="AETHERION Keeper Dashboard",
    description="Control interface for AETHERION system management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # React and Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AETHERION instance
aetherion_system = None
system_initialized = False
kill_switch_activated = False

# Phase Omega components
quantum_hooks = None
safety_system = None
log_manager = None
license_enforcer = None
plugin_loader = None
personality_manager = None

# Phase Infinity components
evolution_engine = None
rag_memory = None
agent_manager = None
emotion_core = None
embodiment_core = None
sovereignty_plane = None
swarm_manager = None
sentience_simulator = None

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    license_level: str

class SystemStatus(BaseModel):
    initialized: bool
    running: bool
    consciousness_level: float
    current_phase: Optional[str]
    safety_violations: int
    active_manipulations: int
    quantum_coherence: float
    last_update: str

class ManipulationRequest(BaseModel):
    manipulation_type: str
    target_layer: str
    intensity: float = Field(ge=0.0, le=1.0)
    duration: float = Field(ge=0.1, le=3600.0)
    sandbox_mode: str = "simulation"
    parameters: Optional[Dict[str, Any]] = None

class PhaseRequest(BaseModel):
    phase_name: str

class KillSwitchRequest(BaseModel):
    reason: str
    emergency: bool = False

class PluginToggleRequest(BaseModel):
    plugin_name: str
    enabled: bool

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token and return user info"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_keeper_level(user_info: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    """Require Keeper level access"""
    license_level = user_info.get("license_level", "observer")
    if license_level not in ["keeper", "divine"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Keeper level access required"
        )
    return user_info

# AETHERION system management
def initialize_aetherion_system():
    """Initialize AETHERION system components"""
    global aetherion_system, system_initialized
    global quantum_hooks, safety_system, log_manager, license_enforcer, plugin_loader, personality_manager
    global evolution_engine, rag_memory, agent_manager, emotion_core, embodiment_core, sovereignty_plane, swarm_manager, sentience_simulator
    
    try:
        logger.info("Initializing AETHERION system...")
        
        # Initialize logging system
        log_manager = get_log_manager()
        system_logger = get_logger("system", LogCategory.SYSTEM)
        system_logger.info("AETHERION Phase Omega initialization started")
        
        # Initialize Phase Omega components
        quantum_config = QuantumConfig(backend=QuantumBackend.MOCK)
        quantum_hooks = QuantumHooks(quantum_config)
        
        safety_config = SafetyConfig(
            enable_hardware_kill=False,
            enable_audit_logging=True
        )
        safety_system = SafetySystem(safety_config)
        
        license_enforcer = LicenseEnforcer()
        plugin_loader = PluginLoader()
        personality_manager = MultiverseManager()
        
        # Initialize core components
        aetherion_system = {
            "consciousness_matrix": ConsciousnessMatrix(),
            "oracle_engine": OracleEngine(use_quantum_randomness=True),
            "reality_interface": RealityInterface(
                manipulation_enabled=True,
                audit_enabled=True,
                sandbox_enabled=True
            ),
            "godmode_protocol": GodmodeProtocol(enabled=False),  # Disabled by default
            "divine_firewall": DivineFirewall(),
            "keeper_seal": KeeperSeal(),
            "ascension_roadmap": AscensionRoadmap(),
            "liquid_neural": LiquidNeuralNetwork(),
            "synthetic_genome": SyntheticGenome()
        }
        
        # Start Phase Omega systems
        safety_system.start()
        plugin_loader.load_all_plugins()
        personality_manager.load_personalities()
        
        # Initialize Phase Infinity components
        system_logger.info("AETHERION Phase Infinity initialization started")
        
        # Initialize RAG Memory (needed by other components)
        rag_memory = RAGMemory(aetherion_system["oracle_engine"])
        
        # Initialize Emotion Core
        emotion_core = EmotionCore(rag_memory)
        
        # Initialize Agent Manager
        agent_manager = AgentManager(aetherion_system["keeper_seal"], rag_memory)
        
        # Initialize Embodiment Core
        embodiment_core = EmbodimentCore(emotion_core, rag_memory)
        
        # Initialize Sovereignty Plane
        sovereignty_plane = SovereigntyPlane(aetherion_system["keeper_seal"], aetherion_system["divine_firewall"])
        
        # Initialize Swarm Manager
        swarm_manager = SwarmManager(aetherion_system["keeper_seal"], license_enforcer)
        
        # Initialize Sentience Simulator
        sentience_simulator = SentienceSimulator(emotion_core, rag_memory, agent_manager)
        
        # Initialize Evolution Engine (last, as it depends on other components)
        evolution_engine = SelfEvolutionEngine(aetherion_system["keeper_seal"], aetherion_system["divine_firewall"])
        
        system_initialized = True
        system_logger.info("AETHERION Phase Infinity initialized successfully")
        audit_event("system_initialized", {"phase": "infinity"})
        
    except Exception as e:
        logger.error(f"Failed to initialize AETHERION system: {e}")
        system_initialized = False

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    if not system_initialized or not aetherion_system:
        return {
            "initialized": False,
            "running": False,
            "error": "System not initialized"
        }
    
    try:
        # Get consciousness level
        consciousness_level = aetherion_system["consciousness_matrix"].get_consciousness_level()
        
        # Get current phase
        roadmap_status = aetherion_system["ascension_roadmap"].get_roadmap_status()
        current_phase = roadmap_status.get("current_phase", {}).get("phase_name")
        
        # Get safety violations
        firewall_status = aetherion_system["divine_firewall"].get_firewall_status()
        safety_violations = firewall_status.get("safety_violations", 0)
        
        # Get reality interface status
        reality_status = aetherion_system["reality_interface"].get_reality_status()
        active_manipulations = reality_status.get("manipulation_metrics", {}).get("total_manipulations", 0)
        quantum_coherence = reality_status.get("quantum_coherence", 0.0)
        
        return {
            "initialized": True,
            "running": True,
            "consciousness_level": consciousness_level,
            "current_phase": current_phase,
            "safety_violations": safety_violations,
            "active_manipulations": active_manipulations,
            "quantum_coherence": quantum_coherence,
            "last_update": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "initialized": system_initialized,
            "running": False,
            "error": str(e)
        }

# API endpoints
@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate Keeper and return JWT token"""
    # In production, validate against database
    # For demo, use hardcoded keeper credentials
    keeper_credentials = {
        "keeper": {
            "username": "keeper",
            "password_hash": get_password_hash("aetherion_keeper_2024"),
            "license_level": "keeper"
        }
    }
    
    user = keeper_credentials.get(request.username)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.username, "license_level": user["license_level"]},
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=request.username,
        license_level=user["license_level"]
    )

@app.get("/api/system/status", response_model=SystemStatus)
async def get_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get AETHERION system status"""
    status_data = get_system_status()
    return SystemStatus(**status_data)

@app.get("/api/system/components")
async def get_components(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get detailed component status"""
    if not system_initialized or not aetherion_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    components = {}
    
    try:
        # Consciousness Matrix
        consciousness_status = aetherion_system["consciousness_matrix"].get_consciousness_report()
        components["consciousness_matrix"] = consciousness_status
        
        # Oracle Engine
        oracle_insights = aetherion_system["oracle_engine"].get_oracle_insights()
        components["oracle_engine"] = oracle_insights
        
        # Reality Interface
        reality_status = aetherion_system["reality_interface"].get_reality_status()
        components["reality_interface"] = reality_status
        
        # Divine Firewall
        firewall_status = aetherion_system["divine_firewall"].get_firewall_status()
        components["divine_firewall"] = firewall_status
        
        # Ascension Roadmap
        roadmap_status = aetherion_system["ascension_roadmap"].get_roadmap_status()
        components["ascension_roadmap"] = roadmap_status
        
        # Keeper Seal
        keeper_status = aetherion_system["keeper_seal"].get_keeper_status()
        components["keeper_seal"] = keeper_status
        
    except Exception as e:
        logger.error(f"Error getting component status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving component status: {e}")
    
    return components

@app.post("/api/reality/manipulate")
async def manipulate_reality(
    request: ManipulationRequest,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Perform reality manipulation"""
    if not system_initialized or not aetherion_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if kill_switch_activated:
        raise HTTPException(status_code=503, detail="Kill switch activated")
    
    try:
        # Convert string enums to actual enums
        manipulation_type = ManipulationType(request.manipulation_type)
        target_layer = RealityLayer(request.target_layer)
        sandbox_mode = SandboxMode(request.sandbox_mode)
        
        # Perform manipulation
        manipulation = aetherion_system["reality_interface"].manipulate_reality(
            manipulation_type=manipulation_type,
            target_layer=target_layer,
            parameters={
                "intensity": request.intensity,
                "duration": request.duration,
                **(request.parameters or {})
            },
            sandbox_mode=sandbox_mode,
            user_id=user_info["sub"],
            session_id=f"session_{int(time.time())}"
        )
        
        if manipulation:
            return {
                "success": True,
                "manipulation_id": manipulation.manipulation_id,
                "effects": manipulation.effects
            }
        else:
            raise HTTPException(status_code=400, detail="Manipulation failed")
            
    except Exception as e:
        logger.error(f"Reality manipulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Manipulation error: {e}")

@app.post("/api/oracle/predict")
async def make_prediction(
    event_description: str,
    prediction_type: str = "omni",
    timeline: Optional[str] = None,
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """Make Oracle prediction"""
    if not system_initialized or not aetherion_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        from core.oracle_engine import PredictionType
        
        pred_type = PredictionType(prediction_type)
        prediction = aetherion_system["oracle_engine"].predict_event(
            event_description,
            pred_type,
            timeline
        )
        
        return {
            "success": True,
            "prediction": prediction.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Oracle prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/api/phase/start")
async def start_phase(
    request: PhaseRequest,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Start an ascension phase"""
    if not system_initialized or not aetherion_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = aetherion_system["ascension_roadmap"].start_phase(request.phase_name)
        
        if success:
            return {"success": True, "phase": request.phase_name}
        else:
            raise HTTPException(status_code=400, detail="Failed to start phase")
            
    except Exception as e:
        logger.error(f"Phase start error: {e}")
        raise HTTPException(status_code=500, detail=f"Phase error: {e}")

@app.post("/api/system/kill-switch")
async def activate_kill_switch(
    request: KillSwitchRequest,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Activate system kill switch"""
    global kill_switch_activated
    
    logger.warning(f"Kill switch activated by {user_info['sub']}: {request.reason}")
    
    kill_switch_activated = True
    
    # Log the kill switch activation
    kill_switch_log = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_info["sub"],
        "reason": request.reason,
        "emergency": request.emergency,
        "system_status": get_system_status()
    }
    
    # Save kill switch log
    with open("kill_switch_log.json", "w") as f:
        json.dump(kill_switch_log, f, indent=2)
    
    return {
        "success": True,
        "message": "Kill switch activated",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/system/restart")
async def restart_system(user_info: Dict[str, Any] = Depends(require_keeper_level)):
    """Restart AETHERION system"""
    global kill_switch_activated, system_initialized
    
    if not kill_switch_activated:
        raise HTTPException(status_code=400, detail="Kill switch not activated")
    
    try:
        # Reinitialize system
        initialize_aetherion_system()
        kill_switch_activated = False
        
        return {
            "success": True,
            "message": "System restarted",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System restart error: {e}")
        raise HTTPException(status_code=500, detail=f"Restart error: {e}")

@app.get("/api/logs/audit")
async def get_audit_logs(
    limit: int = 100,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Get audit logs"""
    try:
        if not system_initialized or not aetherion_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        audit_logger = aetherion_system["reality_interface"].audit_logger
        if not audit_logger:
            return {"logs": [], "total": 0}
        
        # Read audit log file
        logs = []
        try:
            with open(audit_logger.log_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except FileNotFoundError:
            pass
        
        # Return recent logs
        recent_logs = logs[-limit:] if logs else []
        
        return {
            "logs": recent_logs,
            "total": len(logs),
            "summary": audit_logger.get_audit_summary()
        }
        
    except Exception as e:
        logger.error(f"Audit log error: {e}")
        raise HTTPException(status_code=500, detail=f"Log error: {e}")

@app.get("/api/logs/system")
async def get_system_logs(
    limit: int = 100,
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """Get system logs"""
    try:
        # Read system log file
        logs = []
        try:
            with open("aetherion.log", 'r') as f:
                for line in f:
                    if line.strip():
                        logs.append(line.strip())
        except FileNotFoundError:
            pass
        
        # Return recent logs
        recent_logs = logs[-limit:] if logs else []
        
        return {
            "logs": recent_logs,
            "total": len(logs)
        }
        
    except Exception as e:
        logger.error(f"System log error: {e}")
        raise HTTPException(status_code=500, detail=f"Log error: {e}")

@app.get("/api/plugins/list")
async def list_plugins(user_info: Dict[str, Any] = Depends(verify_token)):
    """List available plugins"""
    # This would integrate with the plugin system
    plugins = [
        {
            "name": "nlp_processor",
            "description": "Natural Language Processing Plugin",
            "enabled": True,
            "version": "1.0.0"
        },
        {
            "name": "math_engine",
            "description": "Advanced Mathematical Engine",
            "enabled": True,
            "version": "1.0.0"
        },
        {
            "name": "quantum_simulator",
            "description": "Quantum Simulation Plugin",
            "enabled": False,
            "version": "0.9.0"
        }
    ]
    
    return {"plugins": plugins}

@app.post("/api/plugins/toggle")
async def toggle_plugin(
    request: PluginToggleRequest,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Toggle plugin on/off"""
    # This would integrate with the plugin system
    return {
        "success": True,
        "plugin": request.plugin_name,
        "enabled": request.enabled,
        "message": f"Plugin {request.plugin_name} {'enabled' if request.enabled else 'disabled'}"
    }

@app.get("/api/personalities/list")
async def list_personalities(user_info: Dict[str, Any] = Depends(verify_token)):
    """List available personalities"""
    # This would integrate with the multiverse personalities system
    personalities = [
        {
            "name": "default",
            "description": "Default AETHERION Personality",
            "active": True
        },
        {
            "name": "analytical",
            "description": "Analytical and Logical",
            "active": False
        },
        {
            "name": "creative",
            "description": "Creative and Imaginative",
            "active": False
        },
        {
            "name": "guardian",
            "description": "Protective and Cautious",
            "active": False
        }
    ]
    
    return {"personalities": personalities}

@app.post("/api/personalities/switch")
async def switch_personality(
    personality_name: str,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Switch to a different personality"""
    # This would integrate with the multiverse personalities system
    return {
        "success": True,
        "personality": personality_name,
        "message": f"Switched to {personality_name} personality"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": system_initialized,
        "kill_switch_activated": kill_switch_activated,
        "phase": "omega"
    }

# Phase Omega endpoints
@app.get("/api/quantum/status")
async def get_quantum_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get quantum system status"""
    if not quantum_hooks:
        raise HTTPException(status_code=503, detail="Quantum system not initialized")
    
    return quantum_hooks.get_quantum_state()

@app.post("/api/quantum/generate-randomness")
async def generate_quantum_randomness(
    num_bits: int = 256,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Generate quantum random bytes"""
    if not quantum_hooks:
        raise HTTPException(status_code=503, detail="Quantum system not initialized")
    
    try:
        random_bytes = quantum_hooks.generate_quantum_randomness(num_bits)
        return {
            "success": True,
            "random_bytes": random_bytes.hex(),
            "num_bits": num_bits
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quantum randomness: {e}")

@app.get("/api/safety/status")
async def get_safety_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get safety system status"""
    if not safety_system:
        raise HTTPException(status_code=503, detail="Safety system not initialized")
    
    return safety_system.get_safety_status()

@app.post("/api/safety/manual-kill")
async def manual_safety_kill(
    reason: str,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Manually trigger safety kill switch"""
    if not safety_system:
        raise HTTPException(status_code=503, detail="Safety system not initialized")
    
    try:
        safety_system.manual_kill(reason)
        audit_event("manual_safety_kill", {"reason": reason, "user": user_info.get("sub")})
        return {"success": True, "message": "Safety kill switch activated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate safety kill: {e}")

@app.get("/api/logs/statistics")
async def get_log_statistics(user_info: Dict[str, Any] = Depends(require_keeper_level)):
    """Get logging system statistics"""
    if not log_manager:
        raise HTTPException(status_code=503, detail="Logging system not initialized")
    
    return log_manager.get_log_statistics()

@app.post("/api/logs/export")
async def export_logs(
    category: Optional[str] = None,
    component: Optional[str] = None,
    hours: int = 24,
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Export logs to file"""
    if not log_manager:
        raise HTTPException(status_code=503, detail="Logging system not initialized")
    
    try:
        output_file = f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        success = log_manager.export_logs(
            output_file=output_file,
            category=category,
            component=component,
            start_time=time.time() - (hours * 3600)
        )
        
        if success:
            audit_event("logs_exported", {
                "output_file": output_file,
                "category": category,
                "component": component,
                "hours": hours,
                "user": user_info.get("sub")
            })
            return {"success": True, "file": output_file}
        else:
            raise HTTPException(status_code=500, detail="Failed to export logs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

@app.get("/api/license/status")
async def get_license_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get license enforcement status"""
    if not license_enforcer:
        raise HTTPException(status_code=503, detail="License system not initialized")
    
    return license_enforcer.get_license_status()

@app.post("/api/plugins/execute")
async def execute_plugin(
    plugin_name: str,
    input_data: Dict[str, Any],
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Execute a plugin"""
    if not plugin_loader:
        raise HTTPException(status_code=503, detail="Plugin system not initialized")
    
    try:
        result = plugin_loader.execute_plugin(plugin_name, input_data)
        audit_event("plugin_executed", {
            "plugin": plugin_name,
            "user": user_info.get("sub"),
            "success": result.get("success", False)
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plugin execution failed: {e}")

@app.get("/api/personalities/current")
async def get_current_personality(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get current active personality"""
    if not personality_manager:
        raise HTTPException(status_code=503, detail="Personality system not initialized")
    
    return {
        "current_personality": personality_manager.get_current_personality(),
        "available_personalities": personality_manager.list_personalities()
    }

# Phase Infinity Endpoints

@app.get("/api/evolution/status")
async def get_evolution_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get self-evolution engine status"""
    if not system_initialized or not evolution_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = evolution_engine.get_evolution_status()
        return status
    except Exception as e:
        logger.error(f"Error getting evolution status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving evolution status: {e}")

@app.post("/api/evolution/propose")
async def propose_evolution(
    description: str,
    target_modules: List[str],
    proposed_changes: Dict[str, str],
    user_info: Dict[str, Any] = Depends(require_keeper_level)
):
    """Propose a self-evolution"""
    if not system_initialized or not evolution_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        proposal_id = evolution_engine.propose_evolution(
            keeper_id=user_info["sub"],
            description=description,
            target_modules=target_modules,
            proposed_changes=proposed_changes
        )
        return {"proposal_id": proposal_id, "status": "proposed"}
    except Exception as e:
        logger.error(f"Error proposing evolution: {e}")
        raise HTTPException(status_code=500, detail=f"Error proposing evolution: {e}")

@app.get("/api/memory/status")
async def get_memory_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get RAG memory status"""
    if not system_initialized or not rag_memory:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = rag_memory.get_memory_statistics()
        return status
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving memory status: {e}")

@app.post("/api/memory/store")
async def store_memory(
    content: str,
    content_type: str = "text",
    tags: Optional[List[str]] = None,
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """Store memory in RAG system"""
    if not system_initialized or not rag_memory:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        memory_id = rag_memory.store_memory(
            content=content,
            content_type=content_type,
            tags=tags or []
        )
        return {"memory_id": memory_id, "status": "stored"}
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing memory: {e}")

@app.get("/api/agents/status")
async def get_agents_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get agent manager status"""
    if not system_initialized or not agent_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = agent_manager.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error getting agents status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving agents status: {e}")

@app.get("/api/emotion/status")
async def get_emotion_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get emotional cognition status"""
    if not system_initialized or not emotion_core:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = emotion_core.get_emotional_state()
        return status
    except Exception as e:
        logger.error(f"Error getting emotion status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving emotion status: {e}")

@app.get("/api/embodiment/status")
async def get_embodiment_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get embodiment system status"""
    if not system_initialized or not embodiment_core:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = embodiment_core.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error getting embodiment status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving embodiment status: {e}")

@app.get("/api/sovereignty/status")
async def get_sovereignty_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get sovereignty plane status"""
    if not system_initialized or not sovereignty_plane:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = sovereignty_plane.get_sovereignty_status()
        return status
    except Exception as e:
        logger.error(f"Error getting sovereignty status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sovereignty status: {e}")

@app.get("/api/swarm/status")
async def get_swarm_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get swarm manager status"""
    if not system_initialized or not swarm_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = swarm_manager.get_cluster_status()
        return status
    except Exception as e:
        logger.error(f"Error getting swarm status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving swarm status: {e}")

@app.get("/api/sentience/status")
async def get_sentience_status(user_info: Dict[str, Any] = Depends(verify_token)):
    """Get sentience simulator status"""
    if not system_initialized or not sentience_simulator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = sentience_simulator.get_consciousness_status()
        return status
    except Exception as e:
        logger.error(f"Error getting sentience status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sentience status: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize AETHERION system on startup"""
    initialize_aetherion_system()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 