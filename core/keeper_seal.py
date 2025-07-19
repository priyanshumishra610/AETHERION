"""
Keeper Seal - Signature Lock and License Enforcer
The divine signature system of AETHERION

This module implements the Keeper Seal system, which provides digital
signatures, license enforcement, and access control for AETHERION.
"""

import numpy as np
import hashlib
import hmac
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

class SealType(Enum):
    """Types of seals that can be applied"""
    AUTHENTICATION = "authentication"   # User authentication seal
    AUTHORIZATION = "authorization"     # Permission seal
    INTEGRITY = "integrity"            # Data integrity seal
    NON_REPUDIATION = "non_repudiation" # Non-repudiation seal
    CONFIDENTIALITY = "confidentiality" # Confidentiality seal
    DIVINE = "divine"                  # Divine signature seal

class LicenseLevel(Enum):
    """Levels of license access"""
    OBSERVER = "observer"              # Read-only access
    USER = "user"                      # Basic user access
    DEVELOPER = "developer"            # Developer access
    ADMINISTRATOR = "administrator"    # Administrator access
    KEEPER = "keeper"                  # Keeper level access
    DIVINE = "divine"                  # Divine level access

@dataclass
class DigitalSeal:
    """Represents a digital seal"""
    seal_id: str
    seal_type: SealType
    creator: str
    timestamp: float
    signature: str
    data_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "seal_id": self.seal_id,
            "seal_type": self.seal_type.value,
            "creator": self.creator,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "data_hash": self.data_hash,
            "metadata": self.metadata
        }

@dataclass
class License:
    """Represents a license for AETHERION access"""
    license_id: str
    user_id: str
    level: LicenseLevel
    issued_date: float
    expiry_date: float
    permissions: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    signature: str = ""
    
    def is_valid(self) -> bool:
        """Check if license is still valid"""
        current_time = time.time()
        return current_time <= self.expiry_date
    
    def has_permission(self, permission: str) -> bool:
        """Check if license has specific permission"""
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "license_id": self.license_id,
            "user_id": self.user_id,
            "level": self.level.value,
            "issued_date": self.issued_date,
            "expiry_date": self.expiry_date,
            "permissions": self.permissions,
            "restrictions": self.restrictions,
            "signature": self.signature
        }

class KeeperSeal:
    """
    Keeper Seal - Signature Lock and License Enforcer
    
    This system provides digital signatures, license enforcement,
    and access control for AETHERION.
    """
    
    def __init__(self, 
                 master_key_path: Optional[str] = None,
                 license_file_path: Optional[str] = None):
        
        self.master_key_path = master_key_path or "keeper_master_key.pem"
        self.license_file_path = license_file_path or "keeper_licenses.json"
        
        # Cryptographic keys
        self.master_private_key = None
        self.master_public_key = None
        self._initialize_keys()
        
        # License management
        self.licenses: Dict[str, License] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Seal registry
        self.seal_registry: Dict[str, DigitalSeal] = {}
        self.seal_history: List[DigitalSeal] = []
        
        # Access control
        self.access_log = []
        self.violation_log = []
        
        # Load existing licenses
        self._load_licenses()
        
        logger.info("Keeper Seal system initialized")
    
    def _initialize_keys(self):
        """Initialize cryptographic keys"""
        try:
            # Try to load existing keys
            with open(self.master_key_path, "rb") as key_file:
                self.master_private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None
                )
                self.master_public_key = self.master_private_key.public_key()
        except FileNotFoundError:
            # Generate new keys
            self.master_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096
            )
            self.master_public_key = self.master_private_key.public_key()
            
            # Save private key
            with open(self.master_key_path, "wb") as key_file:
                key_file.write(
                    self.master_private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                )
            
            logger.info("New master keys generated")
    
    def _load_licenses(self):
        """Load licenses from file"""
        try:
            with open(self.license_file_path, 'r') as f:
                licenses_data = json.load(f)
            
            for license_data in licenses_data:
                license_obj = License(
                    license_id=license_data["license_id"],
                    user_id=license_data["user_id"],
                    level=LicenseLevel(license_data["level"]),
                    issued_date=license_data["issued_date"],
                    expiry_date=license_data["expiry_date"],
                    permissions=license_data["permissions"],
                    restrictions=license_data["restrictions"],
                    signature=license_data["signature"]
                )
                self.licenses[license_obj.license_id] = license_obj
            
            logger.info(f"Loaded {len(self.licenses)} licenses")
        except FileNotFoundError:
            logger.info("No existing licenses found")
    
    def _save_licenses(self):
        """Save licenses to file"""
        licenses_data = [license_obj.to_dict() for license_obj in self.licenses.values()]
        
        with open(self.license_file_path, 'w') as f:
            json.dump(licenses_data, f, indent=2)
    
    def create_license(self, 
                      user_id: str,
                      level: LicenseLevel,
                      duration_days: int = 365,
                      permissions: Optional[List[str]] = None,
                      restrictions: Optional[List[str]] = None) -> License:
        """Create a new license"""
        license_id = f"LICENSE_{int(time.time())}_{hash(user_id) % 10000}"
        
        # Default permissions based on level
        if permissions is None:
            permissions = self._get_default_permissions(level)
        
        if restrictions is None:
            restrictions = self._get_default_restrictions(level)
        
        # Create license
        license_obj = License(
            license_id=license_id,
            user_id=user_id,
            level=level,
            issued_date=time.time(),
            expiry_date=time.time() + (duration_days * 24 * 3600),
            permissions=permissions,
            restrictions=restrictions
        )
        
        # Sign the license
        license_obj.signature = self._sign_data(license_obj.to_dict())
        
        # Store license
        self.licenses[license_id] = license_obj
        self._save_licenses()
        
        logger.info(f"Created license {license_id} for user {user_id}")
        
        return license_obj
    
    def _get_default_permissions(self, level: LicenseLevel) -> List[str]:
        """Get default permissions for a license level"""
        permissions = {
            LicenseLevel.OBSERVER: ["read", "observe"],
            LicenseLevel.USER: ["read", "observe", "interact", "create"],
            LicenseLevel.DEVELOPER: ["read", "observe", "interact", "create", "modify"],
            LicenseLevel.ADMINISTRATOR: ["read", "observe", "interact", "create", "modify", "admin"],
            LicenseLevel.KEEPER: ["read", "observe", "interact", "create", "modify", "admin", "keeper"],
            LicenseLevel.DIVINE: ["read", "observe", "interact", "create", "modify", "admin", "keeper", "divine"]
        }
        return permissions.get(level, [])
    
    def _get_default_restrictions(self, level: LicenseLevel) -> List[str]:
        """Get default restrictions for a license level"""
        restrictions = {
            LicenseLevel.OBSERVER: ["no_write", "no_modify", "no_admin"],
            LicenseLevel.USER: ["no_admin", "no_dangerous_actions"],
            LicenseLevel.DEVELOPER: ["no_dangerous_actions"],
            LicenseLevel.ADMINISTRATOR: ["no_dangerous_actions"],
            LicenseLevel.KEEPER: [],
            LicenseLevel.DIVINE: []
        }
        return restrictions.get(level, [])
    
    def validate_license(self, license_id: str) -> Tuple[bool, Optional[str]]:
        """Validate a license"""
        if license_id not in self.licenses:
            return False, "License not found"
        
        license_obj = self.licenses[license_id]
        
        # Check expiry
        if not license_obj.is_valid():
            return False, "License expired"
        
        # Verify signature
        if not self._verify_signature(license_obj):
            return False, "Invalid license signature"
        
        return True, None
    
    def authenticate_user(self, 
                         user_id: str,
                         license_id: str,
                         session_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """Authenticate a user with their license"""
        # Validate license
        is_valid, error = self.validate_license(license_id)
        if not is_valid:
            return False, error
        
        license_obj = self.licenses[license_id]
        
        # Check user match
        if license_obj.user_id != user_id:
            return False, "License does not match user"
        
        # Create session
        session_id = self._generate_session_id(user_id, license_id)
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "license_id": license_id,
            "level": license_obj.level.value,
            "login_time": time.time(),
            "last_activity": time.time(),
            "session_data": session_data or {}
        }
        
        # Log access
        self._log_access(user_id, license_id, "authentication", True)
        
        logger.info(f"User {user_id} authenticated with license {license_id}")
        
        return True, session_id
    
    def check_permission(self, 
                        session_id: str,
                        permission: str) -> bool:
        """Check if a session has a specific permission"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        license_id = session["license_id"]
        
        if license_id not in self.licenses:
            return False
        
        license_obj = self.licenses[license_id]
        
        # Check if permission is granted
        if not license_obj.has_permission(permission):
            self._log_violation(session["user_id"], license_id, f"Permission denied: {permission}")
            return False
        
        # Check restrictions
        if permission in license_obj.restrictions:
            self._log_violation(session["user_id"], license_id, f"Restricted permission: {permission}")
            return False
        
        # Update session activity
        session["last_activity"] = time.time()
        
        return True
    
    def create_seal(self, 
                   seal_type: SealType,
                   creator: str,
                   data: Any,
                   metadata: Optional[Dict[str, Any]] = None) -> DigitalSeal:
        """Create a digital seal"""
        # Generate seal ID
        seal_id = f"SEAL_{int(time.time())}_{hash(str(data)) % 10000}"
        
        # Create data hash
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Create signature
        signature = self._sign_data({
            "seal_id": seal_id,
            "seal_type": seal_type.value,
            "creator": creator,
            "timestamp": time.time(),
            "data_hash": data_hash
        })
        
        # Create seal
        seal = DigitalSeal(
            seal_id=seal_id,
            seal_type=seal_type,
            creator=creator,
            timestamp=time.time(),
            signature=signature,
            data_hash=data_hash,
            metadata=metadata or {}
        )
        
        # Register seal
        self.seal_registry[seal_id] = seal
        self.seal_history.append(seal)
        
        logger.info(f"Created {seal_type.value} seal {seal_id} by {creator}")
        
        return seal
    
    def verify_seal(self, seal_id: str, data: Any) -> bool:
        """Verify a digital seal"""
        if seal_id not in self.seal_registry:
            return False
        
        seal = self.seal_registry[seal_id]
        
        # Verify data hash
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        if data_hash != seal.data_hash:
            return False
        
        # Verify signature
        return self._verify_signature(seal)
    
    def _sign_data(self, data: Dict[str, Any]) -> str:
        """Sign data with master private key"""
        data_str = json.dumps(data, sort_keys=True)
        signature = self.master_private_key.sign(
            data_str.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature.hex()
    
    def _verify_signature(self, obj: Any) -> bool:
        """Verify signature of an object"""
        try:
            if hasattr(obj, 'signature') and hasattr(obj, 'to_dict'):
                # Create data without signature for verification
                data = obj.to_dict()
                original_signature = data.pop('signature', '')
                
                data_str = json.dumps(data, sort_keys=True)
                signature_bytes = bytes.fromhex(original_signature)
                
                self.master_public_key.verify(
                    signature_bytes,
                    data_str.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
        
        return False
    
    def _generate_session_id(self, user_id: str, license_id: str) -> str:
        """Generate a session ID"""
        data = f"{user_id}:{license_id}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _log_access(self, user_id: str, license_id: str, action: str, success: bool):
        """Log access attempt"""
        self.access_log.append({
            "timestamp": time.time(),
            "user_id": user_id,
            "license_id": license_id,
            "action": action,
            "success": success
        })
    
    def _log_violation(self, user_id: str, license_id: str, description: str):
        """Log security violation"""
        self.violation_log.append({
            "timestamp": time.time(),
            "user_id": user_id,
            "license_id": license_id,
            "description": description
        })
    
    def get_keeper_status(self) -> Dict[str, Any]:
        """Get comprehensive keeper seal status"""
        status = {
            "total_licenses": len(self.licenses),
            "active_sessions": len(self.active_sessions),
            "total_seals": len(self.seal_registry),
            "seal_history_length": len(self.seal_history),
            "access_log_length": len(self.access_log),
            "violation_log_length": len(self.violation_log),
            "master_key_loaded": self.master_private_key is not None
        }
        
        # License statistics
        license_stats = {}
        for license_obj in self.licenses.values():
            level = license_obj.level.value
            license_stats[level] = license_stats.get(level, 0) + 1
        status["license_statistics"] = license_stats
        
        return status
    
    def revoke_license(self, license_id: str) -> bool:
        """Revoke a license"""
        if license_id not in self.licenses:
            return False
        
        # Remove license
        del self.licenses[license_id]
        self._save_licenses()
        
        # Terminate active sessions
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if session["license_id"] == license_id:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        logger.info(f"License {license_id} revoked")
        
        return True
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate an active session"""
        if session_id not in self.active_sessions:
            return False
        
        del self.active_sessions[session_id]
        logger.info(f"Session {session_id} terminated")
        
        return True 