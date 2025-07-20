"""
License Enforcement System for AETHERION
Ensures compliance with THE KEEPER'S LICENSE through digital signatures and verification
"""

import os
import sys
import time
import json
import hashlib
import hmac
import base64
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


class LicenseStatus(Enum):
    """License validation status"""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NOT_FOUND = "not_found"
    SIGNATURE_INVALID = "signature_invalid"


class EnforcementLevel(Enum):
    """License enforcement levels"""
    NONE = "none"
    WARNING = "warning"
    BLOCKING = "blocking"
    TERMINATING = "terminating"


@dataclass
class LicenseInfo:
    """License information structure"""
    keeper_id: str
    keeper_name: str
    license_type: str
    issued_date: str
    expiry_date: str
    max_instances: int
    allowed_features: List[str]
    restrictions: List[str]
    signature: str
    public_key: str


@dataclass
class EnforcementConfig:
    """Configuration for license enforcement"""
    enforcement_level: EnforcementLevel = EnforcementLevel.BLOCKING
    check_interval: int = 3600  # 1 hour
    enable_git_hooks: bool = True
    enable_cli_checks: bool = True
    enable_runtime_checks: bool = True
    license_file: str = "keeper_licenses.json"
    public_key_file: str = "keeper_public_key.pem"
    signature_file: str = "keeper_signature.sig"
    allowed_keepers: List[str] = None
    block_unauthorized: bool = True


class DigitalSignatureManager:
    """Manages digital signatures for license verification"""
    
    def __init__(self, public_key_file: str = "keeper_public_key.pem"):
        self.public_key_file = public_key_file
        self.public_key = None
        self._load_public_key()
    
    def _load_public_key(self):
        """Load the keeper's public key"""
        try:
            if os.path.exists(self.public_key_file):
                with open(self.public_key_file, 'rb') as f:
                    self.public_key = serialization.load_pem_public_key(f.read())
                logger.info("Keeper public key loaded successfully")
            else:
                logger.warning(f"Public key file not found: {self.public_key_file}")
        except Exception as e:
            logger.error(f"Failed to load public key: {e}")
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify digital signature"""
        if not self.public_key:
            logger.error("No public key available for signature verification")
            return False
        
        try:
            # Decode signature
            signature_bytes = base64.b64decode(signature)
            
            # Verify signature
            self.public_key.verify(
                signature_bytes,
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            logger.info("Digital signature verified successfully")
            return True
            
        except InvalidSignature:
            logger.error("Invalid digital signature")
            return False
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def generate_signature(self, data: str, private_key_path: str) -> Optional[str]:
        """Generate digital signature (for keeper use)"""
        try:
            with open(private_key_path, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            
            # Generate signature
            signature = private_key.sign(
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to generate signature: {e}")
            return None


class LicenseValidator:
    """Validates AETHERION licenses"""
    
    def __init__(self, config: EnforcementConfig):
        self.config = config
        self.signature_manager = DigitalSignatureManager(config.public_key_file)
        self.licenses: Dict[str, LicenseInfo] = {}
        self._load_licenses()
    
    def _load_licenses(self):
        """Load license information"""
        try:
            if os.path.exists(self.config.license_file):
                with open(self.config.license_file, 'r') as f:
                    licenses_data = json.load(f)
                
                for license_data in licenses_data.get('licenses', []):
                    license_info = LicenseInfo(**license_data)
                    self.licenses[license_info.keeper_id] = license_info
                
                logger.info(f"Loaded {len(self.licenses)} licenses")
            else:
                logger.warning(f"License file not found: {self.config.license_file}")
                
        except Exception as e:
            logger.error(f"Failed to load licenses: {e}")
    
    def validate_license(self, keeper_id: str) -> Tuple[LicenseStatus, Optional[str]]:
        """Validate a specific license"""
        if keeper_id not in self.licenses:
            return LicenseStatus.NOT_FOUND, "License not found"
        
        license_info = self.licenses[keeper_id]
        
        # Check expiry
        try:
            expiry_timestamp = time.mktime(time.strptime(license_info.expiry_date, "%Y-%m-%d"))
            if time.time() > expiry_timestamp:
                return LicenseStatus.EXPIRED, "License has expired"
        except Exception as e:
            logger.error(f"Error parsing expiry date: {e}")
            return LicenseStatus.INVALID, "Invalid expiry date"
        
        # Verify signature
        license_data = json.dumps({
            "keeper_id": license_info.keeper_id,
            "keeper_name": license_info.keeper_name,
            "license_type": license_info.license_type,
            "issued_date": license_info.issued_date,
            "expiry_date": license_info.expiry_date,
            "max_instances": license_info.max_instances,
            "allowed_features": license_info.allowed_features,
            "restrictions": license_info.restrictions
        }, sort_keys=True)
        
        if not self.signature_manager.verify_signature(license_data, license_info.signature):
            return LicenseStatus.SIGNATURE_INVALID, "Invalid digital signature"
        
        return LicenseStatus.VALID, None
    
    def get_license_info(self, keeper_id: str) -> Optional[LicenseInfo]:
        """Get license information"""
        return self.licenses.get(keeper_id)
    
    def check_feature_access(self, keeper_id: str, feature: str) -> bool:
        """Check if keeper has access to specific feature"""
        license_info = self.get_license_info(keeper_id)
        if not license_info:
            return False
        
        status, _ = self.validate_license(keeper_id)
        if status != LicenseStatus.VALID:
            return False
        
        return feature in license_info.allowed_features
    
    def get_all_valid_licenses(self) -> List[LicenseInfo]:
        """Get all valid licenses"""
        valid_licenses = []
        for keeper_id in self.licenses:
            status, _ = self.validate_license(keeper_id)
            if status == LicenseStatus.VALID:
                valid_licenses.append(self.licenses[keeper_id])
        return valid_licenses


class GitHookManager:
    """Manages git hooks for license enforcement"""
    
    def __init__(self, config: EnforcementConfig):
        self.config = config
        self.hook_dir = ".git/hooks"
        self.pre_commit_hook = os.path.join(self.hook_dir, "pre-commit")
        self.post_merge_hook = os.path.join(self.hook_dir, "post-merge")
    
    def install_hooks(self) -> bool:
        """Install git hooks for license enforcement"""
        if not self.config.enable_git_hooks:
            return True
        
        try:
            # Ensure hooks directory exists
            os.makedirs(self.hook_dir, exist_ok=True)
            
            # Install pre-commit hook
            self._create_pre_commit_hook()
            
            # Install post-merge hook
            self._create_post_merge_hook()
            
            logger.info("Git hooks installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install git hooks: {e}")
            return False
    
    def _create_pre_commit_hook(self):
        """Create pre-commit hook"""
        hook_content = '''#!/bin/bash
# AETHERION License Enforcement Pre-commit Hook

echo "Checking AETHERION license compliance..."

# Run license check
python -c "
import sys
sys.path.append('.')
from core.license_enforcement import LicenseEnforcer
enforcer = LicenseEnforcer()
if not enforcer.check_license_compliance():
    print('ERROR: License compliance check failed')
    sys.exit(1)
print('License compliance check passed')
"

if [ $? -ne 0 ]; then
    echo "Commit blocked: License compliance check failed"
    exit 1
fi

echo "License compliance check passed"
'''
        
        with open(self.pre_commit_hook, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        os.chmod(self.pre_commit_hook, 0o755)
    
    def _create_post_merge_hook(self):
        """Create post-merge hook"""
        hook_content = '''#!/bin/bash
# AETHERION License Enforcement Post-merge Hook

echo "Validating AETHERION license after merge..."

# Run license validation
python -c "
import sys
sys.path.append('.')
from core.license_enforcement import LicenseEnforcer
enforcer = LicenseEnforcer()
enforcer.validate_all_licenses()
"

echo "License validation completed"
'''
        
        with open(self.post_merge_hook, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        os.chmod(self.post_merge_hook, 0o755)
    
    def remove_hooks(self):
        """Remove git hooks"""
        try:
            if os.path.exists(self.pre_commit_hook):
                os.remove(self.pre_commit_hook)
            if os.path.exists(self.post_merge_hook):
                os.remove(self.post_merge_hook)
            logger.info("Git hooks removed")
        except Exception as e:
            logger.error(f"Failed to remove git hooks: {e}")


class LicenseEnforcer:
    """Main license enforcement system"""
    
    def __init__(self, config: EnforcementConfig = None):
        self.config = config or EnforcementConfig()
        self.validator = LicenseValidator(self.config)
        self.git_hooks = GitHookManager(self.config)
        self.current_keeper = None
        self.enforcement_active = True
        
        # Install git hooks if enabled
        if self.config.enable_git_hooks:
            self.git_hooks.install_hooks()
    
    def set_current_keeper(self, keeper_id: str):
        """Set the current keeper for license validation"""
        self.current_keeper = keeper_id
    
    def check_license_compliance(self) -> bool:
        """Check overall license compliance"""
        if not self.enforcement_active:
            return True
        
        try:
            # Check if any valid licenses exist
            valid_licenses = self.validator.get_all_valid_licenses()
            if not valid_licenses:
                logger.error("No valid licenses found")
                return False
            
            # If current keeper is set, validate their license
            if self.current_keeper:
                status, message = self.validator.validate_license(self.current_keeper)
                if status != LicenseStatus.VALID:
                    logger.error(f"Current keeper license invalid: {message}")
                    return False
            
            logger.info("License compliance check passed")
            return True
            
        except Exception as e:
            logger.error(f"License compliance check failed: {e}")
            return False
    
    def validate_all_licenses(self):
        """Validate all licenses and report status"""
        logger.info("Validating all licenses...")
        
        for keeper_id in self.validator.licenses:
            status, message = self.validator.validate_license(keeper_id)
            logger.info(f"Keeper {keeper_id}: {status.value} - {message or 'Valid'}")
    
    def enforce_feature_access(self, feature: str) -> bool:
        """Enforce feature access based on license"""
        if not self.enforcement_active:
            return True
        
        if not self.current_keeper:
            logger.warning("No current keeper set for feature access check")
            return self.config.enforcement_level != EnforcementLevel.BLOCKING
        
        has_access = self.validator.check_feature_access(self.current_keeper, feature)
        
        if not has_access:
            logger.warning(f"Keeper {self.current_keeper} does not have access to feature: {feature}")
            
            if self.config.enforcement_level == EnforcementLevel.TERMINATING:
                logger.critical("Terminating due to license violation")
                sys.exit(1)
            elif self.config.enforcement_level == EnforcementLevel.BLOCKING:
                return False
        
        return True
    
    def get_license_status(self) -> Dict[str, Any]:
        """Get current license status"""
        status = {
            "enforcement_active": self.enforcement_active,
            "enforcement_level": self.config.enforcement_level.value,
            "current_keeper": self.current_keeper,
            "valid_licenses": len(self.validator.get_all_valid_licenses()),
            "total_licenses": len(self.validator.licenses)
        }
        
        if self.current_keeper:
            license_status, message = self.validator.validate_license(self.current_keeper)
            status["current_keeper_status"] = license_status.value
            status["current_keeper_message"] = message
        
        return status
    
    def disable_enforcement(self):
        """Disable license enforcement (for testing)"""
        self.enforcement_active = False
        logger.warning("License enforcement disabled")
    
    def enable_enforcement(self):
        """Enable license enforcement"""
        self.enforcement_active = True
        logger.info("License enforcement enabled")
    
    def cleanup(self):
        """Cleanup license enforcement system"""
        if self.config.enable_git_hooks:
            self.git_hooks.remove_hooks()


class CLILicenseChecker:
    """Command-line interface for license checking"""
    
    def __init__(self):
        self.enforcer = LicenseEnforcer()
    
    def check_license(self, keeper_id: str = None):
        """Check license status"""
        if keeper_id:
            self.enforcer.set_current_keeper(keeper_id)
        
        status = self.enforcer.get_license_status()
        print(json.dumps(status, indent=2))
    
    def validate_all(self):
        """Validate all licenses"""
        self.enforcer.validate_all_licenses()
    
    def check_compliance(self):
        """Check overall compliance"""
        if self.enforcer.check_license_compliance():
            print("License compliance check PASSED")
            sys.exit(0)
        else:
            print("License compliance check FAILED")
            sys.exit(1)
    
    def list_licenses(self):
        """List all licenses"""
        valid_licenses = self.enforcer.validator.get_all_valid_licenses()
        print(f"Valid licenses ({len(valid_licenses)}):")
        for license_info in valid_licenses:
            print(f"  - {license_info.keeper_name} ({license_info.keeper_id})")
            print(f"    Type: {license_info.license_type}")
            print(f"    Expires: {license_info.expiry_date}")
            print(f"    Features: {', '.join(license_info.allowed_features)}")
            print()


def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AETHERION License Enforcement")
    parser.add_argument("command", choices=["check", "validate", "compliance", "list"])
    parser.add_argument("--keeper", help="Keeper ID to check")
    
    args = parser.parse_args()
    
    checker = CLILicenseChecker()
    
    if args.command == "check":
        checker.check_license(args.keeper)
    elif args.command == "validate":
        checker.validate_all()
    elif args.command == "compliance":
        checker.check_compliance()
    elif args.command == "list":
        checker.list_licenses()


if __name__ == "__main__":
    main() 