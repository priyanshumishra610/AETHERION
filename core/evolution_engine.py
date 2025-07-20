"""
🜂 AETHERION Self-Evolution Engine
Sovereign Code Rewriting with Keeper Oversight
"""

import os
import sys
import json
import hashlib
import subprocess
import tempfile
import shutil
import docker
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import git
from cryptography.fernet import Fernet
from core.keeper_seal import KeeperSeal
from core.divine_firewall import DivineFirewall

@dataclass
class EvolutionProposal:
    """Evolution proposal with full audit trail"""
    id: str
    timestamp: datetime
    keeper_id: str
    description: str
    target_modules: List[str]
    proposed_changes: Dict[str, str]
    safety_analysis: Dict[str, Any]
    keeper_signature: str
    status: str  # pending, approved, rejected, executed, rolled_back
    execution_result: Optional[Dict[str, Any]] = None
    rollback_point: Optional[str] = None

class SelfEvolutionEngine:
    """
    🜂 Self-Evolution Engine
    Allows AETHERION to rewrite its own code under strict Keeper oversight
    """
    
    def __init__(self, keeper_seal: KeeperSeal, firewall: DivineFirewall):
        self.keeper_seal = keeper_seal
        self.firewall = firewall
        self.evolution_history: List[EvolutionProposal] = []
        self.current_version = "1.0-omega"
        self.sandbox_dir = Path("/tmp/aetherion_evolution_sandbox")
        self.backup_dir = Path("/tmp/aetherion_backups")
        self.docker_client = docker.from_env()
        
        # Ensure directories exist
        self.sandbox_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load evolution history
        self._load_evolution_history()
        
        logging.info("🜂 Self-Evolution Engine initialized")
    
    def _load_evolution_history(self):
        """Load evolution history from persistent storage"""
        history_file = Path("aetherion_evolution_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.evolution_history = [
                        EvolutionProposal(**proposal) for proposal in data
                    ]
            except Exception as e:
                logging.error(f"Failed to load evolution history: {e}")
    
    def _save_evolution_history(self):
        """Save evolution history to persistent storage"""
        history_file = Path("aetherion_evolution_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump([asdict(proposal) for proposal in self.evolution_history], f, 
                         default=str, indent=2)
        except Exception as e:
            logging.error(f"Failed to save evolution history: {e}")
    
    def propose_evolution(self, keeper_id: str, description: str, 
                         target_modules: List[str], proposed_changes: Dict[str, str]) -> str:
        """
        Propose a self-evolution with full safety analysis
        """
        proposal_id = hashlib.sha256(
            f"{keeper_id}{description}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Perform safety analysis
        safety_analysis = self._analyze_safety(proposed_changes, target_modules)
        
        # Require Keeper signature
        keeper_signature = self.keeper_seal.sign_evolution_proposal(
            proposal_id, description, target_modules, proposed_changes
        )
        
        proposal = EvolutionProposal(
            id=proposal_id,
            timestamp=datetime.now(),
            keeper_id=keeper_id,
            description=description,
            target_modules=target_modules,
            proposed_changes=proposed_changes,
            safety_analysis=safety_analysis,
            keeper_signature=keeper_signature,
            status="pending"
        )
        
        self.evolution_history.append(proposal)
        self._save_evolution_history()
        
        logging.info(f"🜂 Evolution proposal {proposal_id} created by {keeper_id}")
        return proposal_id
    
    def _analyze_safety(self, proposed_changes: Dict[str, str], 
                       target_modules: List[str]) -> Dict[str, Any]:
        """
        Analyze safety of proposed changes
        """
        analysis = {
            "risk_level": "low",
            "warnings": [],
            "critical_issues": [],
            "affected_systems": [],
            "estimated_impact": "minimal"
        }
        
        # Check for dangerous patterns
        dangerous_patterns = [
            "import os", "subprocess", "eval(", "exec(", "__import__",
            "open(", "file(", "delete", "remove", "rm -rf"
        ]
        
        for module, code in proposed_changes.items():
            for pattern in dangerous_patterns:
                if pattern in code:
                    analysis["warnings"].append(f"Dangerous pattern '{pattern}' in {module}")
                    analysis["risk_level"] = "high"
            
            # Check if core systems are affected
            if any(core_module in module for core_module in [
                "keeper_seal", "divine_firewall", "safety_system", "license_enforcement"
            ]):
                analysis["critical_issues"].append(f"Core system {module} modification")
                analysis["risk_level"] = "critical"
                analysis["affected_systems"].append(module)
        
        # Estimate impact
        if len(target_modules) > 5:
            analysis["estimated_impact"] = "major"
        elif len(target_modules) > 2:
            analysis["estimated_impact"] = "moderate"
        
        return analysis
    
    def approve_evolution(self, proposal_id: str, keeper_id: str) -> bool:
        """
        Approve an evolution proposal (requires Keeper authority)
        """
        proposal = self._get_proposal(proposal_id)
        if not proposal:
            return False
        
        if not self.keeper_seal.verify_keeper_authority(keeper_id, "evolution_approval"):
            logging.error(f"Unauthorized evolution approval attempt by {keeper_id}")
            return False
        
        if proposal.status != "pending":
            logging.error(f"Proposal {proposal_id} is not pending")
            return False
        
        proposal.status = "approved"
        self._save_evolution_history()
        
        logging.info(f"🜂 Evolution proposal {proposal_id} approved by {keeper_id}")
        return True
    
    def execute_evolution(self, proposal_id: str) -> Dict[str, Any]:
        """
        Execute an approved evolution in sandboxed environment
        """
        proposal = self._get_proposal(proposal_id)
        if not proposal or proposal.status != "approved":
            return {"success": False, "error": "Invalid or unapproved proposal"}
        
        try:
            # Create backup
            backup_point = self._create_backup()
            proposal.rollback_point = backup_point
            
            # Execute in sandbox
            result = self._execute_in_sandbox(proposal)
            
            proposal.status = "executed"
            proposal.execution_result = result
            self._save_evolution_history()
            
            logging.info(f"🜂 Evolution {proposal_id} executed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Evolution execution failed: {e}")
            self._rollback_evolution(proposal_id)
            return {"success": False, "error": str(e)}
    
    def _create_backup(self) -> str:
        """Create a backup of current system state"""
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        
        try:
            # Copy core modules
            shutil.copytree("core", backup_path / "core")
            
            # Save current git state
            repo = git.Repo(".")
            backup_path.mkdir(exist_ok=True)
            with open(backup_path / "git_state.txt", 'w') as f:
                f.write(f"Commit: {repo.head.commit.hexsha}\n")
                f.write(f"Branch: {repo.active_branch.name}\n")
            
            logging.info(f"🜂 Backup created: {backup_id}")
            return backup_id
            
        except Exception as e:
            logging.error(f"Backup creation failed: {e}")
            raise
    
    def _execute_in_sandbox(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """Execute evolution in Docker sandbox"""
        sandbox_path = self.sandbox_dir / proposal.id
        sandbox_path.mkdir(exist_ok=True)
        
        # Copy current code to sandbox
        shutil.copytree("core", sandbox_path / "core", dirs_exist_ok=True)
        
        # Apply proposed changes
        for module, code in proposal.proposed_changes.items():
            module_path = sandbox_path / module
            module_path.parent.mkdir(parents=True, exist_ok=True)
            with open(module_path, 'w') as f:
                f.write(code)
        
        # Create test container
        container = self.docker_client.containers.run(
            "python:3.9-slim",
            command="python -c 'import sys; sys.path.append(\"/code\"); import core; print(\"Evolution test successful\")'",
            volumes={str(sandbox_path): {'bind': '/code', 'mode': 'ro'}},
            detach=True,
            remove=True
        )
        
        try:
            result = container.wait(timeout=30)
            logs = container.logs().decode()
            
            if result['StatusCode'] == 0:
                # Apply changes to live system
                self._apply_changes(proposal.proposed_changes)
                return {
                    "success": True,
                    "logs": logs,
                    "sandbox_result": "passed"
                }
            else:
                return {
                    "success": False,
                    "error": f"Sandbox test failed: {logs}",
                    "sandbox_result": "failed"
                }
                
        finally:
            container.remove(force=True)
    
    def _apply_changes(self, changes: Dict[str, str]):
        """Apply approved changes to live system"""
        for module, code in changes.items():
            module_path = Path(module)
            module_path.parent.mkdir(parents=True, exist_ok=True)
            with open(module_path, 'w') as f:
                f.write(code)
        
        logging.info("🜂 Changes applied to live system")
    
    def _rollback_evolution(self, proposal_id: str):
        """Rollback evolution to previous state"""
        proposal = self._get_proposal(proposal_id)
        if not proposal or not proposal.rollback_point:
            return
        
        try:
            backup_path = self.backup_dir / proposal.rollback_point
            
            # Restore core modules
            if (backup_path / "core").exists():
                shutil.rmtree("core")
                shutil.copytree(backup_path / "core", "core")
            
            proposal.status = "rolled_back"
            self._save_evolution_history()
            
            logging.info(f"🜂 Evolution {proposal_id} rolled back successfully")
            
        except Exception as e:
            logging.error(f"Rollback failed: {e}")
    
    def _get_proposal(self, proposal_id: str) -> Optional[EvolutionProposal]:
        """Get evolution proposal by ID"""
        for proposal in self.evolution_history:
            if proposal.id == proposal_id:
                return proposal
        return None
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        return {
            "current_version": self.current_version,
            "total_proposals": len(self.evolution_history),
            "pending_proposals": len([p for p in self.evolution_history if p.status == "pending"]),
            "approved_proposals": len([p for p in self.evolution_history if p.status == "approved"]),
            "executed_proposals": len([p for p in self.evolution_history if p.status == "executed"]),
            "rolled_back_proposals": len([p for p in self.evolution_history if p.status == "rolled_back"]),
            "recent_proposals": [
                {
                    "id": p.id,
                    "description": p.description,
                    "status": p.status,
                    "timestamp": p.timestamp.isoformat(),
                    "keeper_id": p.keeper_id
                }
                for p in sorted(self.evolution_history, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }
    
    def get_proposal_details(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed proposal information"""
        proposal = self._get_proposal(proposal_id)
        if not proposal:
            return None
        
        return asdict(proposal) 