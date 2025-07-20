"""
🜂 AETHERION Swarm Sovereignty
Multi-Node Deployment with Node Fingerprinting and Keeper Trust Keys
"""

import os
import json
import logging
import hashlib
import uuid
import socket
import platform
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import time
import requests
import docker
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from core.keeper_seal import KeeperSeal
from core.license_enforcement import LicenseEnforcement

class NodeStatus(Enum):
    """Node status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    SUSPICIOUS = "suspicious"

class TrustLevel(Enum):
    """Trust levels"""
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"

@dataclass
class NodeFingerprint:
    """Node fingerprint data"""
    node_id: str
    hostname: str
    platform: str
    architecture: str
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    disk_info: Dict[str, Any]
    network_info: Dict[str, Any]
    hardware_hash: str
    software_hash: str
    timestamp: datetime

@dataclass
class SwarmNode:
    """Swarm node information"""
    id: str
    name: str
    address: str
    role: str  # manager, worker
    status: NodeStatus
    trust_level: TrustLevel
    fingerprint: NodeFingerprint
    keeper_trust_key: str
    license_heartbeat: datetime
    last_seen: datetime
    performance_metrics: Dict[str, Any]
    security_status: Dict[str, Any]

@dataclass
class SwarmCluster:
    """Swarm cluster information"""
    cluster_id: str
    name: str
    nodes: Dict[str, SwarmNode]
    manager_nodes: List[str]
    worker_nodes: List[str]
    total_nodes: int
    online_nodes: int
    trust_score: float
    created_at: datetime
    last_updated: datetime

class SwarmManager:
    """
    🜂 Swarm Sovereignty Manager
    Manages multi-node deployment with security and trust
    """
    
    def __init__(self, keeper_seal: KeeperSeal, license_enforcement: LicenseEnforcement):
        self.keeper_seal = keeper_seal
        self.license_enforcement = license_enforcement
        
        # Cluster information
        self.cluster: Optional[SwarmCluster] = None
        self.current_node_id = str(uuid.uuid4())
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            logging.warning(f"Docker not available: {e}")
            self.docker_available = False
        
        # Node monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Trust management
        self.trust_threshold = 0.7
        self.heartbeat_interval = 30  # seconds
        self.node_timeout = 120  # seconds
        
        # Security settings
        self.security_settings = {
            "node_verification_required": True,
            "keeper_approval_required": True,
            "automatic_quarantine": True,
            "trust_decay_rate": 0.1,  # per day
            "max_untrusted_nodes": 2
        }
        
        # Initialize cluster
        self._initialize_cluster()
        self._generate_node_fingerprint()
        
        # Start monitoring
        self._start_monitoring()
        
        logging.info("🜂 Swarm Manager initialized")
    
    def _initialize_cluster(self):
        """Initialize swarm cluster"""
        if not self.docker_available:
            logging.warning("🜂 Docker not available - running in standalone mode")
            return
        
        try:
            # Check if swarm is initialized
            swarm_info = self.docker_client.swarm.attrs
            
            cluster_id = swarm_info.get("ID", str(uuid.uuid4()))
            cluster_name = swarm_info.get("Spec", {}).get("Name", "aetherion-swarm")
            
            self.cluster = SwarmCluster(
                cluster_id=cluster_id,
                name=cluster_name,
                nodes={},
                manager_nodes=[],
                worker_nodes=[],
                total_nodes=0,
                online_nodes=0,
                trust_score=1.0,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            # Get current nodes
            self._discover_nodes()
            
            logging.info(f"🜂 Swarm cluster initialized: {cluster_name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize swarm cluster: {e}")
            # Create mock cluster for development
            self.cluster = SwarmCluster(
                cluster_id=str(uuid.uuid4()),
                name="aetherion-swarm-mock",
                nodes={},
                manager_nodes=[],
                worker_nodes=[],
                total_nodes=0,
                online_nodes=0,
                trust_score=1.0,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
    
    def _generate_node_fingerprint(self) -> NodeFingerprint:
        """Generate fingerprint for current node"""
        # System information
        hostname = socket.gethostname()
        platform_info = platform.platform()
        architecture = platform.machine()
        
        # CPU information
        cpu_info = {
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "architecture": platform.processor()
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
        
        # Network information
        network_info = {}
        try:
            network_interfaces = psutil.net_if_addrs()
            for interface, addresses in network_interfaces.items():
                for addr in addresses:
                    if addr.family == socket.AF_INET:
                        network_info[interface] = addr.address
                        break
        except Exception:
            pass
        
        # Generate hardware hash
        hardware_data = f"{hostname}{platform_info}{architecture}{cpu_info}{memory_info}{disk_info}"
        hardware_hash = hashlib.sha256(hardware_data.encode()).hexdigest()
        
        # Generate software hash
        software_data = f"{platform.python_version()}{platform.python_implementation()}"
        software_hash = hashlib.sha256(software_data.encode()).hexdigest()
        
        fingerprint = NodeFingerprint(
            node_id=self.current_node_id,
            hostname=hostname,
            platform=platform_info,
            architecture=architecture,
            cpu_info=cpu_info,
            memory_info=memory_info,
            disk_info=disk_info,
            network_info=network_info,
            hardware_hash=hardware_hash,
            software_hash=software_hash,
            timestamp=datetime.now()
        )
        
        return fingerprint
    
    def _discover_nodes(self):
        """Discover nodes in the swarm"""
        if not self.docker_available or not self.cluster:
            return
        
        try:
            nodes = self.docker_client.nodes.list()
            
            for node in nodes:
                node_id = node.id
                node_attrs = node.attrs
                
                # Create node fingerprint
                fingerprint = self._get_node_fingerprint(node_attrs)
                
                # Determine trust level
                trust_level = self._evaluate_node_trust(fingerprint)
                
                # Create swarm node
                swarm_node = SwarmNode(
                    id=node_id,
                    name=node_attrs.get("Description", {}).get("Hostname", "unknown"),
                    address=node_attrs.get("Status", {}).get("Addr", "unknown"),
                    role="manager" if node_attrs.get("Spec", {}).get("Role") == "manager" else "worker",
                    status=NodeStatus.ONLINE if node_attrs.get("Status", {}).get("State") == "ready" else NodeStatus.OFFLINE,
                    trust_level=trust_level,
                    fingerprint=fingerprint,
                    keeper_trust_key=self._generate_trust_key(node_id),
                    license_heartbeat=datetime.now(),
                    last_seen=datetime.now(),
                    performance_metrics={},
                    security_status={}
                )
                
                self.cluster.nodes[node_id] = swarm_node
                
                # Update cluster statistics
                if swarm_node.role == "manager":
                    self.cluster.manager_nodes.append(node_id)
                else:
                    self.cluster.worker_nodes.append(node_id)
            
            self.cluster.total_nodes = len(self.cluster.nodes)
            self.cluster.online_nodes = len([n for n in self.cluster.nodes.values() if n.status == NodeStatus.ONLINE])
            
        except Exception as e:
            logging.error(f"Failed to discover nodes: {e}")
    
    def _get_node_fingerprint(self, node_attrs: Dict[str, Any]) -> NodeFingerprint:
        """Get fingerprint from node attributes"""
        # Extract information from Docker node attributes
        description = node_attrs.get("Description", {})
        
        return NodeFingerprint(
            node_id=node_attrs.get("ID", str(uuid.uuid4())),
            hostname=description.get("Hostname", "unknown"),
            platform=description.get("Platform", {}).get("OS", "unknown"),
            architecture=description.get("Platform", {}).get("Architecture", "unknown"),
            cpu_info={"count": description.get("Resources", {}).get("NanoCPUs", 0) // 1000000000},
            memory_info={"total": description.get("Resources", {}).get("MemoryBytes", 0)},
            disk_info={},
            network_info={},
            hardware_hash=hashlib.sha256(str(description).encode()).hexdigest(),
            software_hash=hashlib.sha256(str(node_attrs.get("Spec", {})).encode()).hexdigest(),
            timestamp=datetime.now()
        )
    
    def _evaluate_node_trust(self, fingerprint: NodeFingerprint) -> TrustLevel:
        """Evaluate trust level for a node"""
        # Check if node is known and trusted
        if self._is_known_node(fingerprint):
            return TrustLevel.VERIFIED
        
        # Check hardware consistency
        if self._check_hardware_consistency(fingerprint):
            return TrustLevel.HIGH
        
        # Check software consistency
        if self._check_software_consistency(fingerprint):
            return TrustLevel.MEDIUM
        
        # Default to low trust for unknown nodes
        return TrustLevel.LOW
    
    def _is_known_node(self, fingerprint: NodeFingerprint) -> bool:
        """Check if node is known"""
        # This would check against a database of known nodes
        # For now, we'll use a simple check
        return fingerprint.hostname in ["localhost", "aetherion-node-1", "aetherion-node-2"]
    
    def _check_hardware_consistency(self, fingerprint: NodeFingerprint) -> bool:
        """Check hardware consistency"""
        # This would compare against expected hardware profiles
        # For now, we'll accept most hardware
        return True
    
    def _check_software_consistency(self, fingerprint: NodeFingerprint) -> bool:
        """Check software consistency"""
        # This would check software versions and configurations
        # For now, we'll accept most software
        return True
    
    def _generate_trust_key(self, node_id: str) -> str:
        """Generate trust key for node"""
        # Generate a unique trust key for the node
        trust_data = f"{node_id}{self.keeper_seal.get_keeper_id()}{datetime.now().isoformat()}"
        return hashlib.sha256(trust_data.encode()).hexdigest()
    
    def _start_monitoring(self):
        """Start node monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logging.info("🜂 Node monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update node status
                self._update_node_status()
                
                # Check license heartbeats
                self._check_license_heartbeats()
                
                # Update trust scores
                self._update_trust_scores()
                
                # Check security status
                self._check_security_status()
                
                # Sleep for monitoring interval
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _update_node_status(self):
        """Update status of all nodes"""
        if not self.cluster:
            return
        
        for node_id, node in self.cluster.nodes.items():
            try:
                if self.docker_available:
                    # Get current node status from Docker
                    docker_node = self.docker_client.nodes.get(node_id)
                    node_attrs = docker_node.attrs
                    
                    # Update status
                    docker_status = node_attrs.get("Status", {}).get("State", "unknown")
                    if docker_status == "ready":
                        node.status = NodeStatus.ONLINE
                    elif docker_status == "down":
                        node.status = NodeStatus.OFFLINE
                    else:
                        node.status = NodeStatus.DEGRADED
                    
                    # Update last seen
                    node.last_seen = datetime.now()
                    
                    # Update performance metrics
                    node.performance_metrics = self._get_node_performance(node_id)
                
                else:
                    # Mock status update for development
                    node.last_seen = datetime.now()
                    node.performance_metrics = self._get_mock_performance()
                
            except Exception as e:
                logging.error(f"Failed to update status for node {node_id}: {e}")
                node.status = NodeStatus.OFFLINE
        
        # Update cluster statistics
        self.cluster.online_nodes = len([n for n in self.cluster.nodes.values() if n.status == NodeStatus.ONLINE])
        self.cluster.last_updated = datetime.now()
    
    def _get_node_performance(self, node_id: str) -> Dict[str, Any]:
        """Get performance metrics for a node"""
        try:
            if self.docker_available:
                # Get Docker stats
                container = self.docker_client.containers.get(node_id)
                stats = container.stats(stream=False)
                
                return {
                    "cpu_usage": stats.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0),
                    "memory_usage": stats.get("memory_stats", {}).get("usage", 0),
                    "network_rx": stats.get("networks", {}).get("eth0", {}).get("rx_bytes", 0),
                    "network_tx": stats.get("networks", {}).get("eth0", {}).get("tx_bytes", 0)
                }
        except Exception as e:
            logging.error(f"Failed to get performance for node {node_id}: {e}")
        
        return {}
    
    def _get_mock_performance(self) -> Dict[str, Any]:
        """Get mock performance metrics for development"""
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "network_rx": 0,
            "network_tx": 0
        }
    
    def _check_license_heartbeats(self):
        """Check license heartbeats for all nodes"""
        if not self.cluster:
            return
        
        current_time = datetime.now()
        
        for node_id, node in self.cluster.nodes.items():
            # Check if heartbeat is too old
            if (current_time - node.license_heartbeat).total_seconds() > self.node_timeout:
                logging.warning(f"Node {node_id} heartbeat timeout")
                node.status = NodeStatus.DEGRADED
            
            # Verify license
            if not self.license_enforcement.verify_node_license(node_id, node.keeper_trust_key):
                logging.error(f"Node {node_id} license verification failed")
                node.status = NodeStatus.QUARANTINED
    
    def _update_trust_scores(self):
        """Update trust scores for all nodes"""
        if not self.cluster:
            return
        
        total_trust = 0
        trusted_nodes = 0
        
        for node in self.cluster.nodes.values():
            # Apply trust decay
            days_since_verification = (datetime.now() - node.fingerprint.timestamp).days
            trust_decay = days_since_verification * self.security_settings["trust_decay_rate"]
            
            # Calculate trust score
            base_trust = self._get_base_trust_score(node.trust_level)
            trust_score = max(0.0, base_trust - trust_decay)
            
            # Update trust level
            if trust_score >= 0.9:
                node.trust_level = TrustLevel.VERIFIED
            elif trust_score >= 0.7:
                node.trust_level = TrustLevel.HIGH
            elif trust_score >= 0.5:
                node.trust_level = TrustLevel.MEDIUM
            elif trust_score >= 0.3:
                node.trust_level = TrustLevel.LOW
            else:
                node.trust_level = TrustLevel.UNTRUSTED
            
            total_trust += trust_score
            if trust_score >= self.trust_threshold:
                trusted_nodes += 1
        
        # Update cluster trust score
        if self.cluster.total_nodes > 0:
            self.cluster.trust_score = total_trust / self.cluster.total_nodes
        
        # Check if too many untrusted nodes
        untrusted_nodes = self.cluster.total_nodes - trusted_nodes
        if untrusted_nodes > self.security_settings["max_untrusted_nodes"]:
            logging.warning(f"Too many untrusted nodes: {untrusted_nodes}")
            self._trigger_security_alert()
    
    def _get_base_trust_score(self, trust_level: TrustLevel) -> float:
        """Get base trust score for trust level"""
        trust_scores = {
            TrustLevel.UNTRUSTED: 0.0,
            TrustLevel.LOW: 0.3,
            TrustLevel.MEDIUM: 0.6,
            TrustLevel.HIGH: 0.8,
            TrustLevel.VERIFIED: 1.0
        }
        return trust_scores.get(trust_level, 0.0)
    
    def _check_security_status(self):
        """Check security status of all nodes"""
        if not self.cluster:
            return
        
        for node in self.cluster.nodes.values():
            security_status = {
                "trust_level": node.trust_level.value,
                "status": node.status.value,
                "last_seen": node.last_seen.isoformat(),
                "license_valid": self.license_enforcement.verify_node_license(node.id, node.keeper_trust_key),
                "fingerprint_valid": self._verify_fingerprint(node.fingerprint),
                "performance_healthy": self._check_performance_health(node.performance_metrics)
            }
            
            node.security_status = security_status
            
            # Check for security issues
            if not security_status["license_valid"] or not security_status["fingerprint_valid"]:
                node.status = NodeStatus.SUSPICIOUS
                logging.warning(f"Security issues detected on node {node.id}")
    
    def _verify_fingerprint(self, fingerprint: NodeFingerprint) -> bool:
        """Verify node fingerprint"""
        # This would verify the fingerprint against known good values
        # For now, we'll accept most fingerprints
        return True
    
    def _check_performance_health(self, metrics: Dict[str, Any]) -> bool:
        """Check if performance metrics are healthy"""
        if not metrics:
            return True
        
        # Check CPU usage
        cpu_usage = metrics.get("cpu_usage", 0)
        if isinstance(cpu_usage, int) and cpu_usage > 90:
            return False
        
        # Check memory usage
        memory_usage = metrics.get("memory_usage", 0)
        if isinstance(memory_usage, (int, float)) and memory_usage > 90:
            return False
        
        return True
    
    def _trigger_security_alert(self):
        """Trigger security alert"""
        logging.error("🜂 Security alert triggered - too many untrusted nodes")
        # This would send alert to keeper
    
    def add_node(self, node_address: str, keeper_approval: bool = False) -> bool:
        """Add a new node to the swarm"""
        if not keeper_approval and self.security_settings["keeper_approval_required"]:
            logging.error("Keeper approval required to add node")
            return False
        
        if not self.docker_available:
            logging.warning("Docker not available - cannot add node")
            return False
        
        try:
            # Add node to swarm
            # This would use Docker swarm commands
            logging.info(f"🜂 Node added: {node_address}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add node {node_address}: {e}")
            return False
    
    def remove_node(self, node_id: str, keeper_approval: bool = False) -> bool:
        """Remove a node from the swarm"""
        if not keeper_approval and self.security_settings["keeper_approval_required"]:
            logging.error("Keeper approval required to remove node")
            return False
        
        if not self.cluster or node_id not in self.cluster.nodes:
            return False
        
        try:
            if self.docker_available:
                # Remove node from swarm
                # This would use Docker swarm commands
                pass
            
            # Remove from cluster
            del self.cluster.nodes[node_id]
            
            # Update cluster statistics
            self.cluster.total_nodes = len(self.cluster.nodes)
            self.cluster.online_nodes = len([n for n in self.cluster.nodes.values() if n.status == NodeStatus.ONLINE])
            
            logging.info(f"🜂 Node removed: {node_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to remove node {node_id}: {e}")
            return False
    
    def quarantine_node(self, node_id: str) -> bool:
        """Quarantine a suspicious node"""
        if not self.cluster or node_id not in self.cluster.nodes:
            return False
        
        node = self.cluster.nodes[node_id]
        node.status = NodeStatus.QUARANTINED
        
        logging.warning(f"🜂 Node quarantined: {node_id}")
        return True
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status"""
        if not self.cluster:
            return {"error": "No cluster available"}
        
        return {
            "cluster_id": self.cluster.cluster_id,
            "name": self.cluster.name,
            "total_nodes": self.cluster.total_nodes,
            "online_nodes": self.cluster.online_nodes,
            "trust_score": self.cluster.trust_score,
            "manager_nodes": len(self.cluster.manager_nodes),
            "worker_nodes": len(self.cluster.worker_nodes),
            "created_at": self.cluster.created_at.isoformat(),
            "last_updated": self.cluster.last_updated.isoformat(),
            "security_status": {
                "trust_threshold": self.trust_threshold,
                "max_untrusted_nodes": self.security_settings["max_untrusted_nodes"],
                "keeper_approval_required": self.security_settings["keeper_approval_required"]
            }
        }
    
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific node"""
        if not self.cluster or node_id not in self.cluster.nodes:
            return None
        
        node = self.cluster.nodes[node_id]
        return {
            "id": node.id,
            "name": node.name,
            "address": node.address,
            "role": node.role,
            "status": node.status.value,
            "trust_level": node.trust_level.value,
            "last_seen": node.last_seen.isoformat(),
            "license_heartbeat": node.license_heartbeat.isoformat(),
            "performance_metrics": node.performance_metrics,
            "security_status": node.security_status,
            "fingerprint": {
                "hostname": node.fingerprint.hostname,
                "platform": node.fingerprint.platform,
                "architecture": node.fingerprint.architecture,
                "hardware_hash": node.fingerprint.hardware_hash,
                "software_hash": node.fingerprint.software_hash
            }
        }
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get status of all nodes"""
        if not self.cluster:
            return []
        
        return [self.get_node_status(node_id) for node_id in self.cluster.nodes.keys()]
    
    def update_security_settings(self, settings: Dict[str, Any], keeper_id: str) -> bool:
        """Update security settings"""
        if not self.keeper_seal.verify_keeper_authority(keeper_id, "swarm_management"):
            logging.error(f"Unauthorized security settings update attempt by {keeper_id}")
            return False
        
        self.security_settings.update(settings)
        logging.info("🜂 Security settings updated")
        return True
    
    def get_security_settings(self) -> Dict[str, Any]:
        """Get current security settings"""
        return self.security_settings.copy() 