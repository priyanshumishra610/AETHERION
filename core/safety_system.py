"""
Safety System for AETHERION
Implements hardware kill switch, guardian monitoring, and emergency shutdown
"""

import os
import sys
import time
import signal
import threading
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for AETHERION"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class KillSwitchType(Enum):
    """Types of kill switch mechanisms"""
    SOFTWARE = "software"
    HARDWARE = "hardware"
    NETWORK = "network"
    GUARDIAN = "guardian"


@dataclass
class SafetyConfig:
    """Configuration for safety system"""
    enable_hardware_kill: bool = False
    hardware_pin: int = 18  # GPIO pin for hardware kill switch
    guardian_port: int = 8080
    guardian_timeout: int = 30
    emergency_shutdown_delay: int = 5
    max_memory_usage: float = 0.9  # 90% of available memory
    max_cpu_usage: float = 0.95  # 95% of CPU
    max_disk_usage: float = 0.95  # 95% of disk
    safety_check_interval: int = 10  # seconds
    enable_audit_logging: bool = True
    audit_log_file: str = "safety_audit.log"


class HardwareKillSwitch:
    """Hardware kill switch using GPIO (Raspberry Pi)"""
    
    def __init__(self, pin: int = 18):
        self.pin = pin
        self.gpio_available = False
        self.kill_triggered = False
        
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.gpio_available = True
            
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(pin, GPIO.FALLING, callback=self._kill_callback, bouncetime=300)
            
            logger.info(f"Hardware kill switch initialized on GPIO pin {pin}")
            
        except ImportError:
            logger.warning("RPi.GPIO not available, hardware kill switch disabled")
        except Exception as e:
            logger.error(f"Failed to initialize hardware kill switch: {e}")
    
    def _kill_callback(self, channel):
        """Callback when kill switch is triggered"""
        logger.critical("HARDWARE KILL SWITCH TRIGGERED!")
        self.kill_triggered = True
        
    def is_triggered(self) -> bool:
        """Check if hardware kill switch is triggered"""
        if not self.gpio_available:
            return False
            
        try:
            # Check GPIO state
            return self.GPIO.input(self.pin) == self.GPIO.LOW
        except Exception as e:
            logger.error(f"Error reading hardware kill switch: {e}")
            return False
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        if self.gpio_available:
            try:
                self.GPIO.cleanup()
                logger.info("Hardware kill switch cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up hardware kill switch: {e}")


class GuardianListener:
    """Guardian process listener for external kill signals"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = None
        self.running = False
        self.kill_triggered = False
        
    def start(self):
        """Start guardian listener server"""
        try:
            import socket
            import threading
            
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(('localhost', self.port))
            self.server.listen(1)
            self.running = True
            
            # Start listener thread
            listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            listener_thread.start()
            
            logger.info(f"Guardian listener started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start guardian listener: {e}")
    
    def _listen_loop(self):
        """Main listener loop"""
        while self.running:
            try:
                client, addr = self.server.accept()
                data = client.recv(1024).decode('utf-8')
                client.close()
                
                if data.strip() == "KILL":
                    logger.critical("GUARDIAN KILL SIGNAL RECEIVED!")
                    self.kill_triggered = True
                    
            except Exception as e:
                if self.running:
                    logger.error(f"Error in guardian listener: {e}")
    
    def stop(self):
        """Stop guardian listener"""
        self.running = False
        if self.server:
            try:
                self.server.close()
                logger.info("Guardian listener stopped")
            except Exception as e:
                logger.error(f"Error stopping guardian listener: {e}")


class SystemMonitor:
    """Monitor system resources for safety"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.warnings = []
        
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health = {
            "memory": self.check_memory_usage(),
            "cpu": self.check_cpu_usage(),
            "disk": self.check_disk_usage(),
            "processes": self.check_critical_processes(),
            "overall_status": "healthy"
        }
        
        # Determine overall status
        critical_issues = sum(1 for check in health.values() 
                            if isinstance(check, dict) and check.get("status") == "critical")
        
        if critical_issues > 0:
            health["overall_status"] = "critical"
        elif any(check.get("status") == "warning" for check in health.values() 
                if isinstance(check, dict)):
            health["overall_status"] = "warning"
            
        return health
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100
            
            status = "healthy"
            if usage_percent > self.config.max_memory_usage:
                status = "critical" if usage_percent > 0.95 else "warning"
                
            return {
                "status": status,
                "usage_percent": round(usage_percent * 100, 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            usage_percent = cpu_percent / 100
            
            status = "healthy"
            if usage_percent > self.config.max_cpu_usage:
                status = "critical" if usage_percent > 0.98 else "warning"
                
            return {
                "status": status,
                "usage_percent": round(cpu_percent, 2),
                "core_count": psutil.cpu_count()
            }
        except Exception as e:
            logger.error(f"Error checking CPU usage: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = disk.percent / 100
            
            status = "healthy"
            if usage_percent > self.config.max_disk_usage:
                status = "critical" if usage_percent > 0.98 else "warning"
                
            return {
                "status": status,
                "usage_percent": round(disk.percent, 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Error checking disk usage: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_critical_processes(self) -> Dict[str, Any]:
        """Check if critical AETHERION processes are running"""
        try:
            critical_processes = ["python", "aetherion", "keeper"]
            running_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if any(critical in proc.info['name'].lower() for critical in critical_processes):
                        running_processes.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "cmdline": ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            return {
                "status": "healthy" if running_processes else "warning",
                "critical_processes": running_processes,
                "count": len(running_processes)
            }
        except Exception as e:
            logger.error(f"Error checking critical processes: {e}")
            return {"status": "error", "error": str(e)}


class SafetyAuditLogger:
    """Audit logging for safety events"""
    
    def __init__(self, log_file: str = "safety_audit.log"):
        self.log_file = log_file
        self.log_lock = threading.Lock()
        
    def log_event(self, event_type: str, details: Dict[str, Any], level: SafetyLevel = SafetyLevel.NORMAL):
        """Log a safety event"""
        timestamp = time.time()
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "level": level.value,
            "details": details,
            "hash": self._generate_event_hash(timestamp, event_type, details)
        }
        
        with self.log_lock:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(event) + '\n')
                logger.info(f"Safety event logged: {event_type}")
            except Exception as e:
                logger.error(f"Failed to log safety event: {e}")
    
    def _generate_event_hash(self, timestamp: float, event_type: str, details: Dict[str, Any]) -> str:
        """Generate hash for event integrity"""
        event_string = f"{timestamp}:{event_type}:{json.dumps(details, sort_keys=True)}"
        return hashlib.sha256(event_string.encode()).hexdigest()
    
    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent safety events"""
        events = []
        cutoff_time = time.time() - (hours * 3600)
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get("timestamp", 0) >= cutoff_time:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
            
        return sorted(events, key=lambda x: x.get("timestamp", 0), reverse=True)


class SafetySystem:
    """Main safety system for AETHERION"""
    
    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.monitor = SystemMonitor(self.config)
        self.audit_logger = SafetyAuditLogger(self.config.audit_log_file)
        self.hardware_kill = HardwareKillSwitch(self.config.hardware_pin)
        self.guardian = GuardianListener(self.config.guardian_port)
        
        self.running = False
        self.safety_thread = None
        self.kill_callbacks: List[Callable] = []
        self.warning_callbacks: List[Callable] = []
        
        # Safety state
        self.safety_level = SafetyLevel.NORMAL
        self.last_health_check = {}
        self.consecutive_warnings = 0
        
    def start(self):
        """Start the safety system"""
        if self.running:
            logger.warning("Safety system already running")
            return
            
        self.running = True
        
        # Start guardian listener
        if self.config.enable_hardware_kill:
            self.guardian.start()
        
        # Start safety monitoring thread
        self.safety_thread = threading.Thread(target=self._safety_monitor_loop, daemon=True)
        self.safety_thread.start()
        
        self.audit_logger.log_event("safety_system_started", {
            "config": asdict(self.config)
        }, SafetyLevel.NORMAL)
        
        logger.info("Safety system started")
    
    def stop(self):
        """Stop the safety system"""
        self.running = False
        
        # Stop guardian listener
        self.guardian.stop()
        
        # Cleanup hardware kill switch
        self.hardware_kill.cleanup()
        
        self.audit_logger.log_event("safety_system_stopped", {}, SafetyLevel.NORMAL)
        logger.info("Safety system stopped")
    
    def _safety_monitor_loop(self):
        """Main safety monitoring loop"""
        while self.running:
            try:
                # Check hardware kill switch
                if self.hardware_kill.is_triggered() or self.hardware_kill.kill_triggered:
                    self._trigger_emergency_kill("hardware_kill_switch")
                    break
                
                # Check guardian kill signal
                if self.guardian.kill_triggered:
                    self._trigger_emergency_kill("guardian_kill_signal")
                    break
                
                # Check system health
                health = self.monitor.check_system_health()
                self.last_health_check = health
                
                # Update safety level
                new_level = self._determine_safety_level(health)
                if new_level != self.safety_level:
                    self._update_safety_level(new_level, health)
                
                # Check for critical conditions
                if health["overall_status"] == "critical":
                    self.consecutive_warnings += 1
                    if self.consecutive_warnings >= 3:
                        self._trigger_emergency_kill("system_critical_condition")
                        break
                else:
                    self.consecutive_warnings = 0
                
                # Sleep before next check
                time.sleep(self.config.safety_check_interval)
                
            except Exception as e:
                logger.error(f"Error in safety monitor loop: {e}")
                time.sleep(self.config.safety_check_interval)
    
    def _determine_safety_level(self, health: Dict[str, Any]) -> SafetyLevel:
        """Determine current safety level based on health check"""
        if health["overall_status"] == "critical":
            return SafetyLevel.CRITICAL
        elif health["overall_status"] == "warning":
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.NORMAL
    
    def _update_safety_level(self, new_level: SafetyLevel, health: Dict[str, Any]):
        """Update safety level and trigger callbacks"""
        old_level = self.safety_level
        self.safety_level = new_level
        
        self.audit_logger.log_event("safety_level_changed", {
            "old_level": old_level.value,
            "new_level": new_level.value,
            "health": health
        }, new_level)
        
        # Trigger callbacks
        if new_level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL]:
            for callback in self.warning_callbacks:
                try:
                    callback(new_level, health)
                except Exception as e:
                    logger.error(f"Error in warning callback: {e}")
    
    def _trigger_emergency_kill(self, reason: str):
        """Trigger emergency kill switch"""
        logger.critical(f"EMERGENCY KILL TRIGGERED: {reason}")
        
        self.audit_logger.log_event("emergency_kill_triggered", {
            "reason": reason,
            "safety_level": self.safety_level.value
        }, SafetyLevel.EMERGENCY)
        
        # Trigger kill callbacks
        for callback in self.kill_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Error in kill callback: {e}")
        
        # Emergency shutdown
        self._emergency_shutdown(reason)
    
    def _emergency_shutdown(self, reason: str):
        """Perform emergency shutdown"""
        logger.critical(f"Performing emergency shutdown: {reason}")
        
        # Log shutdown
        self.audit_logger.log_event("emergency_shutdown_started", {
            "reason": reason,
            "delay_seconds": self.config.emergency_shutdown_delay
        }, SafetyLevel.EMERGENCY)
        
        # Wait for configured delay
        time.sleep(self.config.emergency_shutdown_delay)
        
        # Force shutdown
        try:
            # Kill all Python processes (AETHERION)
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'python' in proc.info['name'].lower():
                        proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Force exit
            os._exit(1)
            
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
            os._exit(1)
    
    def add_kill_callback(self, callback: Callable[[str], None]):
        """Add callback for kill events"""
        self.kill_callbacks.append(callback)
    
    def add_warning_callback(self, callback: Callable[[SafetyLevel, Dict[str, Any]], None]):
        """Add callback for warning events"""
        self.warning_callbacks.append(callback)
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety system status"""
        return {
            "running": self.running,
            "safety_level": self.safety_level.value,
            "last_health_check": self.last_health_check,
            "consecutive_warnings": self.consecutive_warnings,
            "hardware_kill_available": self.hardware_kill.gpio_available,
            "hardware_kill_triggered": self.hardware_kill.kill_triggered,
            "guardian_kill_triggered": self.guardian.kill_triggered,
            "config": asdict(self.config)
        }
    
    def manual_kill(self, reason: str = "manual_kill"):
        """Manually trigger kill switch"""
        self._trigger_emergency_kill(reason)
    
    def restart_system(self):
        """Restart AETHERION system"""
        logger.info("Restarting AETHERION system")
        
        self.audit_logger.log_event("system_restart_requested", {}, SafetyLevel.NORMAL)
        
        # Stop safety system
        self.stop()
        
        # Restart AETHERION
        try:
            subprocess.Popen([sys.executable, "scripts/start_aetherion.py"])
            logger.info("AETHERION restart initiated")
        except Exception as e:
            logger.error(f"Failed to restart AETHERION: {e}")
    
    def get_audit_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent audit events"""
        return self.audit_logger.get_recent_events(hours) 