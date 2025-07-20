#!/usr/bin/env python3
"""
Guardian Process for AETHERION
External process for sending kill signals and monitoring AETHERION
"""

import os
import sys
import time
import signal
import socket
import json
import logging
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GuardianConfig:
    """Configuration for guardian process"""
    aetherion_host: str = "localhost"
    aetherion_port: int = 8080
    check_interval: int = 30
    kill_timeout: int = 10
    enable_monitoring: bool = True
    enable_auto_kill: bool = False
    max_memory_usage: float = 0.95
    max_cpu_usage: float = 0.98
    log_file: str = "guardian.log"


class GuardianProcess:
    """Guardian process for AETHERION monitoring and kill signals"""
    
    def __init__(self, config: GuardianConfig):
        self.config = config
        self.running = False
        self.last_check = 0
        self.kill_count = 0
        self.monitoring_data = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Guardian process initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down guardian")
        self.running = False
    
    def start(self):
        """Start the guardian process"""
        self.running = True
        logger.info("Guardian process started")
        
        try:
            while self.running:
                self._monitor_cycle()
                time.sleep(self.config.check_interval)
        except KeyboardInterrupt:
            logger.info("Guardian process interrupted")
        except Exception as e:
            logger.error(f"Guardian process error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the guardian process"""
        self.running = False
        logger.info("Guardian process stopped")
    
    def _monitor_cycle(self):
        """Perform one monitoring cycle"""
        try:
            # Check AETHERION health
            health_status = self._check_aetherion_health()
            
            # Check system resources
            system_status = self._check_system_resources()
            
            # Log monitoring data
            monitoring_entry = {
                "timestamp": datetime.now().isoformat(),
                "aetherion_health": health_status,
                "system_resources": system_status
            }
            self.monitoring_data.append(monitoring_entry)
            
            # Keep only last 1000 entries
            if len(self.monitoring_data) > 1000:
                self.monitoring_data = self.monitoring_data[-1000:]
            
            # Check for auto-kill conditions
            if self.config.enable_auto_kill:
                self._check_auto_kill_conditions(health_status, system_status)
            
            self.last_check = time.time()
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
    
    def _check_aetherion_health(self) -> Dict[str, Any]:
        """Check AETHERION system health"""
        try:
            # Try to connect to AETHERION guardian port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.config.aetherion_host, self.config.aetherion_port))
            sock.close()
            
            if result == 0:
                return {"status": "healthy", "reachable": True}
            else:
                return {"status": "unreachable", "reachable": False}
                
        except Exception as e:
            logger.error(f"Error checking AETHERION health: {e}")
            return {"status": "error", "reachable": False, "error": str(e)}
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1) / 100
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100
            
            return {
                "memory_usage": round(memory_usage, 3),
                "cpu_usage": round(cpu_usage, 3),
                "disk_usage": round(disk_usage, 3),
                "memory_available_gb": round(memory.available / (1024**3), 2)
            }
            
        except ImportError:
            logger.warning("psutil not available, skipping system resource check")
            return {"error": "psutil not available"}
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {"error": str(e)}
    
    def _check_auto_kill_conditions(self, health_status: Dict[str, Any], system_status: Dict[str, Any]):
        """Check if auto-kill conditions are met"""
        try:
            # Check memory usage
            if "memory_usage" in system_status:
                memory_usage = system_status["memory_usage"]
                if memory_usage > self.config.max_memory_usage:
                    logger.warning(f"Memory usage critical: {memory_usage:.1%}")
                    self.send_kill_signal("auto_kill_memory_critical")
                    return
            
            # Check CPU usage
            if "cpu_usage" in system_status:
                cpu_usage = system_status["cpu_usage"]
                if cpu_usage > self.config.max_cpu_usage:
                    logger.warning(f"CPU usage critical: {cpu_usage:.1%}")
                    self.send_kill_signal("auto_kill_cpu_critical")
                    return
            
            # Check if AETHERION is unreachable
            if not health_status.get("reachable", True):
                logger.warning("AETHERION is unreachable")
                self.send_kill_signal("auto_kill_unreachable")
                return
                
        except Exception as e:
            logger.error(f"Error in auto-kill check: {e}")
    
    def send_kill_signal(self, reason: str = "guardian_kill") -> bool:
        """Send kill signal to AETHERION"""
        try:
            logger.critical(f"Sending kill signal to AETHERION: {reason}")
            
            # Connect to AETHERION guardian port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.kill_timeout)
            
            try:
                sock.connect((self.config.aetherion_host, self.config.aetherion_port))
                sock.send("KILL".encode('utf-8'))
                sock.close()
                
                self.kill_count += 1
                logger.info(f"Kill signal sent successfully (count: {self.kill_count})")
                return True
                
            except socket.timeout:
                logger.error("Timeout sending kill signal")
                return False
            except ConnectionRefusedError:
                logger.error("Connection refused when sending kill signal")
                return False
            finally:
                sock.close()
                
        except Exception as e:
            logger.error(f"Error sending kill signal: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get guardian process status"""
        return {
            "running": self.running,
            "last_check": self.last_check,
            "kill_count": self.kill_count,
            "config": {
                "aetherion_host": self.config.aetherion_host,
                "aetherion_port": self.config.aetherion_port,
                "check_interval": self.config.check_interval,
                "enable_monitoring": self.config.enable_monitoring,
                "enable_auto_kill": self.config.enable_auto_kill
            },
            "monitoring_entries": len(self.monitoring_data)
        }
    
    def get_monitoring_data(self, limit: int = 100) -> list:
        """Get recent monitoring data"""
        return self.monitoring_data[-limit:] if self.monitoring_data else []


class GuardianCLI:
    """Command-line interface for guardian process"""
    
    def __init__(self):
        self.guardian = None
    
    def start_monitoring(self, config: GuardianConfig):
        """Start guardian monitoring"""
        self.guardian = GuardianProcess(config)
        self.guardian.start()
    
    def send_kill(self, reason: str = "manual_kill"):
        """Send kill signal"""
        if not self.guardian:
            config = GuardianConfig()
            self.guardian = GuardianProcess(config)
        
        success = self.guardian.send_kill_signal(reason)
        if success:
            print("Kill signal sent successfully")
        else:
            print("Failed to send kill signal")
            sys.exit(1)
    
    def get_status(self):
        """Get guardian status"""
        if not self.guardian:
            config = GuardianConfig()
            self.guardian = GuardianProcess(config)
        
        status = self.guardian.get_status()
        print(json.dumps(status, indent=2))
    
    def get_monitoring_data(self, limit: int = 10):
        """Get monitoring data"""
        if not self.guardian:
            config = GuardianConfig()
            self.guardian = GuardianProcess(config)
        
        data = self.guardian.get_monitoring_data(limit)
        print(json.dumps(data, indent=2))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AETHERION Guardian Process")
    parser.add_argument("command", choices=["monitor", "kill", "status", "data"])
    parser.add_argument("--host", default="localhost", help="AETHERION host")
    parser.add_argument("--port", type=int, default=8080, help="AETHERION guardian port")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--reason", default="manual_kill", help="Kill reason")
    parser.add_argument("--limit", type=int, default=10, help="Number of monitoring entries")
    parser.add_argument("--auto-kill", action="store_true", help="Enable auto-kill")
    parser.add_argument("--memory-limit", type=float, default=0.95, help="Memory usage limit for auto-kill")
    parser.add_argument("--cpu-limit", type=float, default=0.98, help="CPU usage limit for auto-kill")
    
    args = parser.parse_args()
    
    config = GuardianConfig(
        aetherion_host=args.host,
        aetherion_port=args.port,
        check_interval=args.interval,
        enable_auto_kill=args.auto_kill,
        max_memory_usage=args.memory_limit,
        max_cpu_usage=args.cpu_limit
    )
    
    cli = GuardianCLI()
    
    if args.command == "monitor":
        print("Starting AETHERION guardian monitoring...")
        cli.start_monitoring(config)
    elif args.command == "kill":
        cli.send_kill(args.reason)
    elif args.command == "status":
        cli.get_status()
    elif args.command == "data":
        cli.get_monitoring_data(args.limit)


if __name__ == "__main__":
    main() 