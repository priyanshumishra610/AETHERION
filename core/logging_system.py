"""
Centralized Logging System for AETHERION
Provides immutable append-only logs, structured logging, and dashboard integration
"""

import os
import sys
import time
import json
import hashlib
import threading
import logging
import logging.handlers
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import gzip
import shutil
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for AETHERION"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(Enum):
    """Log categories for different AETHERION components"""
    SYSTEM = "system"
    ORACLE = "oracle"
    REALITY = "reality"
    FIREWALL = "firewall"
    PLUGINS = "plugins"
    PERSONALITIES = "personalities"
    SAFETY = "safety"
    QUANTUM = "quantum"
    KEEPER = "keeper"
    AUDIT = "audit"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: float
    level: str
    category: str
    component: str
    message: str
    data: Dict[str, Any]
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    hash: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging system"""
    log_directory: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_compression: bool = True
    enable_audit_logging: bool = True
    enable_structured_logging: bool = True
    log_retention_days: int = 30
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    log_format: str = "json"
    timezone: str = "UTC"


class ImmutableLogWriter:
    """Immutable append-only log writer"""
    
    def __init__(self, log_file: str, config: LoggingConfig):
        self.log_file = log_file
        self.config = config
        self.lock = threading.Lock()
        self.current_size = 0
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize file size
        if os.path.exists(log_file):
            self.current_size = os.path.getsize(log_file)
    
    def write_entry(self, entry: LogEntry) -> bool:
        """Write a log entry (immutable append-only)"""
        with self.lock:
            try:
                # Generate hash for entry integrity
                entry.hash = self._generate_entry_hash(entry)
                
                # Convert to JSON
                entry_json = json.dumps(asdict(entry), separators=(',', ':'))
                entry_line = entry_json + '\n'
                
                # Check file size before writing
                if self.current_size + len(entry_line.encode('utf-8')) > self.config.max_file_size:
                    self._rotate_log_file()
                
                # Append to file
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(entry_line)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                self.current_size += len(entry_line.encode('utf-8'))
                return True
                
            except Exception as e:
                logger.error(f"Failed to write log entry: {e}")
                return False
    
    def _generate_entry_hash(self, entry: LogEntry) -> str:
        """Generate hash for log entry integrity"""
        entry_data = asdict(entry)
        entry_data.pop('hash', None)  # Remove hash field for calculation
        entry_string = json.dumps(entry_data, separators=(',', ':'), sort_keys=True)
        return hashlib.sha256(entry_string.encode()).hexdigest()
    
    def _rotate_log_file(self):
        """Rotate log file when size limit reached"""
        try:
            # Create backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.log_file}.{timestamp}"
            
            # Move current file to backup
            shutil.move(self.log_file, backup_file)
            
            # Compress backup if enabled
            if self.config.enable_compression:
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(f"{backup_file}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(backup_file)
                backup_file = f"{backup_file}.gz"
            
            # Reset current size
            self.current_size = 0
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            logger.info(f"Log file rotated: {backup_file}")
            
        except Exception as e:
            logger.error(f"Failed to rotate log file: {e}")
    
    def _cleanup_old_backups(self):
        """Cleanup old backup files"""
        try:
            log_dir = os.path.dirname(self.log_file)
            base_name = os.path.basename(self.log_file)
            
            # Find backup files
            backup_files = []
            for file in os.listdir(log_dir):
                if file.startswith(base_name + "."):
                    backup_files.append(os.path.join(log_dir, file))
            
            # Sort by modification time
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove old backups
            for backup_file in backup_files[self.config.backup_count:]:
                os.remove(backup_file)
                logger.info(f"Removed old backup: {backup_file}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")


class StructuredLogger:
    """Structured logger for AETHERION components"""
    
    def __init__(self, component: str, category: LogCategory, config: LoggingConfig):
        self.component = component
        self.category = category
        self.config = config
        self.trace_id = None
        self.session_id = None
        self.user_id = None
        
        # Initialize log writer
        log_filename = f"{category.value}_{component}.log"
        log_path = os.path.join(config.log_directory, log_filename)
        self.writer = ImmutableLogWriter(log_path, config)
        
        # Setup standard logger
        self.logger = logging.getLogger(f"aetherion.{category.value}.{component}")
        self._setup_standard_logger()
    
    def _setup_standard_logger(self):
        """Setup standard Python logger"""
        if not self.config.enable_console_logging and not self.config.enable_file_logging:
            return
            
        # Create formatter
        if self.config.log_format == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if self.config.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.enable_file_logging:
            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.config.log_directory, f"{self.component}.log"),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.setLevel(logging.DEBUG)
    
    def set_context(self, trace_id: str = None, session_id: str = None, user_id: str = None):
        """Set logging context"""
        if trace_id:
            self.trace_id = trace_id
        if session_id:
            self.session_id = session_id
        if user_id:
            self.user_id = user_id
    
    def _create_entry(self, level: LogLevel, message: str, data: Dict[str, Any] = None) -> LogEntry:
        """Create a log entry"""
        return LogEntry(
            timestamp=time.time(),
            level=level.value,
            category=self.category.value,
            component=self.component,
            message=message,
            data=data or {},
            trace_id=self.trace_id,
            session_id=self.session_id,
            user_id=self.user_id
        )
    
    def debug(self, message: str, data: Dict[str, Any] = None):
        """Log debug message"""
        entry = self._create_entry(LogLevel.DEBUG, message, data)
        self.writer.write_entry(entry)
        self.logger.debug(message)
    
    def info(self, message: str, data: Dict[str, Any] = None):
        """Log info message"""
        entry = self._create_entry(LogLevel.INFO, message, data)
        self.writer.write_entry(entry)
        self.logger.info(message)
    
    def warning(self, message: str, data: Dict[str, Any] = None):
        """Log warning message"""
        entry = self._create_entry(LogLevel.WARNING, message, data)
        self.writer.write_entry(entry)
        self.logger.warning(message)
    
    def error(self, message: str, data: Dict[str, Any] = None):
        """Log error message"""
        entry = self._create_entry(LogLevel.ERROR, message, data)
        self.writer.write_entry(entry)
        self.logger.error(message)
    
    def critical(self, message: str, data: Dict[str, Any] = None):
        """Log critical message"""
        entry = self._create_entry(LogLevel.CRITICAL, message, data)
        self.writer.write_entry(entry)
        self.logger.critical(message)
    
    def exception(self, message: str, exc_info: bool = True, data: Dict[str, Any] = None):
        """Log exception with traceback"""
        if data is None:
            data = {}
        
        import traceback
        data['traceback'] = traceback.format_exc()
        
        entry = self._create_entry(LogLevel.ERROR, message, data)
        self.writer.write_entry(entry)
        self.logger.exception(message)


class LogManager:
    """Centralized log manager for AETHERION"""
    
    def __init__(self, config: LoggingConfig = None):
        self.config = config or LoggingConfig()
        self.loggers: Dict[str, StructuredLogger] = {}
        self.audit_logger = None
        
        # Ensure log directory exists
        os.makedirs(self.config.log_directory, exist_ok=True)
        
        # Initialize audit logger
        if self.config.enable_audit_logging:
            self.audit_logger = StructuredLogger("audit", LogCategory.AUDIT, self.config)
    
    def get_logger(self, component: str, category: LogCategory = LogCategory.SYSTEM) -> StructuredLogger:
        """Get or create a structured logger"""
        logger_key = f"{category.value}_{component}"
        
        if logger_key not in self.loggers:
            self.loggers[logger_key] = StructuredLogger(component, category, self.config)
        
        return self.loggers[logger_key]
    
    def audit_event(self, event_type: str, details: Dict[str, Any], user_id: str = None):
        """Log an audit event"""
        if self.audit_logger:
            self.audit_logger.info(f"AUDIT: {event_type}", {
                "event_type": event_type,
                "details": details,
                "user_id": user_id
            })
    
    def get_logs(self, 
                 category: LogCategory = None,
                 component: str = None,
                 level: LogLevel = None,
                 start_time: float = None,
                 end_time: float = None,
                 limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve logs with filtering"""
        logs = []
        
        # Determine which log files to read
        if category and component:
            log_files = [os.path.join(self.config.log_directory, f"{category.value}_{component}.log")]
        elif category:
            log_files = [f for f in os.listdir(self.config.log_directory) 
                        if f.startswith(f"{category.value}_") and f.endswith('.log')]
            log_files = [os.path.join(self.config.log_directory, f) for f in log_files]
        else:
            log_files = [os.path.join(self.config.log_directory, f) for f in os.listdir(self.config.log_directory)
                        if f.endswith('.log')]
        
        for log_file in log_files:
            if not os.path.exists(log_file):
                continue
                
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(logs) >= limit:
                            break
                            
                        try:
                            entry = json.loads(line.strip())
                            
                            # Apply filters
                            if level and entry.get('level') != level.value:
                                continue
                            if start_time and entry.get('timestamp', 0) < start_time:
                                continue
                            if end_time and entry.get('timestamp', 0) > end_time:
                                continue
                            
                            logs.append(entry)
                            
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return logs[:limit]
    
    def export_logs(self, 
                   output_file: str,
                   category: LogCategory = None,
                   component: str = None,
                   start_time: float = None,
                   end_time: float = None) -> bool:
        """Export logs to file"""
        try:
            logs = self.get_logs(category, component, start_time=start_time, end_time=end_time, limit=100000)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"Exported {len(logs)} log entries to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            return False
    
    def cleanup_old_logs(self):
        """Cleanup logs older than retention period"""
        try:
            cutoff_time = time.time() - (self.config.log_retention_days * 24 * 3600)
            
            for filename in os.listdir(self.config.log_directory):
                file_path = os.path.join(self.config.log_directory, filename)
                
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        logger.info(f"Removed old log file: {filename}")
                        
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "total_log_files": 0,
            "total_log_entries": 0,
            "categories": {},
            "components": {},
            "levels": {},
            "disk_usage": 0
        }
        
        try:
            for filename in os.listdir(self.config.log_directory):
                if filename.endswith('.log'):
                    file_path = os.path.join(self.config.log_directory, filename)
                    stats["total_log_files"] += 1
                    stats["disk_usage"] += os.path.getsize(file_path)
                    
                    # Parse filename for category and component
                    if '_' in filename:
                        parts = filename.replace('.log', '').split('_', 1)
                        if len(parts) == 2:
                            category, component = parts
                            stats["categories"][category] = stats["categories"].get(category, 0) + 1
                            stats["components"][component] = stats["components"].get(component, 0) + 1
                    
                    # Count entries in file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            entry_count = sum(1 for line in f if line.strip())
                            stats["total_log_entries"] += entry_count
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.error(f"Failed to get log statistics: {e}")
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get logging system health"""
        stats = self.get_log_statistics()
        
        health = {
            "status": "healthy",
            "disk_usage_mb": round(stats["disk_usage"] / (1024 * 1024), 2),
            "log_files": stats["total_log_files"],
            "log_entries": stats["total_log_entries"],
            "warnings": []
        }
        
        # Check for potential issues
        if stats["disk_usage"] > 1024 * 1024 * 1024:  # 1GB
            health["status"] = "warning"
            health["warnings"].append("Log disk usage exceeds 1GB")
        
        if stats["total_log_files"] > 100:
            health["status"] = "warning"
            health["warnings"].append("Too many log files")
        
        return health


# Global log manager instance
_log_manager = None


def get_log_manager() -> LogManager:
    """Get global log manager instance"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager


def get_logger(component: str, category: LogCategory = LogCategory.SYSTEM) -> StructuredLogger:
    """Get a structured logger for a component"""
    return get_log_manager().get_logger(component, category)


def audit_event(event_type: str, details: Dict[str, Any], user_id: str = None):
    """Log an audit event"""
    get_log_manager().audit_event(event_type, details, user_id) 