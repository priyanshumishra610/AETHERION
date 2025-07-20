"""
Plugin Base Interface
Base class and interface for AETHERION plugins
"""

import abc
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Types of plugins"""
    NLP = "nlp"                    # Natural Language Processing
    MATH = "math"                  # Mathematical operations
    QUANTUM = "quantum"            # Quantum computing
    VISUALIZATION = "visualization" # Data visualization
    ANALYSIS = "analysis"          # Data analysis
    INTEGRATION = "integration"    # External system integration
    UTILITY = "utility"            # Utility functions
    CUSTOM = "custom"              # Custom functionality

class PluginStatus(Enum):
    """Plugin status states"""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class PluginMetadata:
    """Plugin metadata information"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    license: str = "MIT"
    homepage: Optional[str] = None
    repository: Optional[str] = None

@dataclass
class PluginConfig:
    """Plugin configuration"""
    enabled: bool = True
    auto_load: bool = True
    priority: int = 0
    timeout: int = 30
    max_memory: str = "512m"
    settings: Dict[str, Any] = field(default_factory=dict)

class PluginBase(abc.ABC):
    """
    Base class for all AETHERION plugins
    
    All plugins must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, 
                 metadata: PluginMetadata,
                 config: Optional[PluginConfig] = None):
        self.metadata = metadata
        self.config = config or PluginConfig()
        self.plugin_id = str(uuid.uuid4())
        self.status = PluginStatus.LOADED
        self.load_time = time.time()
        self.last_used = None
        self.usage_count = 0
        self.error_count = 0
        self.logger = logging.getLogger(f"plugin.{metadata.name}")
        
        # Plugin state
        self._initialized = False
        self._resources = {}
        
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def execute(self, 
                input_data: Any,
                parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the plugin's main functionality
        
        Args:
            input_data: Input data for the plugin
            parameters: Optional parameters for execution
            
        Returns:
            Any: Plugin execution result
        """
        pass
    
    @abc.abstractmethod
    def cleanup(self) -> bool:
        """
        Clean up plugin resources
        
        Returns:
            bool: True if cleanup successful, False otherwise
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information"""
        return {
            "plugin_id": self.plugin_id,
            "name": self.metadata.name,
            "version": self.metadata.version,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "load_time": self.load_time,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "error_count": self.error_count,
            "initialized": self._initialized
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "plugin_type": self.metadata.plugin_type.value,
            "dependencies": self.metadata.dependencies,
            "requirements": self.metadata.requirements,
            "tags": self.metadata.tags,
            "license": self.metadata.license,
            "homepage": self.metadata.homepage,
            "repository": self.metadata.repository
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration"""
        return {
            "enabled": self.config.enabled,
            "auto_load": self.config.auto_load,
            "priority": self.config.priority,
            "timeout": self.config.timeout,
            "max_memory": self.config.max_memory,
            "settings": self.config.settings
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update plugin configuration"""
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            return True
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")
            return False
    
    def enable(self) -> bool:
        """Enable the plugin"""
        if self.status == PluginStatus.ERROR:
            # Try to reinitialize if in error state
            if not self.initialize():
                return False
        
        self.config.enabled = True
        self.status = PluginStatus.ACTIVE
        self.logger.info(f"Plugin {self.metadata.name} enabled")
        return True
    
    def disable(self) -> bool:
        """Disable the plugin"""
        self.config.enabled = False
        self.status = PluginStatus.INACTIVE
        self.logger.info(f"Plugin {self.metadata.name} disabled")
        return True
    
    def reload(self) -> bool:
        """Reload the plugin"""
        try:
            # Cleanup first
            self.cleanup()
            
            # Reinitialize
            if self.initialize():
                self.status = PluginStatus.ACTIVE
                self.logger.info(f"Plugin {self.metadata.name} reloaded successfully")
                return True
            else:
                self.status = PluginStatus.ERROR
                self.logger.error(f"Failed to reload plugin {self.metadata.name}")
                return False
                
        except Exception as e:
            self.status = PluginStatus.ERROR
            self.logger.error(f"Error reloading plugin {self.metadata.name}: {e}")
            return False
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        # Default implementation - accept all input
        # Override in subclasses for specific validation
        return True
    
    def validate_parameters(self, parameters: Optional[Dict[str, Any]]) -> bool:
        """
        Validate execution parameters
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        # Default implementation - accept all parameters
        # Override in subclasses for specific validation
        return True
    
    def pre_execute(self, input_data: Any, parameters: Optional[Dict[str, Any]]) -> bool:
        """
        Pre-execution hook
        
        Args:
            input_data: Input data for execution
            parameters: Execution parameters
            
        Returns:
            bool: True if pre-execution successful, False otherwise
        """
        # Default implementation - always succeed
        # Override in subclasses for specific pre-execution logic
        return True
    
    def post_execute(self, result: Any, execution_time: float) -> None:
        """
        Post-execution hook
        
        Args:
            result: Execution result
            execution_time: Time taken for execution
        """
        # Update usage statistics
        self.last_used = time.time()
        self.usage_count += 1
        
        # Default implementation - do nothing
        # Override in subclasses for specific post-execution logic
        pass
    
    def handle_error(self, error: Exception) -> None:
        """
        Handle execution errors
        
        Args:
            error: The error that occurred
        """
        self.error_count += 1
        self.logger.error(f"Plugin {self.metadata.name} error: {error}")
        
        # Default implementation - log error
        # Override in subclasses for specific error handling
        pass
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get plugin resource usage information"""
        return {
            "memory_usage": self._get_memory_usage(),
            "cpu_usage": self._get_cpu_usage(),
            "execution_time": self._get_execution_time(),
            "resource_count": len(self._resources)
        }
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            process = psutil.Process()
            return process.cpu_percent()
        except ImportError:
            return 0.0
    
    def _get_execution_time(self) -> float:
        """Get average execution time"""
        # This would be implemented with actual timing data
        return 0.0
    
    def __str__(self) -> str:
        return f"Plugin({self.metadata.name} v{self.metadata.version})"
    
    def __repr__(self) -> str:
        return f"Plugin({self.metadata.name} v{self.metadata.version}, status={self.status.value})" 