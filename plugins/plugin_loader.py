"""
Dynamic Plugin Loader for AETHERION
Handles plugin discovery, validation, and dynamic loading
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Optional, Type
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .plugin_base import PluginBase, PluginMetadata, PluginConfig

logger = logging.getLogger(__name__)


class PluginLoader:
    """Dynamic plugin loader with validation and lifecycle management"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.loaded_plugins: Dict[str, PluginBase] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugins directory"""
        plugin_files = []
        
        if not self.plugins_dir.exists():
            logger.warning(f"Plugins directory {self.plugins_dir} does not exist")
            return plugin_files
            
        for file_path in self.plugins_dir.rglob("*.py"):
            if file_path.name.startswith("__"):
                continue
            if file_path.name == "plugin_base.py":
                continue
            if file_path.name == "plugin_loader.py":
                continue
                
            # Convert file path to module path
            relative_path = file_path.relative_to(self.plugins_dir)
            module_path = str(relative_path).replace("/", ".").replace("\\", ".").replace(".py", "")
            plugin_files.append(module_path)
            
        logger.info(f"Discovered {len(plugin_files)} potential plugins: {plugin_files}")
        return plugin_files
    
    def validate_plugin(self, plugin_class: Type[PluginBase]) -> bool:
        """Validate plugin class meets requirements"""
        try:
            # Check if it's a subclass of PluginBase
            if not issubclass(plugin_class, PluginBase):
                logger.error(f"Plugin {plugin_class.__name__} must inherit from PluginBase")
                return False
                
            # Check if it's not the base class itself
            if plugin_class == PluginBase:
                return False
                
            # Check for required methods
            required_methods = ['initialize', 'execute', 'cleanup']
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    logger.error(f"Plugin {plugin_class.__name__} missing required method: {method}")
                    return False
                    
            # Validate metadata
            if hasattr(plugin_class, 'metadata'):
                metadata = plugin_class.metadata
                if not isinstance(metadata, PluginMetadata):
                    logger.error(f"Plugin {plugin_class.__name__} has invalid metadata")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating plugin {plugin_class.__name__}: {e}")
            return False
    
    def load_plugin(self, module_path: str) -> Optional[PluginBase]:
        """Load and instantiate a plugin from module path"""
        try:
            # Add plugins directory to path if not already there
            plugins_parent = str(self.plugins_dir.parent)
            if plugins_parent not in sys.path:
                sys.path.insert(0, plugins_parent)
            
            # Import the module
            module = importlib.import_module(f"plugins.{module_path}")
            
            # Find plugin classes in the module
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginBase) and 
                    obj != PluginBase):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                logger.warning(f"No plugin classes found in {module_path}")
                return None
                
            # Use the first valid plugin class
            plugin_class = plugin_classes[0]
            
            if not self.validate_plugin(plugin_class):
                return None
                
            # Instantiate the plugin
            plugin_instance = plugin_class()
            
            # Store metadata
            if hasattr(plugin_class, 'metadata'):
                self.plugin_metadata[plugin_instance.name] = plugin_class.metadata
                
            logger.info(f"Successfully loaded plugin: {plugin_instance.name}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Error loading plugin {module_path}: {e}")
            return None
    
    def load_all_plugins(self) -> Dict[str, PluginBase]:
        """Load all discovered plugins"""
        plugin_files = self.discover_plugins()
        
        for module_path in plugin_files:
            plugin = self.load_plugin(module_path)
            if plugin:
                self.loaded_plugins[plugin.name] = plugin
                
        logger.info(f"Loaded {len(self.loaded_plugins)} plugins")
        return self.loaded_plugins
    
    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """Get a loaded plugin by name"""
        return self.loaded_plugins.get(name)
    
    def list_plugins(self) -> List[Dict]:
        """List all loaded plugins with their metadata"""
        plugins_info = []
        for name, plugin in self.loaded_plugins.items():
            info = {
                "name": name,
                "version": getattr(plugin, 'version', 'unknown'),
                "description": getattr(plugin, 'description', ''),
                "status": "loaded",
                "metadata": self.plugin_metadata.get(name, {})
            }
            plugins_info.append(info)
        return plugins_info
    
    def execute_plugin(self, name: str, input_data: dict = None) -> dict:
        """Execute a plugin with input data"""
        plugin = self.get_plugin(name)
        if not plugin:
            raise ValueError(f"Plugin {name} not found")
            
        try:
            # Initialize if not already done
            if not getattr(plugin, '_initialized', False):
                plugin.initialize()
                plugin._initialized = True
                
            # Execute the plugin
            result = plugin.execute(input_data or {})
            return {
                "success": True,
                "plugin": name,
                "result": result,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error executing plugin {name}: {e}")
            return {
                "success": False,
                "plugin": name,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    def execute_plugin_async(self, name: str, input_data: dict = None):
        """Execute a plugin asynchronously"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            self.executor, 
            self.execute_plugin, 
            name, 
            input_data
        )
    
    def cleanup_plugin(self, name: str):
        """Cleanup a specific plugin"""
        plugin = self.get_plugin(name)
        if plugin:
            try:
                plugin.cleanup()
                logger.info(f"Cleaned up plugin: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up plugin {name}: {e}")
    
    def cleanup_all_plugins(self):
        """Cleanup all loaded plugins"""
        for name in list(self.loaded_plugins.keys()):
            self.cleanup_plugin(name)
        self.loaded_plugins.clear()
        self.plugin_metadata.clear()
        logger.info("Cleaned up all plugins")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_all_plugins()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 