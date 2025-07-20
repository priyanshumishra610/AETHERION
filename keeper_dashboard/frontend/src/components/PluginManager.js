import React from 'react';
import { Plugin } from 'lucide-react';

function PluginManager() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Plugin Manager</h1>
        <p className="text-gray-400 mt-2">
          Dynamic plugin system management
        </p>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center mb-4">
          <Plugin className="h-6 w-6 text-purple-400 mr-3" />
          <h2 className="text-xl font-bold text-white">Available Plugins</h2>
        </div>
        <p className="text-gray-400">
          Plugin manager will be implemented here.
        </p>
      </div>
    </div>
  );
}

export default PluginManager; 