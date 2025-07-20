import React from 'react';
import { User } from 'lucide-react';

function PersonalityManager() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Personality Manager</h1>
        <p className="text-gray-400 mt-2">
          Multiverse personality switching and management
        </p>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center mb-4">
          <User className="h-6 w-6 text-blue-400 mr-3" />
          <h2 className="text-xl font-bold text-white">Available Personalities</h2>
        </div>
        <p className="text-gray-400">
          Personality manager will be implemented here.
        </p>
      </div>
    </div>
  );
}

export default PersonalityManager; 