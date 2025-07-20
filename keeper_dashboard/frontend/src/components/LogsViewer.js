import React from 'react';
import { FileText } from 'lucide-react';

function LogsViewer() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Logs & Audit</h1>
        <p className="text-gray-400 mt-2">
          System logs and audit trail viewer
        </p>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center mb-4">
          <FileText className="h-6 w-6 text-yellow-400 mr-3" />
          <h2 className="text-xl font-bold text-white">Audit Logs</h2>
        </div>
        <p className="text-gray-400">
          Logs viewer will be implemented here.
        </p>
      </div>
    </div>
  );
}

export default LogsViewer; 