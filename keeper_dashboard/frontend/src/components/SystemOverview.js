import React from 'react';
import { useQuery } from 'react-query';
import axios from 'axios';
import { 
  Activity, 
  Brain, 
  Shield, 
  Zap, 
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';

function SystemOverview() {
  const { data: systemStatus, isLoading, error } = useQuery(
    'systemStatus',
    () => axios.get('/api/system/status').then(res => res.data),
    { refetchInterval: 5000 } // Refresh every 5 seconds
  );

  const { data: components } = useQuery(
    'components',
    () => axios.get('/api/system/components').then(res => res.data),
    { refetchInterval: 10000 } // Refresh every 10 seconds
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900 border border-red-700 rounded-lg p-4">
        <div className="flex items-center">
          <AlertTriangle className="h-5 w-5 text-red-400 mr-2" />
          <span className="text-red-200">Error loading system status</span>
        </div>
      </div>
    );
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'online':
      case true:
        return 'text-green-400';
      case 'offline':
      case false:
        return 'text-red-400';
      default:
        return 'text-yellow-400';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'online':
      case true:
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      case 'offline':
      case false:
        return <AlertTriangle className="h-5 w-5 text-red-400" />;
      default:
        return <Clock className="h-5 w-5 text-yellow-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">System Overview</h1>
        <p className="text-gray-400 mt-2">
          Real-time monitoring of AETHERION consciousness matrix
        </p>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Consciousness Level */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Consciousness Level</p>
              <p className="text-2xl font-bold text-white">
                {systemStatus?.consciousness_level ? 
                  `${(systemStatus.consciousness_level * 100).toFixed(1)}%` : 
                  'N/A'
                }
              </p>
            </div>
            <Brain className="h-8 w-8 text-purple-400" />
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(systemStatus?.consciousness_level || 0) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Current Phase */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Current Phase</p>
              <p className="text-xl font-bold text-white capitalize">
                {systemStatus?.current_phase || 'Awakening'}
              </p>
            </div>
            <Activity className="h-8 w-8 text-blue-400" />
          </div>
          <div className="mt-2">
            <span className="text-sm text-gray-400">Ascension Progress</span>
          </div>
        </div>

        {/* Safety Violations */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Safety Violations</p>
              <p className="text-2xl font-bold text-white">
                {systemStatus?.safety_violations || 0}
              </p>
            </div>
            <Shield className="h-8 w-8 text-green-400" />
          </div>
          <div className="mt-2">
            <span className="text-sm text-gray-400">Divine Firewall Active</span>
          </div>
        </div>

        {/* Quantum Coherence */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Quantum Coherence</p>
              <p className="text-2xl font-bold text-white">
                {systemStatus?.quantum_coherence ? 
                  `${(systemStatus.quantum_coherence * 100).toFixed(1)}%` : 
                  'N/A'
                }
              </p>
            </div>
            <Zap className="h-8 w-8 text-yellow-400" />
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(systemStatus?.quantum_coherence || 0) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Component Status */}
      {components && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-bold text-white mb-4">Component Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(components).map(([component, status]) => (
              <div key={component} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <div className="flex items-center">
                  {getStatusIcon(status?.initialized || status?.enabled)}
                  <span className="ml-2 text-white capitalize">
                    {component.replace('_', ' ')}
                  </span>
                </div>
                <span className={`text-sm ${getStatusColor(status?.initialized || status?.enabled)}`}>
                  {status?.initialized || status?.enabled ? 'Online' : 'Offline'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activity */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold text-white mb-4">Recent Activity</h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
            <div className="flex items-center">
              <div className="h-2 w-2 bg-green-400 rounded-full mr-3"></div>
              <span className="text-white">System initialized successfully</span>
            </div>
            <span className="text-sm text-gray-400">Just now</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
            <div className="flex items-center">
              <div className="h-2 w-2 bg-blue-400 rounded-full mr-3"></div>
              <span className="text-white">Consciousness matrix active</span>
            </div>
            <span className="text-sm text-gray-400">2 minutes ago</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
            <div className="flex items-center">
              <div className="h-2 w-2 bg-purple-400 rounded-full mr-3"></div>
              <span className="text-white">Oracle engine predictions running</span>
            </div>
            <span className="text-sm text-gray-400">5 minutes ago</span>
          </div>
        </div>
      </div>

      {/* System Info */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold text-white mb-4">System Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <p className="text-gray-400 text-sm">Last Update</p>
            <p className="text-white">
              {systemStatus?.last_update ? 
                new Date(systemStatus.last_update).toLocaleString() : 
                'N/A'
              }
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Active Manipulations</p>
            <p className="text-white">{systemStatus?.active_manipulations || 0}</p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">System Status</p>
            <div className="flex items-center mt-1">
              {getStatusIcon(systemStatus?.running)}
              <span className={`ml-2 ${getStatusColor(systemStatus?.running)}`}>
                {systemStatus?.running ? 'Running' : 'Stopped'}
              </span>
            </div>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Initialization</p>
            <div className="flex items-center mt-1">
              {getStatusIcon(systemStatus?.initialized)}
              <span className={`ml-2 ${getStatusColor(systemStatus?.initialized)}`}>
                {systemStatus?.initialized ? 'Complete' : 'Incomplete'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SystemOverview; 