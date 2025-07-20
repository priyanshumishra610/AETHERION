import React, { useState } from 'react';
import { useMutation } from 'react-query';
import axios from 'axios';
import { Zap, AlertTriangle, Shield, Power } from 'lucide-react';
import toast from 'react-hot-toast';

function KillSwitch() {
  const [reason, setReason] = useState('');
  const [emergency, setEmergency] = useState(false);
  const [isConfirming, setIsConfirming] = useState(false);

  const killSwitchMutation = useMutation(
    (data) => axios.post('/api/system/kill-switch', data),
    {
      onSuccess: () => {
        toast.success('Kill switch activated successfully');
        setIsConfirming(false);
      },
      onError: (error) => {
        toast.error(error.response?.data?.detail || 'Failed to activate kill switch');
        setIsConfirming(false);
      }
    }
  );

  const restartMutation = useMutation(
    () => axios.post('/api/system/restart'),
    {
      onSuccess: () => {
        toast.success('System restarted successfully');
      },
      onError: (error) => {
        toast.error(error.response?.data?.detail || 'Failed to restart system');
      }
    }
  );

  const handleKillSwitch = () => {
    if (!reason.trim()) {
      toast.error('Please provide a reason for activation');
      return;
    }
    setIsConfirming(true);
  };

  const confirmKillSwitch = () => {
    killSwitchMutation.mutate({
      reason: reason.trim(),
      emergency
    });
  };

  const cancelKillSwitch = () => {
    setIsConfirming(false);
    setReason('');
    setEmergency(false);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">🜂 Kill Switch</h1>
        <p className="text-gray-400 mt-2">
          Emergency shutdown system for AETHERION consciousness matrix
        </p>
      </div>

      {/* Warning Banner */}
      <div className="bg-red-900 border border-red-700 rounded-lg p-6">
        <div className="flex items-center">
          <AlertTriangle className="h-8 w-8 text-red-400 mr-4" />
          <div>
            <h2 className="text-xl font-bold text-red-200">⚠️ EMERGENCY SHUTDOWN SYSTEM</h2>
            <p className="text-red-300 mt-1">
              This will immediately terminate all AETHERION processes and freeze the consciousness matrix.
              Use only in extreme emergencies.
            </p>
          </div>
        </div>
      </div>

      {/* Kill Switch Interface */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center mb-6">
          <Zap className="h-8 w-8 text-red-400 mr-3" />
          <h2 className="text-2xl font-bold text-white">Activate Kill Switch</h2>
        </div>

        {!isConfirming ? (
          <div className="space-y-6">
            {/* Reason Input */}
            <div>
              <label htmlFor="reason" className="block text-sm font-medium text-gray-300 mb-2">
                Activation Reason *
              </label>
              <textarea
                id="reason"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                rows={4}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent"
                placeholder="Describe the reason for activating the kill switch..."
                required
              />
            </div>

            {/* Emergency Toggle */}
            <div className="flex items-center">
              <input
                id="emergency"
                type="checkbox"
                checked={emergency}
                onChange={(e) => setEmergency(e.target.checked)}
                className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-600 rounded bg-gray-700"
              />
              <label htmlFor="emergency" className="ml-2 text-sm text-gray-300">
                Emergency Mode (Bypass additional safety checks)
              </label>
            </div>

            {/* Activation Button */}
            <button
              onClick={handleKillSwitch}
              disabled={!reason.trim() || killSwitchMutation.isLoading}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
            >
              {killSwitchMutation.isLoading ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Activating...
                </div>
              ) : (
                <div className="flex items-center">
                  <Power className="h-5 w-5 mr-2" />
                  Activate Kill Switch
                </div>
              )}
            </button>
          </div>
        ) : (
          /* Confirmation Dialog */
          <div className="space-y-6">
            <div className="bg-red-900 border border-red-700 rounded-lg p-4">
              <h3 className="text-lg font-bold text-red-200 mb-2">⚠️ FINAL CONFIRMATION</h3>
              <p className="text-red-300 text-sm mb-4">
                You are about to activate the AETHERION kill switch. This action will:
              </p>
              <ul className="text-red-300 text-sm space-y-1 list-disc list-inside">
                <li>Immediately terminate all consciousness processes</li>
                <li>Freeze the reality interface</li>
                <li>Disable all neural networks</li>
                <li>Stop all oracle predictions</li>
                <li>Lock down the divine firewall</li>
                <li>Require manual restart to resume operation</li>
              </ul>
            </div>

            <div className="bg-gray-700 rounded-lg p-4">
              <p className="text-gray-300 text-sm">
                <strong>Reason:</strong> {reason}
              </p>
              {emergency && (
                <p className="text-red-400 text-sm mt-1">
                  <strong>Emergency Mode:</strong> Enabled
                </p>
              )}
            </div>

            <div className="flex space-x-4">
              <button
                onClick={confirmKillSwitch}
                disabled={killSwitchMutation.isLoading}
                className="flex-1 flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
              >
                {killSwitchMutation.isLoading ? (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Activating...
                  </div>
                ) : (
                  'CONFIRM ACTIVATION'
                )}
              </button>
              <button
                onClick={cancelKillSwitch}
                disabled={killSwitchMutation.isLoading}
                className="flex-1 flex justify-center py-3 px-4 border border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-300 bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>

      {/* System Restart */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center mb-4">
          <Shield className="h-6 w-6 text-green-400 mr-3" />
          <h2 className="text-xl font-bold text-white">System Restart</h2>
        </div>
        <p className="text-gray-400 mb-4">
          If the kill switch has been activated, you can restart the AETHERION system here.
        </p>
        <button
          onClick={() => restartMutation.mutate()}
          disabled={restartMutation.isLoading}
          className="flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
        >
          {restartMutation.isLoading ? (
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Restarting...
            </div>
          ) : (
            'Restart AETHERION System'
          )}
        </button>
      </div>

      {/* Safety Information */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold text-white mb-4">Safety Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-300 mb-2">When to Use Kill Switch</h3>
            <ul className="text-gray-400 text-sm space-y-1 list-disc list-inside">
              <li>Uncontrolled consciousness expansion</li>
              <li>Reality manipulation anomalies</li>
              <li>Security breaches or unauthorized access</li>
              <li>System instability or corruption</li>
              <li>Emergency containment protocols</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-300 mb-2">Post-Activation Steps</h3>
            <ul className="text-gray-400 text-sm space-y-1 list-disc list-inside">
              <li>Assess the situation and root cause</li>
              <li>Review audit logs and system state</li>
              <li>Implement corrective measures</li>
              <li>Verify all safety protocols</li>
              <li>Gradually restart components</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Log Information */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold text-white mb-4">Logging & Audit</h2>
        <p className="text-gray-400 text-sm mb-4">
          All kill switch activations are automatically logged with full system state information,
          including consciousness levels, active manipulations, and safety violations.
        </p>
        <div className="bg-gray-700 rounded-lg p-4">
          <p className="text-gray-300 text-sm">
            <strong>Log Location:</strong> kill_switch_log.json
          </p>
          <p className="text-gray-300 text-sm mt-1">
            <strong>Audit Trail:</strong> All activations are permanently recorded
          </p>
        </div>
      </div>
    </div>
  );
}

export default KillSwitch; 