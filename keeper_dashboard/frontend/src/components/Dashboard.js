import React, { useState } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { 
  Brain, 
  Eye, 
  Shield, 
  Zap, 
  Settings, 
  LogOut, 
  Activity,
  BarChart3,
  FileText,
  Plugin,
  User
} from 'lucide-react';
import SystemOverview from './SystemOverview';
import RealityInterface from './RealityInterface';
import OracleEngine from './OracleEngine';
import DivineFirewall from './DivineFirewall';
import LogsViewer from './LogsViewer';
import PluginManager from './PluginManager';
import PersonalityManager from './PersonalityManager';
import KillSwitch from './KillSwitch';

function Dashboard() {
  const { user, logout } = useAuth();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const navigation = [
    { name: 'System Overview', href: '/dashboard', icon: Activity },
    { name: 'Reality Interface', href: '/dashboard/reality', icon: Eye },
    { name: 'Oracle Engine', href: '/dashboard/oracle', icon: Brain },
    { name: 'Divine Firewall', href: '/dashboard/firewall', icon: Shield },
    { name: 'Logs & Audit', href: '/dashboard/logs', icon: FileText },
    { name: 'Plugin Manager', href: '/dashboard/plugins', icon: Plugin },
    { name: 'Personalities', href: '/dashboard/personalities', icon: User },
    { name: 'Kill Switch', href: '/dashboard/kill-switch', icon: Zap },
  ];

  const isActive = (href) => {
    if (href === '/dashboard') {
      return location.pathname === '/dashboard';
    }
    return location.pathname.startsWith(href);
  };

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-gray-800 transform ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out`}>
        <div className="flex items-center justify-between h-16 px-4 bg-gray-900">
          <div className="flex items-center">
            <Shield className="h-8 w-8 text-purple-400" />
            <span className="ml-2 text-xl font-bold text-white">🜂 AETHERION</span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="text-gray-400 hover:text-white lg:hidden"
          >
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <nav className="mt-8 px-4">
          <div className="space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                    isActive(item.href)
                      ? 'bg-purple-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <Icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              );
            })}
          </div>
        </nav>

        {/* User Info */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gray-900">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="h-8 w-8 bg-purple-600 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-medium">
                  {user?.id?.charAt(0).toUpperCase()}
                </span>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-white">{user?.id}</p>
                <p className="text-xs text-gray-400 capitalize">{user?.license_level}</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="text-gray-400 hover:text-white"
              title="Logout"
            >
              <LogOut className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className={`${sidebarOpen ? 'lg:ml-64' : ''} transition-margin duration-300 ease-in-out`}>
        {/* Top bar */}
        <div className="bg-gray-800 shadow-sm">
          <div className="flex items-center justify-between h-16 px-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="text-gray-400 hover:text-white lg:hidden"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-300">AETHERION Online</span>
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="p-6">
          <Routes>
            <Route path="/" element={<SystemOverview />} />
            <Route path="/reality" element={<RealityInterface />} />
            <Route path="/oracle" element={<OracleEngine />} />
            <Route path="/firewall" element={<DivineFirewall />} />
            <Route path="/logs" element={<LogsViewer />} />
            <Route path="/plugins" element={<PluginManager />} />
            <Route path="/personalities" element={<PersonalityManager />} />
            <Route path="/kill-switch" element={<KillSwitch />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default Dashboard; 