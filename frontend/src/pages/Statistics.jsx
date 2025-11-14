import React, { useState, useEffect } from 'react';
import { BarChart3, Users, FileText, TrendingUp, Loader2, AlertCircle, RefreshCw } from 'lucide-react';
import { Link } from 'react-router-dom';
import { apiService } from '../services/api';

const Statistics = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dbInitialized, setDbInitialized] = useState(false);

  useEffect(() => {
    checkHealthAndFetchStats();
  }, []);

  const checkHealthAndFetchStats = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const health = await apiService.checkHealth();
      setDbInitialized(health.database_initialized);
      
      if (health.database_initialized) {
        const data = await apiService.getStatistics();
        setStats(data);
      }
    } catch (err) {
      setError(err.message || 'Failed to load statistics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="animate-spin text-primary-600" size={40} />
          <span className="ml-3 text-lg text-gray-600">Loading statistics...</span>
        </div>
      </div>
    );
  }

  if (!dbInitialized) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">System Statistics</h1>
          <p className="text-gray-600">Overview of resume database and system metrics</p>
        </div>
        <div className="card bg-yellow-50 border-2 border-yellow-200">
          <div className="flex items-start">
            <AlertCircle className="text-yellow-600 mt-1 flex-shrink-0" size={40} />
            <div className="ml-4 flex-grow">
              <h3 className="text-xl font-semibold text-yellow-900 mb-3">
                Database Not Initialized
              </h3>
              <p className="text-yellow-800 mb-4">
                The system has not been initialized yet. Please go to the Search page 
                and initialize the system first.
              </p>
              <Link to="/search" className="btn-primary inline-block">
                Go to Search Page
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">System Statistics</h1>
          <p className="text-gray-600">Overview of resume database and system metrics</p>
        </div>
        <div className="card bg-red-50 border-2 border-red-200">
          <div className="flex items-start">
            <AlertCircle className="text-red-600 mt-1" size={24} />
            <div className="ml-3">
              <h3 className="text-lg font-semibold text-red-900 mb-2">Error</h3>
              <p className="text-red-800 mb-4">{error}</p>
              <button onClick={checkHealthAndFetchStats} className="btn-primary">
                <RefreshCw className="inline mr-2" size={16} />
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">System Statistics</h1>
        <p className="text-gray-600">Overview of resume database and system metrics</p>
      </div>

      <div className="grid md:grid-cols-4 gap-6 mb-8">
        <div className="card bg-gradient-to-br from-blue-50 to-blue-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Total Resumes</p>
              <p className="text-3xl font-bold text-gray-900">{stats?.total_resumes || 0}</p>
            </div>
            <Users className="text-primary-600" size={40} />
          </div>
        </div>

        <div className="card bg-gradient-to-br from-green-50 to-green-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Total Chunks</p>
              <p className="text-3xl font-bold text-gray-900">{stats?.total_chunks || 0}</p>
            </div>
            <FileText className="text-green-600" size={40} />
          </div>
        </div>

        <div className="card bg-gradient-to-br from-purple-50 to-purple-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Avg. Experience</p>
              <p className="text-3xl font-bold text-gray-900">
                {stats?.average_experience?.toFixed(1) || 0}y
              </p>
            </div>
            <TrendingUp className="text-purple-600" size={40} />
          </div>
        </div>

        <div className="card bg-gradient-to-br from-orange-50 to-orange-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Database Size</p>
              <p className="text-3xl font-bold text-gray-900">
                {stats?.database_size_mb?.toFixed(1) || 0}MB
              </p>
            </div>
            <BarChart3 className="text-orange-600" size={40} />
          </div>
        </div>
      </div>

      {stats?.roles && Object.keys(stats.roles).length > 0 && (
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Role Distribution</h2>
          <div className="space-y-4">
            {Object.entries(stats.roles).map(([role, count]) => {
              const percentage = stats.total_resumes > 0 
                ? ((count / stats.total_resumes) * 100).toFixed(1) 
                : 0;
              return (
                <div key={role}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-700 font-medium capitalize">
                      {role.replace(/_/g, ' ')}
                    </span>
                    <span className="text-gray-600">
                      {count} ({percentage}%)
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-primary-600 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default Statistics;
