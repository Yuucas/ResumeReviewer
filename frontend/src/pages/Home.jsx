import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Search, BarChart3, Zap, Shield, CheckCircle, AlertCircle } from 'lucide-react';
import { apiService } from '../services/api';

const Home = () => {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const data = await apiService.checkHealth();
      setHealth(data);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealth({ status: 'error', database_initialized: false });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gradient-to-b from-primary-50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            AI-Powered Resume Screening
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Find the best candidates in seconds using advanced RAG technology and local LLMs
          </p>

          {!loading && health && (
            <div className="flex justify-center mb-8">
              <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full ${
                health.status === 'healthy' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  health.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-sm font-medium">
                  {health.status === 'healthy' ? 'System Online' : 'System Offline'}
                </span>
              </div>
            </div>
          )}

          {!loading && health && !health.database_initialized && health.status === 'healthy' && (
            <div className="max-w-2xl mx-auto mb-8">
              <div className="card bg-yellow-50 border-2 border-yellow-200">
                <div className="flex items-start text-left">
                  <AlertCircle className="text-yellow-600 mt-1 flex-shrink-0" size={24} />
                  <div className="ml-3">
                    <h3 className="font-semibold text-yellow-900 mb-1">Database Not Initialized</h3>
                    <p className="text-sm text-yellow-800">
                      Initialize the system to start using the application.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="flex justify-center space-x-4">
            <Link to="/search" className="btn-primary text-lg px-8 py-3">
              <Search className="inline-block mr-2" size={20} />
              Start Searching
            </Link>
            <Link to="/statistics" className="btn-secondary text-lg px-8 py-3">
              <BarChart3 className="inline-block mr-2" size={20} />
              View Statistics
            </Link>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
          Why Choose Resume RAG?
        </h2>

        <div className="grid md:grid-cols-3 gap-8">
          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Zap className="text-primary-600" size={24} />
            </div>
            <h3 className="text-xl font-semibold mb-2">Lightning Fast</h3>
            <p className="text-gray-600">
              Process 100+ resumes in minutes with semantic search
            </p>
          </div>

          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Shield className="text-primary-600" size={24} />
            </div>
            <h3 className="text-xl font-semibold mb-2">Privacy First</h3>
            <p className="text-gray-600">
              All processing happens locally. Your data stays secure
            </p>
          </div>

          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="text-primary-600" size={24} />
            </div>
            <h3 className="text-xl font-semibold mb-2">Intelligent Analysis</h3>
            <p className="text-gray-600">
              LLM-powered evaluation with detailed recommendations
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
