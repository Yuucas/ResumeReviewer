import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Search, BarChart3, Home, Clock } from 'lucide-react';

const Header = () => {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  return (
    <header className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">R</span>
            </div>
            <span className="text-xl font-bold text-gray-900">Resume Reviewer</span>
          </Link>

          <nav className="flex space-x-4">
            <Link
              to="/"
              className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive('/') ? 'bg-primary-100 text-primary-700' : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Home size={18} />
              <span>Home</span>
            </Link>

            <Link
              to="/search"
              className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive('/search') ? 'bg-primary-100 text-primary-700' : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Search size={18} />
              <span>Search</span>
            </Link>

            <Link
              to="/statistics"
              className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive('/statistics') ? 'bg-primary-100 text-primary-700' : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <BarChart3 size={18} />
              <span>Statistics</span>
            </Link>

            <Link
              to="/history"
              className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive('/history') || location.pathname.startsWith('/history/') ? 'bg-primary-100 text-primary-700' : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Clock size={18} />
              <span>History</span>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
