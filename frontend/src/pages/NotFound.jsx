import React from 'react';
import { Link } from 'react-router-dom';
import { Home, Search } from 'lucide-react';

const NotFound = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <h1 className="text-9xl font-bold text-primary-600">404</h1>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Page Not Found</h2>
        <p className="text-gray-600 mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="flex justify-center space-x-4">
          <Link to="/" className="btn-primary">
            <Home className="inline mr-2" size={18} />
            Go Home
          </Link>
          <Link to="/search" className="btn-secondary">
            <Search className="inline mr-2" size={18} />
            Search Candidates
          </Link>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
