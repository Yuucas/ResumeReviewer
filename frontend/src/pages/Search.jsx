import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, Loader2, ArrowRight, Upload as UploadIcon } from 'lucide-react';
import SearchForm from '../components/search/SearchForm';
import ResultsList from '../components/results/ResultsList';
import UploadResume from '../components/upload/UploadResume';
import { apiService } from '../services/api';

const Search = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [dbInitialized, setDbInitialized] = useState(false);
  const [checkingHealth, setCheckingHealth] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [needsReindex, setNeedsReindex] = useState(false);

  useEffect(() => {
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const health = await apiService.checkHealth();
      setDbInitialized(health.database_initialized);
    } catch (err) {
      setError('Cannot connect to backend. Please ensure the API is running.');
    } finally {
      setCheckingHealth(false);
    }
  };

  const handleInitialize = async () => {
    setInitializing(true);
    setError(null);
    try {
      const result = await apiService.initializeSystem(false);
      if (result.success) {
        setDbInitialized(true);
        alert(`System initialized successfully!\n\nDocuments processed: ${result.documents_processed}\nChunks created: ${result.chunks_created}\nTime: ${result.processing_time}s`);
      }
    } catch (err) {
      setError(err.message || 'Failed to initialize system');
    } finally {
      setInitializing(false);
    }
  };

  const handleSearch = async (searchParams) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await apiService.searchCandidates(searchParams);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Search failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadSuccess = (result) => {
    setNeedsReindex(true);
    // Optionally hide upload form after successful upload
    // setShowUpload(false);
  };

  const handleReindex = async () => {
    setInitializing(true);
    setError(null);
    try {
      const result = await apiService.initializeSystem(true); // Force reindex
      if (result.success) {
        setNeedsReindex(false);
        alert(`Database re-indexed successfully!\n\nDocuments processed: ${result.documents_processed}\nChunks created: ${result.chunks_created}\nTime: ${result.processing_time}s`);
      }
    } catch (err) {
      setError(err.message || 'Failed to re-index database');
    } finally {
      setInitializing(false);
    }
  };

  if (checkingHealth) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex items-center justify-center">
          <Loader2 className="animate-spin text-primary-600" size={40} />
          <span className="ml-3 text-lg text-gray-600">Checking system status...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Find Best Candidates</h1>
          <p className="text-gray-600">
            Search through resumes using AI-powered semantic matching
          </p>
        </div>
        <button
          onClick={() => setShowUpload(!showUpload)}
          className={`btn-secondary flex items-center ${showUpload ? 'bg-primary-100 text-primary-700' : ''}`}
        >
          <UploadIcon className="mr-2" size={18} />
          {showUpload ? 'Hide Upload' : 'Upload Resume'}
        </button>
      </div>

      {/* Initialization Warning */}
      {!dbInitialized && (
        <div className="card bg-yellow-50 border-2 border-yellow-200 mb-6">
          <div className="flex items-start">
            <AlertCircle className="text-yellow-600 mt-1 flex-shrink-0" size={24} />
            <div className="ml-3 flex-grow">
              <h3 className="text-lg font-semibold text-yellow-900 mb-2">
                System Not Initialized
              </h3>
              <p className="text-yellow-800 mb-4">
                The database hasn't been initialized yet. Please initialize the system first to load and index all resumes.
              </p>
              <button
                onClick={handleInitialize}
                disabled={initializing}
                className="btn-primary disabled:opacity-50"
              >
                {initializing ? (
                  <>
                    <Loader2 className="animate-spin mr-2 inline" size={16} />
                    Initializing...
                  </>
                ) : (
                  'Initialize System'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="card bg-red-50 border-2 border-red-200 mb-6">
          <div className="flex items-start">
            <AlertCircle className="text-red-600 mt-1" size={24} />
            <div className="ml-3">
              <h3 className="text-lg font-semibold text-red-900">Error</h3>
              <p className="text-red-800">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Upload Section */}
      {showUpload && (
        <div className="mb-6">
          <UploadResume onUploadSuccess={handleUploadSuccess} />
        </div>
      )}

      {/* Reindex Notification */}
      {needsReindex && (
        <div className="card bg-orange-50 border-2 border-orange-200 mb-6">
          <div className="flex items-start">
            <AlertCircle className="text-orange-600 mt-1 flex-shrink-0" size={24} />
            <div className="ml-3 flex-grow">
              <h3 className="text-lg font-semibold text-orange-900 mb-2">
                Re-indexing Required
              </h3>
              <p className="text-orange-800 mb-4">
                New resumes have been uploaded. Please re-index the database to make them searchable.
              </p>
              <button
                onClick={handleReindex}
                disabled={initializing}
                className="btn-primary disabled:opacity-50"
              >
                {initializing ? (
                  <>
                    <Loader2 className="animate-spin mr-2 inline" size={16} />
                    Re-indexing...
                  </>
                ) : (
                  'Re-index Database'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading Message */}
      {loading && (
        <div className="card bg-blue-50 border-2 border-blue-200 mb-6">
          <div className="flex items-center">
            <Loader2 className="animate-spin text-blue-600 mr-3" size={24} />
            <div>
              <h3 className="text-lg font-semibold text-blue-900 mb-1">
                Analyzing Candidates...
              </h3>
              <p className="text-blue-800 text-sm">
                This may take 2-5 minutes as the AI analyzes each candidate's qualifications.
                Please be patient.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Search Form */}
      <SearchForm onSearch={handleSearch} loading={loading || !dbInitialized} />

      {/* Results */}
      {results && (
        <>
          <ResultsList
            results={results.results}
            processingTime={results.processing_time}
          />

          {/* View Detailed Analysis Button */}
          {results.results && results.results.length > 0 && (
            <div className="mt-6 flex justify-center">
              <button
                onClick={() => navigate('/analysis', { state: { searchData: results } })}
                className="btn-primary flex items-center text-lg px-8 py-4"
              >
                View Detailed Analysis
                <ArrowRight className="ml-2" size={20} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Search;
