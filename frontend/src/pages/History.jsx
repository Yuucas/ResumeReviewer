import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Clock, Users, Trash2, Eye, Search as SearchIcon } from 'lucide-react';
import axios from 'axios';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteLoading, setDeleteLoading] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:8000/api/history');
      setHistory(response.data.history);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load search history');
      console.error('Error fetching history:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (searchId) => {
    if (!window.confirm('Are you sure you want to delete this search?')) {
      return;
    }

    try {
      setDeleteLoading(searchId);
      await axios.delete(`http://localhost:8000/api/history/${searchId}`);
      // Remove from state
      setHistory(history.filter(item => item.id !== searchId));
    } catch (err) {
      alert('Failed to delete search: ' + (err.response?.data?.detail || err.message));
    } finally {
      setDeleteLoading(null);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}m ${secs}s`;
  };

  const truncateText = (text, maxLength = 150) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Search History</h1>
        <p className="text-gray-600">View and manage your previous job searches</p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
          <p>{error}</p>
        </div>
      )}

      {/* Empty State */}
      {!loading && history.length === 0 && (
        <div className="text-center py-16">
          <SearchIcon size={64} className="mx-auto text-gray-300 mb-4" />
          <h3 className="text-xl font-semibold text-gray-700 mb-2">No search history yet</h3>
          <p className="text-gray-500 mb-6">
            Your job searches will appear here
          </p>
          <Link
            to="/search"
            className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            Start Searching
          </Link>
        </div>
      )}

      {/* History List */}
      {history.length > 0 && (
        <div className="space-y-4">
          {history.map((item) => (
            <div
              key={item.id}
              className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition"
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  {/* Job Title */}
                  <div className="mb-3">
                    <h3 className="text-lg font-semibold text-gray-900 mb-1">
                      {item.job_title || `Job Search #${item.id}`}
                    </h3>
                    <p className="text-gray-700 leading-relaxed">
                      {truncateText(item.job_description)}
                    </p>
                  </div>

                  {/* Metadata */}
                  <div className="flex flex-wrap gap-4 text-sm text-gray-600">
                    <div className="flex items-center">
                      <Clock size={16} className="mr-1" />
                      {formatDate(item.created_at)}
                    </div>
                    <div className="flex items-center">
                      <Users size={16} className="mr-1" />
                      {item.result_count} {item.result_count === 1 ? 'result' : 'results'}
                    </div>
                    {item.processing_time && (
                      <div className="text-gray-500">
                        Duration: {formatDuration(item.processing_time)}
                      </div>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 ml-4">
                  <Link
                    to={`/history/${item.id}`}
                    className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition text-sm"
                  >
                    <Eye size={16} className="mr-1" />
                    View Results
                  </Link>
                  <button
                    onClick={() => handleDelete(item.id)}
                    disabled={deleteLoading === item.id}
                    className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition text-sm disabled:opacity-50"
                  >
                    {deleteLoading === item.id ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                    ) : (
                      <>
                        <Trash2 size={16} className="mr-1" />
                        Delete
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default History;
