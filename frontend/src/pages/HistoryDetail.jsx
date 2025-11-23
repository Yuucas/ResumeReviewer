import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, Clock, Users } from 'lucide-react';
import axios from 'axios';
import CandidateCard from '../components/results/CandidateCard';

const HistoryDetail = () => {
  const { searchId } = useParams();
  const navigate = useNavigate();
  const [searchData, setSearchData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSearchDetail();
  }, [searchId]);

  const fetchSearchDetail = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`http://localhost:8000/api/history/${searchId}`);
      setSearchData(response.data);
      setError(null);
    } catch (err) {
      if (err.response?.status === 404) {
        setError('Search not found');
      } else {
        setError(err.response?.data?.detail || 'Failed to load search details');
      }
      console.error('Error fetching search detail:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'long',
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

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
          <p>{error}</p>
        </div>
        <Link
          to="/history"
          className="inline-flex items-center text-blue-600 hover:text-blue-700"
        >
          <ArrowLeft size={20} className="mr-2" />
          Back to History
        </Link>
      </div>
    );
  }

  const { search, results } = searchData;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Back Button */}
      <Link
        to="/history"
        className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-6"
      >
        <ArrowLeft size={20} className="mr-2" />
        Back to History
      </Link>

      {/* Search Info */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">
          {search.job_title || `Search Results #${search.id}`}
        </h1>

        {/* Job Description */}
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Job Description:</h3>
          <p className="text-gray-700 leading-relaxed bg-gray-50 p-4 rounded-lg">
            {search.job_description}
          </p>
        </div>

        {/* Metadata */}
        <div className="flex flex-wrap gap-6 text-sm text-gray-600">
          <div className="flex items-center">
            <Clock size={16} className="mr-2 text-gray-500" />
            <span className="font-medium mr-1">Date:</span>
            {formatDate(search.created_at)}
          </div>
          <div className="flex items-center">
            <Users size={16} className="mr-2 text-gray-500" />
            <span className="font-medium mr-1">Results:</span>
            {results.length} {results.length === 1 ? 'candidate' : 'candidates'}
          </div>
          {search.processing_time && (
            <div>
              <span className="font-medium mr-1">Duration:</span>
              {formatDuration(search.processing_time)}
            </div>
          )}
          <div>
            <span className="font-medium mr-1">Requested:</span>
            Top {search.top_k}
          </div>
        </div>
      </div>

      {/* Results */}
      <div>
        <h2 className="text-xl font-bold text-gray-900 mb-4">
          Candidate Results ({results.length})
        </h2>

        {results.length === 0 ? (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
            <p className="text-gray-600">No candidates found for this search</p>
          </div>
        ) : (
          <div className="space-y-6">
            {results.map((candidate, index) => (
              <CandidateCard
                key={index}
                candidate={candidate}
                rank={index + 1}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default HistoryDetail;
