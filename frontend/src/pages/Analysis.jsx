import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, CheckCircle, AlertCircle, Clock, Mail, Briefcase } from 'lucide-react';

const Analysis = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const searchData = location.state?.searchData;

  if (!searchData || !searchData.results) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <AlertCircle className="mx-auto text-red-500 mb-4" size={48} />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">No Results Available</h2>
          <p className="text-gray-600 mb-6">
            Please perform a search first to view the analysis.
          </p>
          <button
            onClick={() => navigate('/search')}
            className="btn-primary"
          >
            Go to Search
          </button>
        </div>
      </div>
    );
  }

  const { query, results, total_candidates, processing_time } = searchData;

  const getRecommendationColor = (recommendation) => {
    const colors = {
      'strongly_recommend': 'bg-green-100 text-green-800 border-green-300',
      'recommend': 'bg-blue-100 text-blue-800 border-blue-300',
      'consider': 'bg-yellow-100 text-yellow-800 border-yellow-300',
      'not_recommend': 'bg-red-100 text-red-800 border-red-300',
    };
    return colors[recommendation] || 'bg-gray-100 text-gray-800 border-gray-300';
  };

  const getRecommendationIcon = (recommendation) => {
    if (recommendation === 'strongly_recommend' || recommendation === 'recommend') {
      return <CheckCircle className="inline mr-2" size={20} />;
    }
    return <AlertCircle className="inline mr-2" size={20} />;
  };

  const formatRecommendation = (rec) => {
    return rec.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={() => navigate('/search')}
          className="flex items-center text-primary-600 hover:text-primary-700 mb-4"
        >
          <ArrowLeft className="mr-2" size={20} />
          Back to Search
        </button>
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Candidate Analysis</h1>
        <p className="text-gray-600">Detailed breakdown of top matching candidates</p>
      </div>

      {/* Search Info */}
      <div className="card mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-gray-600 mb-1">Search Query</p>
            <p className="font-semibold text-gray-900">{query}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600 mb-1">Candidates Found</p>
            <p className="font-semibold text-gray-900">{total_candidates}</p>
          </div>
          <div className="flex items-center">
            <Clock className="text-gray-400 mr-2" size={18} />
            <div>
              <p className="text-sm text-gray-600">Processing Time</p>
              <p className="font-semibold text-gray-900">{processing_time}s</p>
            </div>
          </div>
        </div>
      </div>

      {/* Results List */}
      <div className="space-y-6">
        {results.map((candidate, index) => (
          <div key={index} className="card hover:shadow-xl transition-shadow">
            {/* Header */}
            <div className="flex justify-between items-start mb-4 pb-4 border-b border-gray-200">
              <div className="flex-grow">
                <div className="flex items-center mb-2">
                  <span className="text-3xl font-bold text-primary-600 mr-3">
                    #{index + 1}
                  </span>
                  <h3 className="text-2xl font-bold text-gray-900">
                    {candidate.filename}
                  </h3>
                </div>
                <div className="flex flex-wrap gap-4 text-sm text-gray-600">
                  {candidate.email && (
                    <div className="flex items-center">
                      <Mail size={16} className="mr-1" />
                      {candidate.email}
                    </div>
                  )}
                  {candidate.experience_years !== null && candidate.experience_years !== undefined && (
                    <div className="flex items-center">
                      <Briefcase size={16} className="mr-1" />
                      {candidate.experience_years} years experience
                    </div>
                  )}
                  {candidate.role && (
                    <div className="flex items-center">
                      <span className="px-2 py-1 bg-gray-100 rounded text-xs font-medium">
                        {candidate.role}
                      </span>
                    </div>
                  )}
                </div>
              </div>
              <div className="text-right ml-4">
                <div className="text-4xl font-bold text-primary-600 mb-1">
                  {candidate.match_score}
                </div>
                <div className="text-sm text-gray-600">Match Score</div>
              </div>
            </div>

            {/* Recommendation Badge */}
            <div className="mb-4">
              <span className={`inline-flex items-center px-4 py-2 rounded-lg border-2 font-semibold ${getRecommendationColor(candidate.recommendation)}`}>
                {getRecommendationIcon(candidate.recommendation)}
                {formatRecommendation(candidate.recommendation)}
              </span>
            </div>

            {/* Overall Assessment */}
            <div className="mb-4 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-900 mb-2">Overall Assessment</h4>
              <p className="text-gray-700">{candidate.overall_assessment}</p>
            </div>

            {/* Strengths & Weaknesses Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Strengths */}
              <div>
                <h4 className="font-semibold text-green-900 mb-3 flex items-center">
                  <CheckCircle className="mr-2 text-green-600" size={20} />
                  Strengths
                </h4>
                <ul className="space-y-2">
                  {candidate.strengths.map((strength, idx) => (
                    <li key={idx} className="flex items-start">
                      <span className="text-green-500 mr-2 mt-1">✓</span>
                      <span className="text-gray-700">{strength}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Weaknesses */}
              <div>
                <h4 className="font-semibold text-orange-900 mb-3 flex items-center">
                  <AlertCircle className="mr-2 text-orange-600" size={20} />
                  Areas of Concern
                </h4>
                <ul className="space-y-2">
                  {candidate.weaknesses.map((weakness, idx) => (
                    <li key={idx} className="flex items-start">
                      <span className="text-orange-500 mr-2 mt-1">•</span>
                      <span className="text-gray-700">{weakness}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="card mt-8 bg-primary-50 border-2 border-primary-200">
        <h3 className="text-xl font-bold text-primary-900 mb-3">Analysis Summary</h3>
        <p className="text-primary-800">
          Based on the job description, we've identified {total_candidates} matching candidate{total_candidates !== 1 ? 's' : ''}
          {' '}from the resume database. Each candidate has been analyzed and scored based on their qualifications,
          experience, and fit for the role.
        </p>
      </div>
    </div>
  );
};

export default Analysis;
