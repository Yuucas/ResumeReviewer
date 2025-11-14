import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { ROLE_CATEGORIES } from '../../utils/constants';

const SearchForm = ({ onSearch, loading }) => {
  const [formData, setFormData] = useState({
    job_description: '',
    top_k: 3, // Reduced from 5 to 3 for faster processing
    role_category: '',
    min_experience: 0,
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(formData);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'top_k' || name === 'min_experience' ? Number(value) : value,
    }));
  };

  return (
    <form onSubmit={handleSubmit} className="card">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Search Candidates</h2>

      {/* Job Description */}
      <div className="mb-6">
        <label className="label">
          Job Description <span className="text-red-500">*</span>
        </label>
        <textarea
          name="job_description"
          value={formData.job_description}
          onChange={handleChange}
          required
          rows={6}
          className="input resize-none"
          placeholder="Enter job description, requirements, and qualifications...&#10;&#10;Example:&#10;Senior Data Scientist with 5+ years experience in Python, ML, and production deployments. Strong background in TensorFlow, PyTorch, and cloud platforms."
        />
        <p className="text-sm text-gray-500 mt-2">
          Describe the role, requirements, and ideal candidate profile
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-4 mb-6">
        {/* Top K */}
        <div>
          <label className="label">Number of Candidates</label>
          <input
            type="number"
            name="top_k"
            value={formData.top_k}
            onChange={handleChange}
            min="1"
            max="20"
            className="input"
          />
          <p className="text-sm text-gray-500 mt-1">1-20 candidates</p>
        </div>

        {/* Role Category */}
        <div>
          <label className="label">Role Category</label>
          <select
            name="role_category"
            value={formData.role_category}
            onChange={handleChange}
            className="input"
          >
            {ROLE_CATEGORIES.map((role) => (
              <option key={role.value} value={role.value}>
                {role.label}
              </option>
            ))}
          </select>
        </div>

        {/* Min Experience */}
        <div>
          <label className="label">Min. Experience (years)</label>
          <input
            type="number"
            name="min_experience"
            value={formData.min_experience}
            onChange={handleChange}
            min="0"
            max="50"
            step="0.5"
            className="input"
          />
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={loading || !formData.job_description.trim()}
        className="btn-primary w-full text-lg py-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
      >
        {loading ? (
          <>
            <Loader2 className="animate-spin mr-2" size={20} />
            Searching...
          </>
        ) : (
          <>
            <Search className="mr-2" size={20} />
            Find Best Candidates
          </>
        )}
      </button>
    </form>
  );
};

export default SearchForm;
