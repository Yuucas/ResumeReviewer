import React, { useState } from 'react';
import { Upload, CheckCircle, AlertCircle, FileText, Loader2 } from 'lucide-react';
import { apiService } from '../../services/api';

const UploadResume = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [roleCategory, setRoleCategory] = useState('data_scientist');
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // 'success', 'error', null
  const [message, setMessage] = useState('');

  const roles = [
    { value: 'data_scientist', label: 'Data Scientist' },
    { value: 'fullstack_engineer', label: 'Full Stack Engineer' },
    { value: 'it', label: 'IT Professional' }
  ];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
        setUploadStatus(null);
        setMessage('');
      } else {
        setUploadStatus('error');
        setMessage('Please select a PDF file');
        setFile(null);
      }
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file) {
      setUploadStatus('error');
      setMessage('Please select a file');
      return;
    }

    setUploading(true);
    setUploadStatus(null);
    setMessage('');

    try {
      const result = await apiService.uploadResume(file, roleCategory);

      if (result.success) {
        setUploadStatus('success');
        setMessage(`Resume "${result.filename}" uploaded successfully to ${result.role_category}`);
        setFile(null);

        // Reset file input
        const fileInput = document.getElementById('resume-file');
        if (fileInput) fileInput.value = '';

        // Notify parent component
        if (onUploadSuccess) {
          onUploadSuccess(result);
        }
      }
    } catch (err) {
      setUploadStatus('error');
      setMessage(err.message || 'Failed to upload resume');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="card">
      <div className="flex items-center mb-6">
        <Upload className="text-primary-600 mr-3" size={28} />
        <h2 className="text-2xl font-bold text-gray-900">Upload New Resume</h2>
      </div>

      <p className="text-gray-600 mb-6">
        Add a new resume to the database. The resume will be indexed and available for future searches.
      </p>

      <form onSubmit={handleUpload} className="space-y-6">
        {/* Role Category Selection */}
        <div>
          <label className="label">
            Role Category <span className="text-red-500">*</span>
          </label>
          <select
            value={roleCategory}
            onChange={(e) => setRoleCategory(e.target.value)}
            className="input"
            disabled={uploading}
          >
            {roles.map((role) => (
              <option key={role.value} value={role.value}>
                {role.label}
              </option>
            ))}
          </select>
          <p className="text-sm text-gray-500 mt-1">
            Select the appropriate role category for this resume
          </p>
        </div>

        {/* File Upload */}
        <div>
          <label className="label">
            Resume PDF <span className="text-red-500">*</span>
          </label>
          <div className="mt-1">
            <input
              id="resume-file"
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              disabled={uploading}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-primary-50 file:text-primary-700
                hover:file:bg-primary-100
                cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>
          {file && (
            <div className="mt-2 flex items-center text-sm text-gray-700">
              <FileText size={16} className="mr-2 text-primary-600" />
              <span className="font-medium">{file.name}</span>
              <span className="ml-2 text-gray-500">
                ({(file.size / 1024).toFixed(1)} KB)
              </span>
            </div>
          )}
          <p className="text-sm text-gray-500 mt-1">
            Only PDF files are accepted (max 10MB recommended)
          </p>
        </div>

        {/* Status Messages */}
        {uploadStatus === 'success' && (
          <div className="flex items-start p-4 bg-green-50 border-2 border-green-200 rounded-lg">
            <CheckCircle className="text-green-600 mt-0.5 flex-shrink-0" size={20} />
            <div className="ml-3">
              <h4 className="font-semibold text-green-900">Upload Successful!</h4>
              <p className="text-green-800 text-sm mt-1">{message}</p>
              <p className="text-green-700 text-sm mt-2">
                <strong>Next step:</strong> Click "Re-index Database" below to make this resume searchable.
              </p>
            </div>
          </div>
        )}

        {uploadStatus === 'error' && (
          <div className="flex items-start p-4 bg-red-50 border-2 border-red-200 rounded-lg">
            <AlertCircle className="text-red-600 mt-0.5 flex-shrink-0" size={20} />
            <div className="ml-3">
              <h4 className="font-semibold text-red-900">Upload Failed</h4>
              <p className="text-red-800 text-sm mt-1">{message}</p>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex gap-3">
          <button
            type="submit"
            disabled={uploading || !file}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? (
              <>
                <Loader2 className="animate-spin mr-2 inline" size={18} />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="mr-2 inline" size={18} />
                Upload Resume
              </>
            )}
          </button>

          {file && !uploading && (
            <button
              type="button"
              onClick={() => {
                setFile(null);
                setUploadStatus(null);
                setMessage('');
                const fileInput = document.getElementById('resume-file');
                if (fileInput) fileInput.value = '';
              }}
              className="btn-secondary"
            >
              Clear
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default UploadResume;
