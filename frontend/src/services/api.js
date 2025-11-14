// src/services/api.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 600000, // Increased to 10 minutes for LLM processing
});

export const apiService = {
  checkHealth: async () => {
    const response = await api.get('/api/health');
    return response.data;
  },

  initializeSystem: async (force = false) => {
    const response = await api.post('/api/initialize', { force });
    return response.data;
  },

  searchCandidates: async (searchParams) => {
    const response = await api.post('/api/search', searchParams);
    return response.data;
  },

  getStatistics: async () => {
    const response = await api.get('/api/stats');
    return response.data;
  },

  uploadResume: async (file, roleCategory) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('role_category', roleCategory);

    const response = await api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getAvailableRoles: async () => {
    const response = await api.get('/api/roles');
    return response.data;
  },
};

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    if (error.response) {
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
      throw new Error(error.response.data.detail || JSON.stringify(error.response.data) || 'An error occurred');
    } else if (error.request) {
      console.error('Request error:', error.request);
      console.error('Error message:', error.message);
      throw new Error(`Network error: ${error.message}`);
    } else {
      console.error('Error:', error.message);
      throw error;
    }
  }
);

export default api;
