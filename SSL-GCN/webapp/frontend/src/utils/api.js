import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available (for future use)
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error
      const data = error.response.data;
      
      // Format FastAPI validation errors for better logging
      if (error.response.status === 422 && Array.isArray(data.detail)) {
        const validationErrors = data.detail.map(err => 
          `${err.loc?.join('.') || 'field'}: ${err.msg || err.message}`
        ).join(', ');
        console.error('Validation Error:', validationErrors);
      } else if (typeof data.detail === 'string') {
        console.error('API Error:', data.detail);
      } else {
        console.error('API Error:', data);
      }
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.message);
    } else {
      // Something else happened
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// API Methods
export const toxicityAPI = {
  // Health check
  healthCheck: () => api.get('/api/health'),

  // Get available endpoints
  getEndpoints: () => api.get('/api/endpoints'),

  // Get available models
  getModels: () => api.get('/api/models'),

  // Get endpoint preset
  getEndpointPreset: (presetName) => api.get(`/api/endpoint-presets/${presetName}`),

  // Validate single SMILES
  validateSMILES: (smiles, compoundId = null) =>
    api.post('/api/validate', { smiles, compound_id: compoundId }),

  // Validate batch SMILES
  validateBatchSMILES: (smilesList, compoundIds = null) =>
    api.post('/api/validate-batch', {
      smiles_list: smilesList,
      compound_ids: compoundIds,
    }),

  // Predict toxicity for single compound
  predictToxicity: (smiles, endpoints, compareBaseline = false, userId = null) =>
    api.post('/api/predict', {
      smiles,
      endpoints,
      compare_baseline: compareBaseline,
      user_id: userId,
    }),

  // Batch predict toxicity
  batchPredictToxicity: (smilesList, endpoints, compareBaseline = false, userId = null) =>
    api.post('/api/batch-predict', {
      smiles_list: smilesList,
      endpoints,
      compare_baseline: compareBaseline,
      user_id: userId,
    }),

  // Future endpoints (placeholders)
  compareModels: () => api.post('/api/compare-models'),
  generateVisualizations: () => api.post('/api/visualize'),
  explainGCN: () => api.post('/api/explain-gcn'),
  getResearchMetrics: () => api.get('/api/research-metrics'),
};

export default api;
