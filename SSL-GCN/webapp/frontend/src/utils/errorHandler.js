/**
 * Utility functions for handling API errors
 */

/**
 * Extract a user-friendly error message from an API error response
 * @param {Error} error - The error object from axios
 * @param {string} defaultMessage - Default message if no specific error found
 * @returns {string} User-friendly error message
 */
export const getErrorMessage = (error, defaultMessage = 'An error occurred') => {
  if (!error.response?.data) {
    return error.message || defaultMessage;
  }

  const data = error.response.data;

  // Handle FastAPI validation errors (422)
  if (Array.isArray(data.detail)) {
    // Get the first validation error
    const firstError = data.detail[0];
    
    // Format: "field: message"
    const field = firstError.loc?.slice(-1)[0] || 'input';
    const message = firstError.msg || firstError.message || 'Invalid value';
    
    return `${field}: ${message}`;
  }

  // Handle string detail messages
  if (typeof data.detail === 'string') {
    return data.detail;
  }

  // Handle error field
  if (data.error) {
    return typeof data.error === 'string' ? data.error : defaultMessage;
  }

  // Handle message field
  if (data.message) {
    return data.message;
  }

  return defaultMessage;
};

/**
 * Extract all validation error messages
 * @param {Error} error - The error object from axios
 * @returns {Array<string>} Array of error messages
 */
export const getValidationErrors = (error) => {
  if (!error.response?.data?.detail || !Array.isArray(error.response.data.detail)) {
    return [getErrorMessage(error)];
  }

  return error.response.data.detail.map(err => {
    const field = err.loc?.slice(-1)[0] || 'field';
    const message = err.msg || err.message || 'Invalid value';
    return `${field}: ${message}`;
  });
};

/**
 * Check if error is a validation error (422)
 * @param {Error} error - The error object from axios
 * @returns {boolean}
 */
export const isValidationError = (error) => {
  return error.response?.status === 422;
};

/**
 * Check if error is a network error
 * @param {Error} error - The error object from axios
 * @returns {boolean}
 */
export const isNetworkError = (error) => {
  return !error.response && error.request;
};
