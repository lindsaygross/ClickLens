import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function parseError(err) {
  if (!err.response) {
    // No response at all = backend not reachable
    return 'Cannot reach the backend. Make sure it is running: cd app/backend && uvicorn main:app --reload --port 8000';
  }
  if (err.response.status === 503) {
    return 'Backend is running but the AI model is not loaded yet. Using demo mode is unavailable — check server logs.';
  }
  return err.response?.data?.detail || err.message || 'Something went wrong.';
}

export async function predictThumbnails(files) {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  try {
    const { data } = await axios.post(`${API_BASE}/predict`, formData);
    return { results: data.results, mock: data.mock || false };
  } catch (err) {
    throw new Error(parseError(err));
  }
}

export async function getGradcam(file) {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const { data } = await axios.post(`${API_BASE}/gradcam`, formData);
    return data;
  } catch (err) {
    throw new Error(parseError(err));
  }
}
