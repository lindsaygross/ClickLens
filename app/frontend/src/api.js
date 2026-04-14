import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function parseError(err) {
  if (!err.response) {
    return 'Cannot reach the backend. Please try again in a moment — the server may be waking up.';
  }
  if (err.response.status === 503) {
    return 'The server is starting up. Please wait a few seconds and try again.';
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

export async function getRecommendations(file, niche = '') {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const { data } = await axios.post(
      `${API_BASE}/recommend?niche=${encodeURIComponent(niche)}`,
      formData,
    );
    return data;
  } catch (err) {
    throw new Error(parseError(err));
  }
}

export async function analyzeThumb(file, niche = 'Gaming') {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const { data } = await axios.post(
      `${API_BASE}/analyze?niche=${encodeURIComponent(niche)}`,
      formData,
    );
    return data;
  } catch (err) {
    throw new Error(parseError(err));
  }
}
