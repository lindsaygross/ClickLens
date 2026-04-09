import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export async function predictThumbnails(files) {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  const { data } = await axios.post(`${API_BASE}/predict`, formData);
  return data.results;
}

export async function getGradcam(file) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await axios.post(`${API_BASE}/gradcam`, formData);
  return data;
}
