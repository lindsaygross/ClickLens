import { useState, useRef, useCallback } from 'react';
import { predictThumbnails } from '../api';

const MAX_FILES = 4;
const MIN_FILES = 2;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];

export default function UploadZone({ onAnalyze, onLoading, onError }) {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [validationMsg, setValidationMsg] = useState('');
  const inputRef = useRef(null);

  const addFiles = useCallback((incoming) => {
    const valid = incoming.filter(f => ACCEPTED_TYPES.includes(f.type));
    if (valid.length < incoming.length) {
      setValidationMsg('Some files were skipped -- only images (JPEG, PNG, WebP, GIF) are accepted.');
    } else {
      setValidationMsg('');
    }

    setFiles(prev => {
      const combined = [...prev, ...valid].slice(0, MAX_FILES);
      if (prev.length + valid.length > MAX_FILES) {
        setValidationMsg(`Maximum ${MAX_FILES} thumbnails allowed. Only the first ${MAX_FILES} were kept.`);
      }
      // Build previews
      const newPreviews = combined.map(f => ({
        name: f.name,
        url: URL.createObjectURL(f),
      }));
      setPreviews(oldPreviews => {
        oldPreviews.forEach(p => URL.revokeObjectURL(p.url));
        return newPreviews;
      });
      return combined;
    });
  }, []);

  function removeFile(index) {
    setFiles(prev => {
      const updated = prev.filter((_, i) => i !== index);
      const newPreviews = updated.map(f => ({
        name: f.name,
        url: URL.createObjectURL(f),
      }));
      setPreviews(oldPreviews => {
        oldPreviews.forEach(p => URL.revokeObjectURL(p.url));
        return newPreviews;
      });
      setValidationMsg('');
      return updated;
    });
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const dropped = Array.from(e.dataTransfer.files);
    addFiles(dropped);
  }

  function handleDragOver(e) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave(e) {
    e.preventDefault();
    setDragOver(false);
  }

  function handleInputChange(e) {
    const selected = Array.from(e.target.files);
    addFiles(selected);
    e.target.value = '';
  }

  async function handleSubmit() {
    if (files.length < MIN_FILES) {
      setValidationMsg(`Please upload at least ${MIN_FILES} thumbnails to compare.`);
      return;
    }
    onLoading();
    try {
      const { results, mock } = await predictThumbnails(files);
      // Attach original file references by matching filename (results may be reordered by backend)
      const enriched = results.map(r => {
        const matchingFile = files.find(f => f.name === r.filename) ?? files[0];
        return {
          ...r,
          file: matchingFile,
          previewUrl: URL.createObjectURL(matchingFile),
        };
      });
      // Sort by confidence descending (High > Medium > Low)
      const classOrder = { High: 3, Medium: 2, Low: 1 };
      enriched.sort((a, b) => {
        const classCompare = (classOrder[b.predicted_class] || 0) - (classOrder[a.predicted_class] || 0);
        if (classCompare !== 0) return classCompare;
        return (b.confidence || 0) - (a.confidence || 0);
      });
      onAnalyze(files, enriched, mock);
    } catch (err) {
      onError(err.message);
    }
  }

  const canSubmit = files.length >= MIN_FILES && files.length <= MAX_FILES;

  return (
    <div className="upload-zone-wrapper">
      <div
        className={`upload-zone ${dragOver ? 'upload-zone--active' : ''} ${files.length > 0 ? 'upload-zone--has-files' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => files.length === 0 && inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleInputChange}
          className="upload-input"
        />

        {files.length === 0 ? (
          <div className="upload-placeholder">
            <div className="upload-icon">
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <rect x="6" y="12" width="36" height="24" rx="4" stroke="rgba(108,99,255,0.5)" strokeWidth="2" strokeDasharray="4 3" />
                <path d="M24 20v8M20 24h8" stroke="#6c63ff" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </div>
            <p className="upload-text">
              Drag & drop thumbnails here
            </p>
            <p className="upload-subtext">
              or <button type="button" className="upload-browse" onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}>browse files</button>
            </p>
            <p className="upload-hint">Upload 2 -- 4 images (JPEG, PNG, WebP)</p>
          </div>
        ) : (
          <div className="upload-previews">
            {previews.map((p, i) => (
              <div key={p.name + i} className="preview-card">
                <img src={p.url} alt={p.name} className="preview-img" />
                <button
                  className="preview-remove"
                  onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                  aria-label={`Remove ${p.name}`}
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                    <path d="M3.5 3.5l7 7M10.5 3.5l-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none" />
                  </svg>
                </button>
                <span className="preview-label">Thumbnail {i + 1}</span>
              </div>
            ))}
            {files.length < MAX_FILES && (
              <button
                className="preview-add"
                onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 5v14M5 12h14" strokeLinecap="round" />
                </svg>
                <span>Add</span>
              </button>
            )}
          </div>
        )}
      </div>

      {validationMsg && (
        <p className="validation-msg">{validationMsg}</p>
      )}

      <button
        className={`btn-primary btn-analyze ${canSubmit ? '' : 'btn-disabled'}`}
        onClick={handleSubmit}
        disabled={!canSubmit}
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" style={{ marginRight: 8 }}>
          <circle cx="8.5" cy="8.5" r="5.5" stroke="currentColor" strokeWidth="1.5" />
          <path d="M13 13l4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        Analyze Thumbnails
      </button>
    </div>
  );
}
