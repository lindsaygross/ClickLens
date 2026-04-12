import { useState } from 'react';
import UploadZone from './components/UploadZone';
import Results from './components/Results';
import InsightsPanel from './components/InsightsPanel';

function App() {
  const [view, setView] = useState('upload');
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [isMock, setIsMock] = useState(false);

  function handleAnalyze(selectedFiles, apiResults, mock = false) {
    setFiles(selectedFiles);
    setResults(apiResults);
    setIsMock(mock);
    setView('results');
  }

  function handleLoading() {
    setError(null);
    setView('loading');
  }

  function handleError(msg) {
    setError(msg);
    setView('upload');
  }

  function handleReset() {
    setView('upload');
    setFiles([]);
    setResults([]);
    setError(null);
    setIsMock(false);
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo" onClick={handleReset} role="button" tabIndex={0}>
          <span className="logo-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect x="2" y="6" width="24" height="16" rx="3" stroke="url(#logoGrad)" strokeWidth="2" />
              <polygon points="11,11 11,19 19,15" fill="url(#logoGrad)" />
              <circle cx="22" cy="8" r="4" fill="#00d2ff" opacity="0.9" />
              <defs>
                <linearGradient id="logoGrad" x1="0" y1="0" x2="28" y2="28">
                  <stop offset="0%" stopColor="#6c63ff" />
                  <stop offset="100%" stopColor="#00d2ff" />
                </linearGradient>
              </defs>
            </svg>
          </span>
          <span className="logo-text">ClickLens</span>
        </div>
        {view === 'results' && (
          <button className="btn-ghost" onClick={handleReset}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" style={{ marginRight: 6 }}>
              <path d="M2.5 1a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 0-1H3.83A6 6 0 1 1 2 8a.5.5 0 0 0-1 0 7 7 0 1 0 1.74-4.616V1.5a.5.5 0 0 0-.5-.5z" />
            </svg>
            New Comparison
          </button>
        )}
      </header>

      <main className="app-main">
        {view === 'upload' && (
          <div className="landing fade-in">
            <div className="hero">
              <h1 className="hero-title">
                Which thumbnail will get{' '}
                <span className="gradient-text">more clicks</span>?
              </h1>
              <p className="hero-subtitle">
                AI-powered thumbnail comparison. Upload 2 -- 4 YouTube thumbnails and instantly
                see which one is predicted to drive the most engagement.
              </p>
            </div>
            {error && (
              <div className="error-banner fade-in">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style={{ flexShrink: 0 }}>
                  <circle cx="9" cy="9" r="8" stroke="#ff5252" strokeWidth="1.5" />
                  <line x1="9" y1="5" x2="9" y2="10" stroke="#ff5252" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="9" cy="13" r="0.75" fill="#ff5252" />
                </svg>
                <span>{error}</span>
              </div>
            )}
            <UploadZone
              onAnalyze={handleAnalyze}
              onLoading={handleLoading}
              onError={handleError}
            />
            <div className="features">
              <div className="feature-card">
                <div className="feature-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6c63ff" strokeWidth="2">
                    <path d="M12 20V10M18 20V4M6 20v-4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <h3>CTR Prediction</h3>
                <p>Deep learning model trained on real YouTube data predicts click-through rate class.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <path d="M21 15l-5-5L5 21" />
                  </svg>
                </div>
                <h3>Grad-CAM Heatmaps</h3>
                <p>Visualize exactly which regions of each thumbnail the model focuses on.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00e676" strokeWidth="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" strokeLinecap="round" strokeLinejoin="round" />
                    <polyline points="22 4 12 14.01 9 11.01" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <h3>Ranked Results</h3>
                <p>Side-by-side comparison ranked from best to worst with confidence scores.</p>
              </div>
            </div>
          </div>
        )}

        {view === 'loading' && (
          <div className="loading-state fade-in">
            <div className="loader">
              <div className="loader-ring"></div>
              <div className="loader-ring"></div>
              <div className="loader-ring"></div>
            </div>
            <h2 className="loading-title">Analyzing thumbnails...</h2>
            <p className="loading-sub">Our AI is evaluating visual features, composition, and engagement signals.</p>
          </div>
        )}

        {view === 'results' && (
          <div className="results-state fade-in">
            {isMock && (
              <div className="demo-banner">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
                  <circle cx="8" cy="8" r="7" stroke="#f5a623" strokeWidth="1.5" />
                  <line x1="8" y1="4.5" x2="8" y2="9" stroke="#f5a623" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="8" cy="11.5" r="0.75" fill="#f5a623" />
                </svg>
                <span>Demo Mode — AI model not loaded yet. Results are randomised placeholders.</span>
              </div>
            )}
            <Results results={results} files={files} />
            <InsightsPanel results={results} />
            <div className="results-footer">
              <button className="btn-primary" onClick={handleReset}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" style={{ marginRight: 6 }}>
                  <path d="M8 1a.5.5 0 0 1 .5.5V7h5.5a.5.5 0 0 1 0 1H8.5v5.5a.5.5 0 0 1-1 0V8H2a.5.5 0 0 1 0-1h5.5V1.5A.5.5 0 0 1 8 1z" />
                </svg>
                Compare More Thumbnails
              </button>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <span>ClickLens</span>
        <span className="footer-sep">/</span>
        <span>YouTube Thumbnail CTR Predictor</span>
      </footer>
    </div>
  );
}

export default App;
