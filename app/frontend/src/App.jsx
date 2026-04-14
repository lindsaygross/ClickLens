import { useState } from 'react';
import UploadZone from './components/UploadZone';
import Results from './components/Results';
import InsightsPanel from './components/InsightsPanel';
import Recommendations from './components/Recommendations';
import ThumbnailAdvice from './components/ThumbnailAdvice';

function App() {
  const [view, setView] = useState('upload');
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [isMock, setIsMock] = useState(false);
  const [niche, setNiche] = useState('Gaming');

  function handleAnalyze(selectedFiles, apiResults, mock = false, selectedNiche = 'Gaming') {
    setFiles(selectedFiles);
    setResults(apiResults);
    setIsMock(mock);
    setNiche(selectedNiche);
    setView('results');
  }
  function handleLoading() { setError(null); setView('loading'); }
  function handleError(msg) { setError(msg); setView('upload'); }
  function handleReset() {
    setView('upload'); setFiles([]); setResults([]); setError(null); setIsMock(false); setNiche('Gaming');
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo" onClick={handleReset} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && handleReset()}>
          <svg width="26" height="26" viewBox="0 0 28 28" fill="none">
            <rect x="2" y="6" width="24" height="16" rx="3" stroke="url(#lg)" strokeWidth="2" />
            <polygon points="11,11 11,19 19,15" fill="url(#lg)" />
            <circle cx="22" cy="8" r="4" fill="#00d2ff" opacity="0.9" />
            <defs>
              <linearGradient id="lg" x1="0" y1="0" x2="28" y2="28">
                <stop offset="0%" stopColor="#6c63ff" />
                <stop offset="100%" stopColor="#00d2ff" />
              </linearGradient>
            </defs>
          </svg>
          <span className="logo-text">ClickLens</span>
        </div>
        {view === 'results' && (
          <button className="btn-ghost" onClick={handleReset}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" style={{ marginRight: 6 }}>
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
                Which thumbnail gets<br />
                <span className="gradient-text">more clicks?</span>
              </h1>
              <p className="hero-subtitle">
                Upload 2–4 thumbnails. AI ranks them by predicted CTR in seconds.
              </p>
            </div>

            {error && (
              <div className="error-banner fade-in">
                <svg width="16" height="16" viewBox="0 0 18 18" fill="none" style={{ flexShrink: 0, marginTop: 1 }}>
                  <circle cx="9" cy="9" r="8" stroke="#ff5252" strokeWidth="1.5" />
                  <line x1="9" y1="5" x2="9" y2="10" stroke="#ff5252" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="9" cy="13" r="0.75" fill="#ff5252" />
                </svg>
                <span>{error}</span>
              </div>
            )}

            <UploadZone onAnalyze={handleAnalyze} onLoading={handleLoading} onError={handleError} />

            <div className="feature-pills">
              <div className="feature-pill">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#6c63ff" strokeWidth="2.5">
                  <path d="M12 20V10M18 20V4M6 20v-4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                CTR Prediction
              </div>
              <div className="feature-pill">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="2.5">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <path d="M21 15l-5-5L5 21" />
                </svg>
                Grad-CAM Heatmaps
              </div>
              <div className="feature-pill">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#00e676" strokeWidth="2.5">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" strokeLinecap="round" />
                  <polyline points="22 4 12 14.01 9 11.01" strokeLinecap="round" />
                </svg>
                Ranked Results
              </div>
              <div className="feature-pill">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f5a623" strokeWidth="2.5">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M8 12l3 3 5-5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Similar High-CTR
              </div>
            </div>
          </div>
        )}

        {view === 'loading' && (
          <div className="loading-state fade-in">
            <div className="dot-wave">
              <span /><span /><span />
            </div>
            <p className="loading-label">Analyzing thumbnails</p>
          </div>
        )}

        {view === 'results' && (
          <div className="results-state fade-in">
            {isMock && (
              <div className="demo-banner">
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
                  <circle cx="8" cy="8" r="7" stroke="#f5a623" strokeWidth="1.5" />
                  <line x1="8" y1="4.5" x2="8" y2="9" stroke="#f5a623" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="8" cy="11.5" r="0.75" fill="#f5a623" />
                </svg>
                Demo Mode — results are randomised placeholders
              </div>
            )}
            <Results results={results} files={files} />
            <InsightsPanel results={results} />
            <Recommendations bestFile={results[0]?.file} niche={niche} />
            <ThumbnailAdvice bestFile={results[0]?.file} niche={niche} />
            <div className="results-footer">
              <button className="btn-primary" onClick={handleReset}>
                + Compare More
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
