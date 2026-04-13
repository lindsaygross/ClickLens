import { useState } from 'react';
import { getGradcam } from '../api';

const RANK_COLORS = ['#FFD700', '#C0C0C0', '#CD7F32', '#6c63ff'];
const RANK_LABELS = ['1st', '2nd', '3rd', '4th'];

const CLASS_CONFIG = {
  High:   { color: '#00e676', bg: 'rgba(0,230,118,0.12)',  label: 'High CTR'   },
  Medium: { color: '#ffc107', bg: 'rgba(255,193,7,0.12)',  label: 'Medium CTR' },
  Low:    { color: '#ff5252', bg: 'rgba(255,82,82,0.12)',  label: 'Low CTR'    },
};

export default function Results({ results }) {
  const [heatmaps,       setHeatmaps]       = useState({});
  const [heatmapVisible, setHeatmapVisible] = useState({});
  const [heatmapLoading, setHeatmapLoading] = useState({});
  const [heatmapError,   setHeatmapError]   = useState({});

  async function toggleHeatmap(index) {
    if (heatmapVisible[index]) {
      setHeatmapVisible(p => ({ ...p, [index]: false }));
      return;
    }
    if (heatmaps[index]) {
      setHeatmapVisible(p => ({ ...p, [index]: true }));
      return;
    }
    setHeatmapLoading(p => ({ ...p, [index]: true }));
    setHeatmapError(p => ({ ...p, [index]: null }));
    try {
      const data = await getGradcam(results[index].file);
      setHeatmaps(p => ({ ...p, [index]: data.heatmap }));
      setHeatmapVisible(p => ({ ...p, [index]: true }));
    } catch {
      setHeatmapError(p => ({ ...p, [index]: true }));
    } finally {
      setHeatmapLoading(p => ({ ...p, [index]: false }));
    }
  }

  function confidenceGrad(v) {
    if (v >= 70) return 'linear-gradient(90deg,#00e676,#69f0ae)';
    if (v >= 40) return 'linear-gradient(90deg,#ffc107,#ffeb3b)';
    return 'linear-gradient(90deg,#ff5252,#ff8a80)';
  }

  return (
    <div className="results-container">
      <div className="results-header">
        <h2 className="results-title">Prediction Results</h2>
        <p className="results-subtitle">Ranked by predicted click-through rate</p>
      </div>

      <div className="results-grid">
        {results.map((r, i) => {
          const cls        = CLASS_CONFIG[r.predicted_class] || CLASS_CONFIG.Low;
          const confidence = Math.round((r.confidence || 0) * 100);
          const isWinner   = i === 0;
          const showing    = heatmapVisible[i];
          const loading    = heatmapLoading[i];
          const hasError   = heatmapError[i];

          return (
            <div key={i} className={`result-card ${isWinner ? 'result-card--winner' : ''} fade-in`}
              style={{ animationDelay: `${i * 0.06}s` }}>

              {/* Rank badge */}
              <div className="rank-badge" style={{
                background: RANK_COLORS[i] || RANK_COLORS[3],
                boxShadow: `0 0 12px ${RANK_COLORS[i] || RANK_COLORS[3]}55`,
              }}>#{i + 1}</div>

              {isWinner && <div className="winner-ribbon">Best Pick</div>}

              {/* Image + heatmap overlay + toggle icon */}
              <div className="result-image-wrapper">
                <img src={r.previewUrl} alt={`Thumbnail ${i + 1}`} className="result-image" />
                {showing && heatmaps[i] && (
                  <img src={`data:image/jpeg;base64,${heatmaps[i]}`}
                    alt="Grad-CAM heatmap overlay"
                    className="result-heatmap-overlay" />
                )}
                <button
                  className={`heatmap-icon-btn ${showing ? 'active' : ''}`}
                  onClick={() => toggleHeatmap(i)}
                  disabled={loading}
                  title={showing ? 'Hide heatmap' : 'Show Grad-CAM heatmap'}
                  aria-pressed={showing}
                >
                  {loading ? (
                    <span className="spinner-sm" />
                  ) : showing ? (
                    /* eye-off */
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M2 8s2.5-5 6-5 6 5 6 5-2.5 5-6 5-6-5-6-5z" />
                      <circle cx="8" cy="8" r="2" />
                      <line x1="2" y1="14" x2="14" y2="2" />
                    </svg>
                  ) : (
                    /* eye */
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M2 8s2.5-5 6-5 6 5 6 5-2.5 5-6 5-6-5-6-5z" />
                      <circle cx="8" cy="8" r="2" />
                    </svg>
                  )}
                </button>
              </div>

              {/* Info */}
              <div className="result-info">
                <div className="result-class-badge" style={{ color: cls.color, background: cls.bg }}>
                  <span className="result-class-dot" style={{ background: cls.color }} />
                  {cls.label}
                </div>

                <div className="confidence-section">
                  <div className="confidence-label">
                    <span>Confidence</span>
                    <span className="confidence-value">{confidence}%</span>
                  </div>
                  <div className="confidence-bar-track">
                    <div className="confidence-bar-fill"
                      style={{ width: `${confidence}%`, background: confidenceGrad(confidence) }} />
                  </div>
                </div>

                {r.scores && (
                  <div className="score-breakdown">
                    {['High', 'Medium', 'Low'].map(label => {
                      const pct   = Math.round((r.scores[label] || 0) * 100);
                      const color = CLASS_CONFIG[label]?.color || '#fff';
                      return (
                        <div key={label} className="score-row">
                          <span className="score-label" style={{ color }}>{label}</span>
                          <div className="score-bar-track">
                            <div className="score-bar-fill"
                              style={{ width: `${pct}%`, background: color }} />
                          </div>
                          <span className="score-pct">{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {hasError && (
                  <p className="heatmap-error">Heatmap unavailable</p>
                )}
              </div>

              <div className="rank-label" style={{ color: RANK_COLORS[i] || RANK_COLORS[3] }}>
                {RANK_LABELS[i] || `${i + 1}th`} Place
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
