import { useState } from 'react';
import { getGradcam } from '../api';

const RANK_COLORS = ['#FFD700', '#C0C0C0', '#CD7F32', '#6c63ff'];
const RANK_LABELS = ['1st', '2nd', '3rd', '4th'];

const CLASS_CONFIG = {
  High: { color: '#00e676', bg: 'rgba(0,230,118,0.12)', label: 'High CTR' },
  Medium: { color: '#ffc107', bg: 'rgba(255,193,7,0.12)', label: 'Medium CTR' },
  Low: { color: '#ff5252', bg: 'rgba(255,82,82,0.12)', label: 'Low CTR' },
};

export default function Results({ results, files }) {
  const [heatmaps, setHeatmaps] = useState({});
  const [heatmapVisible, setHeatmapVisible] = useState({});
  const [heatmapLoading, setHeatmapLoading] = useState({});
  const [heatmapError, setHeatmapError] = useState({});
  const [heatmapMock, setHeatmapMock] = useState({});

  async function toggleHeatmap(index) {
    const isVisible = heatmapVisible[index];

    if (isVisible) {
      setHeatmapVisible(prev => ({ ...prev, [index]: false }));
      return;
    }

    // If we already fetched this heatmap, just show it
    if (heatmaps[index]) {
      setHeatmapVisible(prev => ({ ...prev, [index]: true }));
      return;
    }

    // Fetch the heatmap
    setHeatmapLoading(prev => ({ ...prev, [index]: true }));
    setHeatmapError(prev => ({ ...prev, [index]: null }));
    try {
      const result = results[index];
      const file = result.file;
      const data = await getGradcam(file);
      setHeatmaps(prev => ({ ...prev, [index]: data.heatmap }));
      setHeatmapMock(prev => ({ ...prev, [index]: data.mock || false }));
      setHeatmapVisible(prev => ({ ...prev, [index]: true }));
    } catch (err) {
      setHeatmapError(prev => ({ ...prev, [index]: 'Could not load heatmap.' }));
    } finally {
      setHeatmapLoading(prev => ({ ...prev, [index]: false }));
    }
  }

  function getConfidenceGradient(confidence) {
    if (confidence >= 70) return 'linear-gradient(90deg, #00e676, #69f0ae)';
    if (confidence >= 40) return 'linear-gradient(90deg, #ffc107, #ffeb3b)';
    return 'linear-gradient(90deg, #ff5252, #ff8a80)';
  }

  return (
    <div className="results-container">
      <div className="results-header">
        <h2 className="results-title">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="2" style={{ marginRight: 8, verticalAlign: 'middle' }}>
            <path d="M12 20V10M18 20V4M6 20v-4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Prediction Results
        </h2>
        <p className="results-subtitle">Thumbnails ranked by predicted click-through rate</p>
      </div>

      <div className="results-grid">
        {results.map((r, i) => {
          const cls = CLASS_CONFIG[r.predicted_class] || CLASS_CONFIG.Low;
          const confidence = Math.round((r.confidence || 0) * 100);
          const isWinner = i === 0;
          const showingHeatmap = heatmapVisible[i];
          const loadingHeatmap = heatmapLoading[i];
          const heatmapErr = heatmapError[i];
          const isMockHeatmap = heatmapMock[i];

          return (
            <div
              key={i}
              className={`result-card ${isWinner ? 'result-card--winner' : ''}`}
            >
              {/* Rank badge */}
              <div
                className="rank-badge"
                style={{
                  background: RANK_COLORS[i] || RANK_COLORS[3],
                  boxShadow: `0 0 16px ${RANK_COLORS[i] || RANK_COLORS[3]}44`,
                }}
              >
                #{i + 1}
              </div>

              {isWinner && (
                <div className="winner-ribbon">Best Pick</div>
              )}

              {/* Thumbnail image area */}
              <div className="result-image-wrapper">
                <img
                  src={r.previewUrl}
                  alt={`Thumbnail ${i + 1}`}
                  className="result-image"
                />
                {showingHeatmap && heatmaps[i] && (
                  <img
                    src={`data:image/jpeg;base64,${heatmaps[i]}`}
                    alt="Grad-CAM heatmap"
                    className="result-heatmap-overlay"
                  />
                )}
              </div>

              {/* Info section */}
              <div className="result-info">
                <div className="result-class-badge" style={{ color: cls.color, background: cls.bg }}>
                  <span className="result-class-dot" style={{ background: cls.color }}></span>
                  {cls.label}
                </div>

                <div className="confidence-section">
                  <div className="confidence-label">
                    <span>Confidence</span>
                    <span className="confidence-value">{confidence}%</span>
                  </div>
                  <div className="confidence-bar-track">
                    <div
                      className="confidence-bar-fill"
                      style={{
                        width: `${confidence}%`,
                        background: getConfidenceGradient(confidence),
                      }}
                    />
                  </div>
                </div>

                {/* Score breakdown: Low / Medium / High */}
                {r.scores && (
                  <div className="score-breakdown">
                    {['High', 'Medium', 'Low'].map(label => {
                      const pct = Math.round((r.scores[label] || 0) * 100);
                      const color = CLASS_CONFIG[label]?.color || '#fff';
                      return (
                        <div key={label} className="score-row">
                          <span className="score-label" style={{ color }}>{label}</span>
                          <div className="score-bar-track">
                            <div
                              className="score-bar-fill"
                              style={{ width: `${pct}%`, background: color }}
                            />
                          </div>
                          <span className="score-pct">{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                <button
                  className={`btn-heatmap ${showingHeatmap ? 'btn-heatmap--active' : ''}`}
                  onClick={() => toggleHeatmap(i)}
                  disabled={loadingHeatmap}
                >
                  {loadingHeatmap ? (
                    <>
                      <span className="spinner-sm"></span>
                      Loading...
                    </>
                  ) : showingHeatmap ? (
                    <>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M2 8s2.5-5 6-5 6 5 6 5-2.5 5-6 5-6-5-6-5z" />
                        <circle cx="8" cy="8" r="2" />
                        <line x1="2" y1="14" x2="14" y2="2" />
                      </svg>
                      Hide Heatmap
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M2 8s2.5-5 6-5 6 5 6 5-2.5 5-6 5-6-5-6-5z" />
                        <circle cx="8" cy="8" r="2" />
                      </svg>
                      Show Heatmap
                    </>
                  )}
                </button>
              </div>

              {/* Heatmap legend */}
              {showingHeatmap && heatmaps[i] && (
                <div className="heatmap-legend">
                  <div className="heatmap-legend-bar" />
                  <div className="heatmap-legend-labels">
                    <span>Low attention</span>
                    <span>High attention</span>
                  </div>
                  {isMockHeatmap && (
                    <p className="heatmap-demo-note">Demo heatmap — random overlay. Real heatmap available after model training.</p>
                  )}
                </div>
              )}

              {/* Heatmap error */}
              {heatmapErr && (
                <p className="heatmap-error">{heatmapErr}</p>
              )}

              {/* Rank label */}
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
