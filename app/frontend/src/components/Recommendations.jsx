import { useState, useEffect } from 'react';
import { getRecommendations } from '../api';

const NICHE_COLORS = {
  Gaming:  { color: '#6c63ff', bg: 'rgba(108,99,255,0.12)' },
  Travel:  { color: '#00d2ff', bg: 'rgba(0,210,255,0.12)'  },
  Fitness: { color: '#00e676', bg: 'rgba(0,230,118,0.12)'  },
  Other:   { color: '#f5a623', bg: 'rgba(245,166,35,0.12)' },
};

const CTR_COLOR = { High: '#00e676', Medium: '#ffc107', Low: '#ff5252' };

export default function Recommendations({ bestFile, niche }) {
  const [items, setItems]     = useState([]);
  const [loading, setLoading] = useState(false);
  const [isMock, setIsMock]   = useState(false);

  useEffect(() => {
    if (!bestFile) return;
    let cancelled = false;
    setLoading(true);
    setItems([]);
    getRecommendations(bestFile, niche)
      .then(data => {
        if (cancelled) return;
        setItems(data.recommendations || []);
        setIsMock(data.mock || false);
      })
      .catch(() => { /* silently hide section on error */ })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [bestFile, niche]);

  if (!loading && items.length === 0) return null;

  const nicheStyle = NICHE_COLORS[niche] || NICHE_COLORS.Other;

  return (
    <div className="reco-section fade-in">
      <div className="reco-header">
        <h3 className="reco-title">
          Similar High-CTR thumbnails
          <span className="reco-niche-badge" style={{ color: nicheStyle.color, background: nicheStyle.bg }}>
            {niche}
          </span>
        </h3>
        <p className="reco-subtitle">
          What worked in your niche — ranked by visual similarity to your best pick
        </p>
      </div>

      {loading ? (
        <div className="reco-loading">
          <span className="spinner-sm" />
          <span>Finding similar thumbnails…</span>
        </div>
      ) : (
        <div className="reco-grid">
          {items.map((item, i) => {
            const ctrColor = CTR_COLOR[item.CTR_label] || CTR_COLOR.Low;
            const simPct   = Math.round(item.similarity * 100);
            const nStyle   = NICHE_COLORS[item.niche] || NICHE_COLORS.Other;
            return (
              <div key={i} className="reco-card fade-in" style={{ animationDelay: `${i * 0.05}s` }}>
                <div className="reco-rank">#{i + 1}</div>

                <div className="reco-meta">
                  <span className="reco-ctr-badge" style={{ color: ctrColor, background: `${ctrColor}1a` }}>
                    <span className="reco-dot" style={{ background: ctrColor }} />
                    {item.CTR_label} CTR
                  </span>
                  <span className="reco-niche-tag" style={{ color: nStyle.color, background: nStyle.bg }}>
                    {item.niche}
                  </span>
                </div>

                <div className="reco-similarity">
                  <div className="reco-sim-label">
                    <span>Similarity</span>
                    <span className="reco-sim-value">{simPct}%</span>
                  </div>
                  <div className="reco-sim-track">
                    <div className="reco-sim-fill" style={{ width: `${simPct}%` }} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {isMock && (
        <p className="reco-mock-note">Demo data — recommendations use real index after Lindsay's PR merges</p>
      )}
    </div>
  );
}
