import { useState, useEffect } from 'react';
import { analyzeThumb } from '../api';

export default function ThumbnailAdvice({ bestFile, niche }) {
  const [advice,  setAdvice]  = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!bestFile) return;
    let cancelled = false;
    setLoading(true);
    setAdvice(null);
    analyzeThumb(bestFile, niche)
      .then(data => {
        if (cancelled) return;
        // null advice means API key not set — hide the section silently
        if (data.advice) setAdvice(data.advice);
      })
      .catch(() => { /* hide silently on error */ })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [bestFile, niche]);

  if (!loading && !advice) return null;

  // Parse bullet points from Claude's response
  const bullets = advice
    ? advice.split('\n').map(l => l.trim()).filter(l => l.startsWith('•'))
    : [];

  return (
    <div className="advice-card fade-in">
      <div className="advice-header">
        <div className="advice-title-row">
          {/* Sparkle icon */}
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#6c63ff" strokeWidth="2">
            <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z" />
          </svg>
          <h3 className="advice-title">AI Design Feedback</h3>
          <span className="advice-niche">{niche}</span>
        </div>
        <p className="advice-subtitle">Specific improvements for your best thumbnail</p>
      </div>

      {loading ? (
        <div className="advice-loading">
          <span className="spinner-sm" />
          <span>Analyzing thumbnail…</span>
        </div>
      ) : (
        <ul className="advice-list">
          {bullets.length > 0
            ? bullets.map((b, i) => (
                <li key={i} className="advice-item">
                  <span className="advice-bullet">0{i + 1}</span>
                  <span className="advice-text">{b.replace(/^•\s*/, '')}</span>
                </li>
              ))
            : (
                /* Fallback: render raw text if Claude didn't use bullet format */
                <li className="advice-item">
                  <span className="advice-text">{advice}</span>
                </li>
              )
          }
        </ul>
      )}

      {!loading && advice && (
        <p className="advice-powered">Powered by Claude</p>
      )}
    </div>
  );
}
