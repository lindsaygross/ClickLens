export default function InsightsPanel({ results }) {
  if (!results || results.length < 2) return null;

  const best = results[0];
  const worst = results[results.length - 1];
  const bestConf = Math.round((best.confidence || 0) * 100);
  const worstConf = Math.round((worst.confidence || 0) * 100);

  const insights = generateInsights(best, worst, bestConf, worstConf);

  return (
    <div className="insights-panel">
      <div className="insights-header">
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="#ffc107" strokeWidth="1.5" style={{ marginRight: 8, verticalAlign: 'middle' }}>
          <circle cx="11" cy="11" r="9" />
          <path d="M11 7v4l3 2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <h3>Comparison Insights</h3>
      </div>
      <p className="insights-subtitle">
        Key differences between the top-ranked and lowest-ranked thumbnails
      </p>

      {/* Heatmap explanation — shown first so users know how to use the toggle */}
      <div className="heatmap-explainer">
        <div className="heatmap-explainer-title">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="#ffc107" strokeWidth="1.5" style={{ flexShrink: 0 }}>
            <rect x="1" y="1" width="14" height="14" rx="2" />
            <circle cx="8" cy="8" r="3" />
            <circle cx="8" cy="8" r="1" fill="#ffc107" />
          </svg>
          How to read the Grad-CAM heatmap
        </div>
        <p className="heatmap-explainer-desc">
          Click <strong>Show Heatmap</strong> on any thumbnail to see a colour overlay. The colours show where the AI focused when making its prediction:
        </p>
        <div className="heatmap-color-guide">
          <div className="heatmap-color-item">
            <span className="heatmap-swatch" style={{ background: '#d73027' }} />
            <span><strong>Red / warm</strong> — areas the model paid the most attention to</span>
          </div>
          <div className="heatmap-color-item">
            <span className="heatmap-swatch" style={{ background: '#fee090' }} />
            <span><strong>Yellow / orange</strong> — moderate attention</span>
          </div>
          <div className="heatmap-color-item">
            <span className="heatmap-swatch" style={{ background: '#4575b4' }} />
            <span><strong>Blue / cool</strong> — areas the model largely ignored</span>
          </div>
        </div>
        <p className="heatmap-explainer-tip">
          If red regions land on a face or bold text, the model is responding to strong engagement signals — which is a good sign for CTR.
        </p>
      </div>

      <div className="insights-grid">
        {insights.map((insight, i) => (
          <div key={i} className="insight-item">
            <div className="insight-icon" style={{ background: insight.iconBg }}>
              {insight.icon}
            </div>
            <div className="insight-content">
              <span className="insight-title">{insight.title}</span>
              <span className="insight-desc">{insight.description}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="insights-summary">
        <div className="summary-card summary-card--winner">
          <span className="summary-label">Best Performer</span>
          <span className="summary-class" style={{ color: getClassColor(best.predicted_class) }}>
            {best.predicted_class} CTR
          </span>
          <span className="summary-conf">{bestConf}% confidence</span>
        </div>
        <div className="summary-vs">vs</div>
        <div className="summary-card summary-card--loser">
          <span className="summary-label">Lowest Ranked</span>
          <span className="summary-class" style={{ color: getClassColor(worst.predicted_class) }}>
            {worst.predicted_class} CTR
          </span>
          <span className="summary-conf">{worstConf}% confidence</span>
        </div>
      </div>
    </div>
  );
}

function getClassColor(cls) {
  if (cls === 'High') return '#00e676';
  if (cls === 'Medium') return '#ffc107';
  return '#ff5252';
}

function generateInsights(best, worst, bestConf, worstConf) {
  const insights = [];

  // Class-level insight
  if (best.predicted_class !== worst.predicted_class) {
    insights.push({
      title: 'Higher CTR Class',
      description: `The winning thumbnail is predicted as "${best.predicted_class}" CTR vs "${worst.predicted_class}" CTR for the lowest-ranked.`,
      icon: (
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#00e676" strokeWidth="1.5">
          <path d="M9 14V4M5 8l4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ),
      iconBg: 'rgba(0,230,118,0.12)',
    });
  }

  // Confidence gap
  const confDiff = bestConf - worstConf;
  if (confDiff > 0) {
    insights.push({
      title: 'Stronger Confidence',
      description: `The model is ${confDiff} percentage points more confident about the top thumbnail.`,
      icon: (
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#00d2ff" strokeWidth="1.5">
          <rect x="2" y="10" width="3" height="6" rx="0.5" />
          <rect x="7.5" y="6" width="3" height="10" rx="0.5" />
          <rect x="13" y="2" width="3" height="14" rx="0.5" />
        </svg>
      ),
      iconBg: 'rgba(0,210,255,0.12)',
    });
  }

  // High CTR characteristics
  if (best.predicted_class === 'High') {
    insights.push({
      title: 'Engaging Visual Composition',
      description: 'High CTR thumbnails typically feature bold colors, faces, text overlays, and strong contrast.',
      icon: (
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#6c63ff" strokeWidth="1.5">
          <circle cx="9" cy="7" r="4" />
          <path d="M3 17c0-3.3 2.7-6 6-6s6 2.7 6 6" />
        </svg>
      ),
      iconBg: 'rgba(108,99,255,0.12)',
    });
  }

  // Low CTR characteristics
  if (worst.predicted_class === 'Low') {
    insights.push({
      title: 'Low Engagement Signals',
      description: 'Low CTR thumbnails often lack contrast, have cluttered composition, or miss emotional triggers.',
      icon: (
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#ff5252" strokeWidth="1.5">
          <path d="M9 4v6M9 13h.01" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M2 16L9 2l7 14H2z" />
        </svg>
      ),
      iconBg: 'rgba(255,82,82,0.12)',
    });
  }

  // Heatmap suggestion
  insights.push({
    title: 'Explore Heatmaps',
    description: 'Toggle Grad-CAM heatmaps on each thumbnail to see which regions influenced the prediction most.',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#ffc107" strokeWidth="1.5">
        <rect x="2" y="2" width="14" height="14" rx="2" />
        <circle cx="9" cy="9" r="3" />
        <circle cx="9" cy="9" r="1" fill="#ffc107" />
      </svg>
    ),
    iconBg: 'rgba(255,193,7,0.12)',
  });

  // General tip
  if (insights.length < 5) {
    insights.push({
      title: 'Optimization Tip',
      description: 'Try A/B testing thumbnails with clear faces, 3 or fewer words of text, and high color saturation.',
      icon: (
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#00d2ff" strokeWidth="1.5">
          <circle cx="9" cy="9" r="7" />
          <path d="M9 5v4l2.5 1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ),
      iconBg: 'rgba(0,210,255,0.12)',
    });
  }

  return insights;
}
