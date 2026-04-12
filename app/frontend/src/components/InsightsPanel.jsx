export default function InsightsPanel({ results }) {
  if (!results || results.length < 2) return null;

  const best  = results[0];
  const worst = results[results.length - 1];
  const bestPct  = Math.round((best.confidence  || 0) * 100);
  const worstPct = Math.round((worst.confidence || 0) * 100);
  const gap = bestPct - worstPct;

  return (
    <div className="stat-strip fade-in">
      <div className="stat-item">
        <span className="stat-label">Top pick</span>
        <span className="stat-value" style={{ color: classColor(best.predicted_class) }}>
          {best.predicted_class} CTR
        </span>
      </div>
      <div className="stat-sep" />
      <div className="stat-item">
        <span className="stat-label">Lowest ranked</span>
        <span className="stat-value" style={{ color: classColor(worst.predicted_class) }}>
          {worst.predicted_class} CTR
        </span>
      </div>
      <div className="stat-sep" />
      <div className="stat-item">
        <span className="stat-label">Confidence gap</span>
        <span className="stat-value">{gap >= 0 ? `+${gap}` : gap}pp</span>
      </div>
    </div>
  );
}

function classColor(cls) {
  if (cls === 'High')   return '#00e676';
  if (cls === 'Medium') return '#ffc107';
  return '#ff5252';
}
