import UrgencyBadge from "./UrgencyBadge";

const colors = {
  low:       { border:"#166534", bg:"#0a1f12" },
  moderate:  { border:"#92400e", bg:"#1c1500" },
  high:      { border:"#991b1b", bg:"#1c0808" },
  emergency: { border:"#9a3412", bg:"#1c0e05" },
};

export default function TriageResult({ outcome }) {
  if (!outcome) return null;
  const c = colors[outcome.urgency] || colors.moderate;

  return (
    <div style={{
      margin:"0 16px 12px", borderRadius:12, overflow:"hidden",
      border:`1px solid ${c.border}`, background:c.bg, flexShrink:0,
    }}>

      {/* Header */}
      <div style={{
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"10px 16px", borderBottom:"1px solid rgba(255,255,255,0.06)",
      }}>
        <span style={{
          fontSize:11, fontWeight:600, color:"#8b949e",
          textTransform:"uppercase", letterSpacing:"0.06em",
          display:"flex", alignItems:"center", gap:6,
        }}>
          🛡 Triage Assessment
        </span>
        <UrgencyBadge urgency={outcome.urgency}/>
      </div>

      {/* Body */}
      <div style={{ padding:"14px 16px" }}>
        <p style={{ fontSize:13.5, color:"#c9d1d9", lineHeight:1.65 }}>
          {outcome.advice_text}
        </p>
        {outcome.symptoms_summary && (
          <p style={{
            marginTop:10, paddingTop:10, fontSize:12, color:"#8b949e",
            borderTop:"1px solid rgba(255,255,255,0.06)",
          }}>
            <span style={{ color:"#c9d1d9", fontWeight:500 }}>Summary: </span>
            {outcome.symptoms_summary}
          </p>
        )}
      </div>
    </div>
  );
}