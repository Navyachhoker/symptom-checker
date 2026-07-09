const cfg = {
  low:       { dot:"🟢", label:"Low urgency",  bg:"#0d2818", color:"#4ade80", border:"#166534" },
  moderate:  { dot:"🟡", label:"Moderate",      bg:"#2a1f00", color:"#fbbf24", border:"#92400e" },
  high:      { dot:"🔴", label:"High urgency",  bg:"#2a0a0a", color:"#f87171", border:"#991b1b" },
  emergency: { dot:"🚨", label:"Emergency",     bg:"#2a1200", color:"#fb923c", border:"#9a3412" },
};

export default function UrgencyBadge({ urgency }) {
  const c = cfg[urgency] || cfg.moderate;
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap:6,
      padding:"4px 12px", borderRadius:20, fontSize:12, fontWeight:500,
      background:c.bg, color:c.color, border:`1px solid ${c.border}`,
    }}>
      {c.dot} {c.label}
    </span>
  );
}