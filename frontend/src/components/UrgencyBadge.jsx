const map = {
  low:       { icon: "🟢", label: "Low Urgency",   cls: "text-green-400  bg-green-400/10  border-green-400/20"  },
  moderate:  { icon: "🟡", label: "Moderate",       cls: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20" },
  high:      { icon: "🔴", label: "High Urgency",   cls: "text-red-400    bg-red-400/10    border-red-400/20"    },
  emergency: { icon: "🚨", label: "Emergency",      cls: "text-orange-400 bg-orange-400/10 border-orange-400/20" },
};

export default function UrgencyBadge({ urgency, large = false }) {
  const c = map[urgency] || map.moderate;
  return (
    <span className={`inline-flex items-center gap-2 border rounded-full font-medium
      ${large ? "px-4 py-1.5 text-sm" : "px-3 py-1 text-xs"} ${c.cls}`}>
      <span>{c.icon}</span>
      {c.label}
    </span>
  );
}