import UrgencyBadge from "./UrgencyBadge";
import { ShieldCheck } from "lucide-react";

const borderColor = {
  low:       "border-green-500/30",
  moderate:  "border-yellow-500/30",
  high:      "border-red-500/30",
  emergency: "border-orange-500/30",
};

const bgColor = {
  low:       "bg-green-500/5",
  moderate:  "bg-yellow-500/5",
  high:      "bg-red-500/5",
  emergency: "bg-orange-500/5",
};

export default function TriageResult({ outcome }) {
  if (!outcome) return null;

  const border = borderColor[outcome.urgency] || "border-brand-border";
  const bg     = bgColor[outcome.urgency]     || "bg-brand-card";

  return (
    <div className={`mx-4 mb-3 rounded-2xl border ${border} ${bg} overflow-hidden`}>

      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3
                      border-b border-white/5">
        <div className="flex items-center gap-2">
          <ShieldCheck size={15} className="text-brand-accent" />
          <span className="text-xs font-semibold text-brand-subtle uppercase tracking-widest">
            Triage Assessment
          </span>
        </div>
        <UrgencyBadge urgency={outcome.urgency} />
      </div>

      {/* Body */}
      <div className="px-5 py-4 space-y-3">
        <p className="text-sm text-brand-text leading-relaxed">
          {outcome.advice_text}
        </p>
        {outcome.symptoms_summary && (
          <div className="pt-3 border-t border-white/5">
            <p className="text-xs text-brand-muted leading-relaxed">
              <span className="text-brand-subtle font-medium">Summary: </span>
              {outcome.symptoms_summary}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}