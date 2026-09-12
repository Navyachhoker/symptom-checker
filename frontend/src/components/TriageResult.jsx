import { useState } from "react";
import UrgencyBadge from "./UrgencyBadge";

// ── Confidence meter ──────────────────────────────────────────
function ConfidenceMeter({ confidence }) {
  if (confidence == null) return null;
  const color =
    confidence >= 80 ? "#4ade80" :
    confidence >= 60 ? "#fbbf24" : "#f87171";

  return (
    <div style={{ marginTop: 12 }}>
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: 5,
      }}>
        <span style={{ fontSize: 11, color: "#8b949e", fontWeight: 500 }}>
          AI Confidence
        </span>
        <span style={{ fontSize: 12, fontWeight: 600, color }}>{confidence}%</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.08)" }}>
        <div style={{
          height: 4, borderRadius: 2,
          width: `${confidence}%`,
          background: color,
          transition: "width 0.6s ease",
        }} />
      </div>
      <p style={{ fontSize: 10, color: "#8b949e", marginTop: 5 }}>
        {confidence >= 80
          ? "High confidence — sufficient clinical context collected"
          : confidence >= 60
          ? "Moderate confidence — some information was missing"
          : "Low confidence — consult a medical professional directly"}
      </p>
    </div>
  );
}

// ── Single trace step ─────────────────────────────────────────
function TraceStep({ step, isLast }) {
  const agentConfig = {
    orchestrator:             { color: "#58a6ff", bg: "rgba(88,166,255,0.12)",   icon: "🧠", label: "Orchestrator"           },
    red_flag_checker:         { color: "#f87171", bg: "rgba(248,113,113,0.12)", icon: "⚠️", label: "Red flag check"          },
    cardiac_specialist:       { color: "#c084fc", bg: "rgba(192,132,252,0.12)", icon: "❤️", label: "Cardiac specialist"      },
    respiratory_specialist:   { color: "#c084fc", bg: "rgba(192,132,252,0.12)", icon: "🫁", label: "Respiratory specialist"  },
    mental_health_specialist: { color: "#c084fc", bg: "rgba(192,132,252,0.12)", icon: "🧩", label: "Mental health specialist"},
    pediatric_specialist:     { color: "#c084fc", bg: "rgba(192,132,252,0.12)", icon: "👶", label: "Pediatric specialist"    },
    general_specialist:       { color: "#c084fc", bg: "rgba(192,132,252,0.12)", icon: "🩺", label: "General specialist"      },
    safety_agent:             { color: "#4ade80", bg: "rgba(74,222,128,0.12)",  icon: "🛡", label: "Safety agent"            },
  };

  const actionConfig = {
    ask_question:         { label: "Asked question",    color: "#58a6ff" },
    call_specialist:      { label: "Called specialist", color: "#c084fc" },
    run_tool:             { label: "Ran tool",          color: "#fbbf24" },
    conclude:             { label: "Concluded",         color: "#4ade80" },
    escalate:             { label: "Escalated",         color: "#f87171" },
    specialist_assessment:{ label: "Assessment",        color: "#c084fc" },
    emergency_bypass:     { label: "Emergency bypass",  color: "#f87171" },
    approve:              { label: "Approved",          color: "#4ade80" },
    urgency_override:     { label: "Upgraded urgency",  color: "#f87171" },
  };

  const agent  = agentConfig[step.agent]   || { color: "#8b949e", bg: "rgba(139,148,158,0.12)", icon: "•", label: step.agent  };
  const action = actionConfig[step.action] || { label: step.action, color: "#8b949e" };

  return (
    <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>

      {/* Dot + connecting line */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
        <div style={{
          width: 24, height: 24, borderRadius: "50%",
          background: agent.bg,
          border: `1px solid ${agent.color}44`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, flexShrink: 0,
        }}>
          {agent.icon}
        </div>
        {!isLast && (
          <div style={{
            width: 1, flex: 1, minHeight: 10,
            background: "rgba(255,255,255,0.06)",
            margin: "3px 0",
          }} />
        )}
      </div>

      {/* Content */}
      <div style={{ flex: 1, paddingBottom: isLast ? 0 : 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, fontWeight: 500, color: "#c9d1d9" }}>
            {agent.label}
          </span>
          <span style={{
            fontSize: 10, padding: "1px 7px", borderRadius: 20,
            background: "rgba(255,255,255,0.06)",
            color: action.color,
            border: `1px solid ${action.color}33`,
          }}>
            {action.label}
          </span>
          {step.confidence != null && (
            <span style={{ fontSize: 10, color: "#8b949e" }}>
              {step.confidence}%
            </span>
          )}
        </div>

        {step.reasoning && (
          <p style={{ fontSize: 11, color: "#8b949e", marginTop: 4, lineHeight: 1.5 }}>
            {step.reasoning}
          </p>
        )}

        {step.output && step.action !== "ask_question" && (
          <p style={{
            fontSize: 11, color: "#6e7681", marginTop: 3,
            lineHeight: 1.4, fontStyle: "italic",
          }}>
            {step.output.length > 120 ? step.output.slice(0, 120) + "…" : step.output}
          </p>
        )}

        {step.action === "urgency_override" && (
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 4,
            marginTop: 4, fontSize: 10, padding: "2px 8px", borderRadius: 20,
            background: "rgba(248,113,113,0.1)", color: "#f87171",
            border: "1px solid rgba(248,113,113,0.2)",
          }}>
            ⬆ Urgency upgraded by safety agent
          </span>
        )}

        {step.action === "approve" && step.agent === "safety_agent" && (
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 4,
            marginTop: 4, fontSize: 10, padding: "2px 8px", borderRadius: 20,
            background: "rgba(74,222,128,0.1)", color: "#4ade80",
            border: "1px solid rgba(74,222,128,0.2)",
          }}>
            ✓ Safety approved
          </span>
        )}

        {step.action === "emergency_bypass" && (
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 4,
            marginTop: 4, fontSize: 10, padding: "2px 8px", borderRadius: 20,
            background: "rgba(248,113,113,0.1)", color: "#f87171",
            border: "1px solid rgba(248,113,113,0.2)",
          }}>
            ⚡ LLM bypassed — hard-coded safety rule triggered
          </span>
        )}
      </div>
    </div>
  );
}

// ── Trace panel ───────────────────────────────────────────────
function TracePanel({ trace }) {
  const [open, setOpen] = useState(false);
  if (!trace || trace.length === 0) return null;

  return (
    <div style={{
      marginTop: 12,
      borderTop: "1px solid rgba(255,255,255,0.06)",
      paddingTop: 12,
    }}>
      {/* Toggle */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          width: "100%", background: "none", border: "none", cursor: "pointer",
          padding: 0, marginBottom: open ? 12 : 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            fontSize: 10, fontWeight: 600, color: "#8b949e",
            textTransform: "uppercase", letterSpacing: "0.07em",
          }}>
            Reasoning trace
          </span>
          <span style={{
            fontSize: 10, padding: "1px 6px", borderRadius: 20,
            background: "rgba(255,255,255,0.06)",
            color: "#8b949e", border: "1px solid rgba(255,255,255,0.08)",
          }}>
            {trace.length} step{trace.length !== 1 ? "s" : ""}
          </span>
        </div>
        <span style={{
          fontSize: 13, color: "#8b949e",
          transform: open ? "rotate(180deg)" : "rotate(0deg)",
          transition: "transform 0.2s ease",
          display: "inline-block",
        }}>
          ▾
        </span>
      </button>

      {/* Steps */}
      {open && (
        <div>
          {trace.map((step, i) => (
            <TraceStep
              key={i}
              step={step}
              isLast={i === trace.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Specialist badge ──────────────────────────────────────────
function SpecialistBadge({ specialist }) {
  if (!specialist) return null;
  const icons = {
    cardiac:      "❤️",
    respiratory:  "🫁",
    mental:       "🧩",
    pediatric:    "👶",
    general:      "🩺",
    mental_health:"🧩",
  };
  const icon = icons[specialist] || "🩺";

  return (
    <div style={{ marginTop: 8 }}>
      <span style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        fontSize: 11, padding: "3px 9px", borderRadius: 20,
        background: "rgba(192,132,252,0.1)", color: "#c084fc",
        border: "1px solid rgba(192,132,252,0.2)",
      }}>
        {icon} {specialist} specialist consulted
      </span>
    </div>
  );
}

// ── Urgency border/bg colours ─────────────────────────────────
const urgencyColors = {
  low:       { border: "#166534", bg: "#0a1f12" },
  moderate:  { border: "#92400e", bg: "#1c1500" },
  high:      { border: "#991b1b", bg: "#1c0808" },
  emergency: { border: "#9a3412", bg: "#1c0e05" },
};

// ── Main export ───────────────────────────────────────────────
export default function TriageResult({ outcome, trace }) {
  if (!outcome) return null;
  const c = urgencyColors[outcome.urgency] || urgencyColors.moderate;

  return (
    <div style={{
      margin: "0 16px 12px",
      borderRadius: 12,
      overflow: "hidden",
      border: `1px solid ${c.border}`,
      background: c.bg,
      flexShrink: 0,
    }}>

      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 16px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}>
        <span style={{
          fontSize: 11, fontWeight: 600, color: "#8b949e",
          textTransform: "uppercase", letterSpacing: "0.06em",
        }}>
          🛡 Triage Assessment
        </span>
        <UrgencyBadge urgency={outcome.urgency} />
      </div>

      {/* Body */}
      <div style={{ padding: "14px 16px" }}>

        {/* Advice */}
        <p style={{ fontSize: 13.5, color: "#c9d1d9", lineHeight: 1.65 }}>
          {outcome.advice_text}
        </p>

        {/* Confidence meter */}
        <ConfidenceMeter confidence={outcome.confidence} />

        {/* Specialist badge */}
        <SpecialistBadge specialist={outcome.specialist_called} />

        {/* Summary */}
        {outcome.symptoms_summary && (
          <p style={{
            marginTop: 10, paddingTop: 10, fontSize: 12, color: "#8b949e",
            borderTop: "1px solid rgba(255,255,255,0.06)",
          }}>
            <span style={{ color: "#c9d1d9", fontWeight: 500 }}>Summary: </span>
            {outcome.symptoms_summary}
          </p>
        )}

        {/* Reasoning trace */}
        <TracePanel trace={trace} />
      </div>
    </div>
  );
}