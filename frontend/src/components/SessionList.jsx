import { MessageSquare, Plus, Clock, Activity } from "lucide-react";

function timeAgo(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1)  return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)  return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default function SessionList({ sessions, activeId, onSelect, onNew }) {
  return (
    <aside className="w-60 flex-shrink-0 flex flex-col h-screen
                      bg-brand-surface border-r border-brand-border">

      {/* Logo */}
      <div className="px-4 pt-5 pb-4 border-b border-brand-border">
        <div className="flex items-center gap-2.5 mb-4">
          <div className="w-8 h-8 rounded-xl bg-brand-accent/15 border border-brand-accent/20
                          flex items-center justify-center">
            <Activity size={15} className="text-brand-accent" />
          </div>
          <div>
            <p className="text-sm font-semibold text-brand-text leading-none">MedTriage</p>
            <p className="text-[10px] text-brand-muted mt-0.5">AI Assistant</p>
          </div>
        </div>

        <button
          onClick={onNew}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-3
                     rounded-xl bg-brand-accent hover:bg-brand-accentHover
                     text-white text-sm font-medium transition-all duration-150
                     shadow-lg shadow-brand-accent/20 active:scale-95">
          <Plus size={15} />
          New Consultation
        </button>
      </div>

      {/* Sessions */}
      <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
        {sessions.length === 0 ? (
          <div className="text-center mt-10 px-4">
            <MessageSquare size={24} className="text-brand-border mx-auto mb-2" />
            <p className="text-xs text-brand-muted">No sessions yet</p>
          </div>
        ) : (
          sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => onSelect(s.id)}
              className={`w-full text-left px-3 py-2.5 rounded-xl transition-all duration-150
                group border
                ${activeId === s.id
                  ? "bg-brand-accent/10 border-brand-accent/25 shadow-sm"
                  : "border-transparent hover:bg-brand-card hover:border-brand-border"}`}>

              <div className="flex items-center justify-between mb-1">
                <span className={`text-xs font-medium truncate
                  ${activeId === s.id ? "text-brand-accent" : "text-brand-subtle"}`}>
                  Session {s.id.slice(0, 8)}
                </span>
                <span className={`text-[10px] flex items-center gap-0.5
                  ${activeId === s.id ? "text-brand-accent/60" : "text-brand-muted"}`}>
                  <Clock size={9} />
                  {timeAgo(s.created_at)}
                </span>
              </div>

              <div className="flex items-center gap-1">
                <MessageSquare size={10} className="text-brand-muted flex-shrink-0" />
                <span className="text-[11px] text-brand-muted">
                  {s.message_count} message{s.message_count !== 1 ? "s" : ""}
                </span>
              </div>
            </button>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-brand-border">
        <p className="text-[10px] text-brand-muted/50 text-center leading-relaxed">
          AI guidance only — not a diagnosis.<br />
          Emergencies: call <span className="text-red-400/70">999 · 112 · 911</span>
        </p>
      </div>
    </aside>
  );
}