import { MessageSquare, Plus, Clock, Activity } from "lucide-react";

function ago(d) {
  const m = Math.floor((Date.now() - new Date(d)) / 60000);
  if (m < 1)  return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function SessionList({ sessions, activeId, onSelect, onNew }) {
  return (
    <aside style={{
      width:230, flexShrink:0, display:"flex", flexDirection:"column",
      height:"100vh", background:"#161b22", borderRight:"1px solid #30363d",
    }}>

      {/* Logo + New button */}
      <div style={{ padding:"18px 16px 14px", borderBottom:"1px solid #30363d" }}>
        <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:14 }}>
          <div style={{
            width:34, height:34, borderRadius:8,
            background:"rgba(88,166,255,0.1)", border:"1px solid rgba(88,166,255,0.2)",
            display:"flex", alignItems:"center", justifyContent:"center",
          }}>
            <Activity size={16} color="#58a6ff"/>
          </div>
          <div>
            <div style={{ fontSize:14, fontWeight:600, color:"#e2e8f0", lineHeight:1.2 }}>
              MedTriage
            </div>
            <div style={{ fontSize:11, color:"#8b949e", marginTop:2 }}>
              AI Assistant
            </div>
          </div>
        </div>

        <button
          onClick={onNew}
          style={{
            display:"flex", alignItems:"center", justifyContent:"center", gap:6,
            width:"100%", padding:"9px 12px", borderRadius:8,
            fontSize:13, fontWeight:500,
            background:"#1f6feb", color:"#fff", border:"none", cursor:"pointer",
            transition:"background 0.15s",
          }}
          onMouseEnter={e => e.currentTarget.style.background="#1a5fcc"}
          onMouseLeave={e => e.currentTarget.style.background="#1f6feb"}
        >
          <Plus size={14} strokeWidth={2.5}/> New Consultation
        </button>
      </div>

      {/* Session list */}
      <div style={{ flex:1, overflowY:"auto", padding:"8px" }}>
        {sessions.length === 0 ? (
          <div style={{ textAlign:"center", marginTop:36, color:"#8b949e" }}>
            <MessageSquare size={24} style={{ margin:"0 auto 8px", opacity:0.3 }}/>
            <p style={{ fontSize:12 }}>No sessions yet</p>
          </div>
        ) : sessions.map(s => (
          <div
            key={s.id}
            onClick={() => onSelect(s.id)}
            style={{
              padding:"10px 12px", borderRadius:8, cursor:"pointer", marginBottom:2,
              background: activeId===s.id ? "rgba(88,166,255,0.1)" : "transparent",
              border:`1px solid ${activeId===s.id ? "rgba(88,166,255,0.25)" : "transparent"}`,
              transition:"all 0.12s",
            }}
            onMouseEnter={e => {
              if (activeId !== s.id) e.currentTarget.style.background = "rgba(255,255,255,0.04)";
            }}
            onMouseLeave={e => {
              if (activeId !== s.id) e.currentTarget.style.background = "transparent";
            }}
          >
            <p style={{
              fontSize:12, fontWeight:500, marginBottom:4, overflow:"hidden",
              textOverflow:"ellipsis", whiteSpace:"nowrap",
              color: activeId===s.id ? "#58a6ff" : "#c9d1d9",
            }}>
              Session {s.id.slice(0,8)}
            </p>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <span style={{
                fontSize:11, color:"#8b949e",
                display:"flex", alignItems:"center", gap:4,
              }}>
                <MessageSquare size={10}/> {s.message_count} msg{s.message_count!==1?"s":""}
              </span>
              <span style={{
                fontSize:10, color:"rgba(139,148,158,0.55)",
                display:"flex", alignItems:"center", gap:3,
              }}>
                <Clock size={9}/> {ago(s.created_at)}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div style={{
        padding:"12px 16px", borderTop:"1px solid #30363d",
        fontSize:11, color:"rgba(139,148,158,0.5)",
        textAlign:"center", lineHeight:1.7,
      }}>
        AI guidance only — not a diagnosis.<br/>
        Emergencies:{" "}
        <span style={{ color:"rgba(248,113,113,0.7)" }}>112 · 108 · 102</span>
      </div>
    </aside>
  );
}