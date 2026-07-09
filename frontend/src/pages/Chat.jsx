import { useState, useEffect, useRef } from "react";
import { Send, Loader2, CheckCircle2 } from "lucide-react";
import ChatBubble   from "../components/ChatBubble";
import TriageResult from "../components/TriageResult";
import SessionList  from "../components/SessionList";
import { sendMessage, fetchHistory, fetchSessions } from "../api";

const WELCOME = "Hello! I'm your AI triage assistant.\nPlease describe your symptoms and I'll help assess the urgency. What's brought you here today?";

export default function Chat() {
  const [sessions,  setSessions]  = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [messages,  setMessages]  = useState([]);
  const [outcome,   setOutcome]   = useState(null);
  const [input,     setInput]     = useState("");
  const [loading,   setLoading]   = useState(false);
  const [done,      setDone]      = useState(false);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => { loadSessions(); }, []);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior:"smooth" });
  }, [messages, loading]);

  async function loadSessions() {
    try { const r = await fetchSessions(); setSessions(r.data.sessions); } catch {}
  }

  async function selectSession(id) {
    setSessionId(id); setMessages([]); setOutcome(null); setDone(false); setInput("");
    try {
      const r = await fetchHistory(id);
      setMessages(r.data.messages.map(m => ({ role:m.role, content:m.content })));
      if (r.data.triage_outcome) { setOutcome(r.data.triage_outcome); setDone(true); }
    } catch {}
  }

  function startNew() {
    setSessionId(null);
    setMessages([{ role:"assistant", content:WELCOME }]);
    setOutcome(null); setDone(false); setInput("");
    setTimeout(() => inputRef.current?.focus(), 80);
  }

  async function send() {
    const text = input.trim();
    if (!text || loading || done) return;
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    setMessages(p => [...p, { role:"user", content:text }]);
    setLoading(true);
    try {
      const r = await sendMessage(text, sessionId || undefined);
      const d = r.data;
      if (!sessionId && d.session_id) { setSessionId(d.session_id); loadSessions(); }
      if (d.reply) setMessages(p => [...p, { role:"assistant", content:d.reply }]);
      if (d.triage_outcome) setOutcome(d.triage_outcome);
      if (d.is_complete) { setDone(true); loadSessions(); }
    } catch {
      setMessages(p => [...p, {
        role:"assistant", content:"Something went wrong. Please try again.",
      }]);
    } finally { setLoading(false); }
  }

  function onKey(e) {
    if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  }

  function onInput(e) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  }

  return (
    <div style={{
      display:"flex", height:"100vh", overflow:"hidden",
      background:"#0d1117", fontFamily:"Inter, system-ui, sans-serif",
    }}>
      <SessionList sessions={sessions} activeId={sessionId}
                   onSelect={selectSession} onNew={startNew}/>

      <main style={{ flex:1, display:"flex", flexDirection:"column", minWidth:0 }}>

        {/* Header */}
        <header style={{
          display:"flex", alignItems:"center", justifyContent:"space-between",
          padding:"14px 24px", borderBottom:"1px solid #30363d",
          background:"#161b22", flexShrink:0,
        }}>
          <div>
            <div style={{ fontSize:15, fontWeight:600, color:"#e2e8f0" }}>
              Symptom Consultation
            </div>
            <div style={{ fontSize:11, color:"#8b949e", marginTop:2, fontFamily:"monospace" }}>
              {sessionId ? `${sessionId.slice(0,8)}…` : "Start a new consultation"}
            </div>
          </div>
          {done && (
            <span style={{
              display:"inline-flex", alignItems:"center", gap:5,
              padding:"4px 12px", borderRadius:20, fontSize:11, fontWeight:500,
              background:"rgba(74,222,128,0.1)", color:"#4ade80",
              border:"1px solid rgba(74,222,128,0.2)",
            }}>
              <CheckCircle2 size={12}/> Complete
            </span>
          )}
        </header>

        {/* Messages */}
        <div style={{
          flex:1, overflowY:"auto", padding:"24px",
          display:"flex", flexDirection:"column", gap:14,
        }}>
          {messages.length === 0 ? (
            /* Empty state */
            <div style={{
              display:"flex", flexDirection:"column", alignItems:"center",
              justifyContent:"center", height:"100%", gap:16, textAlign:"center",
            }}>
              <div style={{
                width:58, height:58, borderRadius:16, fontSize:26,
                background:"rgba(88,166,255,0.08)",
                border:"1px solid rgba(88,166,255,0.15)",
                display:"flex", alignItems:"center", justifyContent:"center",
              }}>🩺</div>
              <div>
                <div style={{ fontSize:16, fontWeight:600, color:"#e2e8f0", marginBottom:8 }}>
                  AI Medical Triage
                </div>
                <div style={{ fontSize:13.5, color:"#8b949e", maxWidth:300, lineHeight:1.65 }}>
                  Describe your symptoms and get an instant triage assessment with next-step advice.
                </div>
              </div>
              <button
                onClick={startNew}
                style={{
                  marginTop:4, padding:"10px 22px",
                  background:"#1f6feb", color:"#fff",
                  border:"none", borderRadius:8,
                  fontSize:13, fontWeight:500, cursor:"pointer",
                }}
              >
                Start consultation
              </button>
            </div>
          ) : (
            <>
              {messages.map((m,i) => (
                <ChatBubble key={i} role={m.role} content={m.content}/>
              ))}

              {/* Typing indicator */}
              {loading && (
                <div style={{ display:"flex", gap:10, alignItems:"flex-end" }}>
                  <div style={{
                    width:30, height:30, borderRadius:"50%", flexShrink:0,
                    display:"flex", alignItems:"center", justifyContent:"center",
                    background:"#1c2128", border:"1px solid #30363d",
                  }}>
                    <Loader2 size={13} color="#58a6ff"
                             style={{ animation:"spin 1s linear infinite" }}/>
                  </div>
                  <div style={{
                    background:"#1c2128", border:"1px solid #30363d",
                    borderRadius:16, borderBottomLeftRadius:4,
                    padding:"10px 14px", display:"flex", gap:5, alignItems:"center",
                  }}>
                    {[0,1,2].map(i => (
                      <span key={i} style={{
                        width:6, height:6, borderRadius:"50%", background:"#8b949e",
                        display:"inline-block",
                        animation:`blink 1.2s ${i*160}ms infinite ease-in-out`,
                      }}/>
                    ))}
                  </div>
                </div>
              )}
              <div ref={bottomRef}/>
            </>
          )}
        </div>

        {/* Triage result */}
        {outcome && <TriageResult outcome={outcome}/>}

        {/* Input area */}
        <div style={{
          flexShrink:0, padding:"14px 24px",
          borderTop:"1px solid #30363d", background:"#161b22",
        }}>
          {done ? (
            <div style={{
              display:"flex", alignItems:"center", justifyContent:"space-between",
              maxWidth:820, margin:"0 auto",
            }}>
              <span style={{ fontSize:13, color:"#8b949e" }}>
                Consultation complete.
              </span>
              <button
                onClick={startNew}
                style={{
                  padding:"8px 16px", background:"#1f6feb", color:"#fff",
                  border:"none", borderRadius:8,
                  fontSize:13, fontWeight:500, cursor:"pointer",
                }}
              >
                New consultation
              </button>
            </div>
          ) : (
            <div style={{ maxWidth:820, margin:"0 auto" }}>
              <div style={{ display:"flex", gap:10, alignItems:"flex-end" }}>
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={onInput}
                  onKeyDown={onKey}
                  disabled={loading}
                  placeholder={
                    messages.length===0
                      ? "Describe your symptoms…"
                      : "Reply to the assistant…"
                  }
                  rows={1}
                  style={{
                    flex:1, background:"#0d1117",
                    border:"1px solid #30363d", borderRadius:10,
                    padding:"10px 14px", fontSize:13.5, color:"#e2e8f0",
                    resize:"none", lineHeight:1.55, outline:"none",
                    fontFamily:"Inter, system-ui, sans-serif",
                    transition:"border-color 0.15s",
                  }}
                  onFocus={e  => e.target.style.borderColor="rgba(88,166,255,0.5)"}
                  onBlur={e   => e.target.style.borderColor="#30363d"}
                />
                <button
                  onClick={send}
                  disabled={!input.trim() || loading}
                  style={{
                    width:40, height:40, borderRadius:10,
                    background:"#1f6feb", border:"none",
                    display:"flex", alignItems:"center", justifyContent:"center",
                    cursor: (!input.trim()||loading) ? "not-allowed" : "pointer",
                    opacity: (!input.trim()||loading) ? 0.35 : 1,
                    flexShrink:0, transition:"opacity 0.15s",
                  }}
                >
                  <Send size={15} color="#fff"/>
                </button>
              </div>
              <p style={{
                fontSize:10, color:"rgba(139,148,158,0.4)",
                textAlign:"center", marginTop:7,
              }}>
                Enter to send · Shift+Enter for new line
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}