import { useState, useEffect, useRef } from "react";
import { Send, Loader2, CheckCircle2 } from "lucide-react";
import ChatBubble    from "../components/ChatBubble";
import TriageResult  from "../components/TriageResult";
import SessionList   from "../components/SessionList";
import { sendMessage, fetchHistory, fetchSessions } from "../api";

const WELCOME = "Hello! I'm your AI triage assistant 👋\nPlease describe your symptoms and I'll help assess the urgency. What's brought you here today?";

export default function Chat() {
  const [sessions,      setSessions]    = useState([]);
  const [sessionId,     setSessionId]   = useState(null);
  const [messages,      setMessages]    = useState([]);
  const [triageOutcome, setTriage]      = useState(null);
  const [input,         setInput]       = useState("");
  const [loading,       setLoading]     = useState(false);
  const [isComplete,    setIsComplete]  = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => { loadSessions(); }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function loadSessions() {
    try {
      const res = await fetchSessions();
      setSessions(res.data.sessions);
    } catch {}
  }

  async function selectSession(id) {
    setSessionId(id);
    setMessages([]);
    setTriage(null);
    setIsComplete(false);
    setInput("");
    try {
      const res = await fetchHistory(id);
      setMessages(res.data.messages.map(m => ({ role: m.role, content: m.content })));
      if (res.data.triage_outcome) {
        setTriage(res.data.triage_outcome);
        setIsComplete(true);
      }
    } catch {}
  }

  function startNew() {
    setSessionId(null);
    setMessages([{ role: "assistant", content: WELCOME }]);
    setTriage(null);
    setIsComplete(false);
    setInput("");
    setTimeout(() => textareaRef.current?.focus(), 100);
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || loading || isComplete) return;

    setInput("");
    setMessages(prev => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const res  = await sendMessage(text, sessionId || undefined);
      const data = res.data;

      if (!sessionId && data.session_id) {
        setSessionId(data.session_id);
        loadSessions();
      }

      if (data.reply) {
        setMessages(prev => [...prev, { role: "assistant", content: data.reply }]);
      }

      if (data.triage_outcome) setTriage(data.triage_outcome);
      if (data.is_complete) {
        setIsComplete(true);
        loadSessions();
      }
    } catch {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Something went wrong. Please try again.",
      }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // Auto-resize textarea
  function handleInput(e) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-screen overflow-hidden bg-brand-bg">

      {/* Sidebar */}
      <SessionList
        sessions={sessions}
        activeId={sessionId}
        onSelect={selectSession}
        onNew={startNew}
      />

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0 relative">

        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4
                           border-b border-brand-border bg-brand-surface/80
                           backdrop-blur-sm flex-shrink-0">
          <div>
            <h1 className="text-sm font-semibold text-brand-text">
              Symptom Consultation
            </h1>
            {sessionId
              ? <p className="text-xs text-brand-muted mt-0.5 font-mono">
                  {sessionId.slice(0, 8)}…
                </p>
              : <p className="text-xs text-brand-muted mt-0.5">
                  Start a new consultation
                </p>
            }
          </div>
          {isComplete && (
            <div className="flex items-center gap-1.5 text-xs text-green-400
                            bg-green-400/10 border border-green-400/20
                            px-3 py-1.5 rounded-full">
              <CheckCircle2 size={13} />
              Complete
            </div>
          )}
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto chat-scroll px-6 py-6">
          {isEmpty ? (
            /* Empty state */
            <div className="flex flex-col items-center justify-center h-full gap-5 text-center">
              <div className="w-16 h-16 rounded-2xl bg-brand-accent/10
                              border border-brand-accent/20 flex items-center
                              justify-center text-3xl shadow-xl shadow-brand-accent/5">
                🩺
              </div>
              <div className="space-y-1.5">
                <h2 className="text-base font-semibold text-brand-text">
                  AI Medical Triage
                </h2>
                <p className="text-sm text-brand-muted max-w-xs leading-relaxed">
                  Describe your symptoms and get an instant triage assessment with next-step advice.
                </p>
              </div>
              <button
                onClick={startNew}
                className="px-5 py-2.5 bg-brand-accent hover:bg-brand-accentHover
                           text-white rounded-xl text-sm font-medium transition-all
                           duration-150 shadow-lg shadow-brand-accent/25 active:scale-95">
                Start consultation
              </button>
            </div>
          ) : (
            <div className="space-y-4 max-w-3xl mx-auto">
              {messages.map((m, i) => (
                <ChatBubble key={i} role={m.role} content={m.content} />
              ))}

              {/* Typing indicator */}
              {loading && (
                <div className="flex gap-3 msg-enter">
                  <div className="w-8 h-8 rounded-full bg-brand-card border border-brand-border
                                  flex items-center justify-center flex-shrink-0">
                    <Loader2 size={13} className="text-brand-accent animate-spin" />
                  </div>
                  <div className="bg-brand-card border border-brand-border rounded-2xl
                                  rounded-tl-none px-4 py-3 flex items-center gap-1.5">
                    {[0, 1, 2].map(i => (
                      <span key={i}
                        className="w-1.5 h-1.5 bg-brand-muted rounded-full typing-dot"
                        style={{ animationDelay: `${i * 160}ms` }} />
                    ))}
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Triage result */}
        {triageOutcome && (
          <div className="max-w-3xl mx-auto w-full px-0">
            <TriageResult outcome={triageOutcome} />
          </div>
        )}

        {/* Input */}
        <div className="flex-shrink-0 px-6 py-4 border-t border-brand-border
                        bg-brand-surface/80 backdrop-blur-sm">
          {isComplete ? (
            <div className="max-w-3xl mx-auto flex items-center justify-between">
              <p className="text-sm text-brand-muted">
                Consultation complete.
              </p>
              <button
                onClick={startNew}
                className="px-4 py-2 bg-brand-accent hover:bg-brand-accentHover
                           text-white rounded-xl text-sm font-medium transition-all
                           duration-150 active:scale-95">
                New consultation
              </button>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto flex gap-3 items-end">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInput}
                onKeyDown={handleKey}
                placeholder={isEmpty ? "Start by describing your symptoms…" : "Reply to the assistant…"}
                rows={1}
                disabled={loading}
                className="flex-1 bg-brand-card border border-brand-border rounded-xl
                           px-4 py-3 text-sm text-brand-text placeholder-brand-muted
                           resize-none focus:outline-none focus:border-brand-accent/50
                           transition-colors disabled:opacity-50 leading-relaxed"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="flex-shrink-0 w-10 h-10 rounded-xl bg-brand-accent
                           hover:bg-brand-accentHover disabled:opacity-30
                           disabled:cursor-not-allowed flex items-center justify-center
                           transition-all duration-150 active:scale-95
                           shadow-lg shadow-brand-accent/20">
                <Send size={15} className="text-white" />
              </button>
            </div>
          )}
          <p className="text-[10px] text-brand-muted/40 text-center mt-2">
            Enter to send · Shift+Enter for new line
          </p>
        </div>
      </main>
    </div>
  );
}