import { Bot, User } from "lucide-react";

function fmt(t) {
  return t
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>");
}

export default function ChatBubble({ role, content }) {
  const isUser = role === "user";
  return (
    <div style={{
      display:"flex", gap:10, alignItems:"flex-end",
      flexDirection: isUser ? "row-reverse" : "row",
      animation:"fadeUp 0.18s ease forwards",
    }}>

      {/* Avatar */}
      <div style={{
        width:30, height:30, borderRadius:"50%", flexShrink:0,
        display:"flex", alignItems:"center", justifyContent:"center",
        background: isUser ? "#1f6feb" : "#1c2128",
        border: isUser ? "none" : "1px solid #30363d",
        color: isUser ? "#fff" : "#58a6ff",
      }}>
        {isUser ? <User size={14}/> : <Bot size={14}/>}
      </div>

      {/* Bubble */}
      <div
        style={{
          maxWidth:"70%", padding:"10px 14px", fontSize:13.5,
          lineHeight:1.65, borderRadius:16,
          borderBottomRightRadius: isUser ? 4 : 16,
          borderBottomLeftRadius:  isUser ? 16 : 4,
          background: isUser ? "#1f6feb" : "#1c2128",
          color: isUser ? "#fff" : "#e2e8f0",
          border: isUser ? "none" : "1px solid #30363d",
          wordBreak:"break-word",
        }}
        dangerouslySetInnerHTML={{ __html: fmt(content) }}
      />
    </div>
  );
}