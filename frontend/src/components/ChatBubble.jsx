import { Bot, User } from "lucide-react";

function format(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>");
}

export default function ChatBubble({ role, content }) {
  const isUser = role === "user";

  return (
    <div className={`flex gap-3 msg-enter ${isUser ? "flex-row-reverse" : "flex-row"}`}>

      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center
          justify-center shadow-lg
          ${isUser
            ? "bg-brand-accent text-white"
            : "bg-brand-card border border-brand-border text-brand-accent"}`}>
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>

      {/* Bubble */}
      <div className={`max-w-[72%] px-4 py-3 rounded-2xl text-sm leading-relaxed
          shadow-sm chat-bubble
          ${isUser
            ? "bg-brand-accent text-white rounded-tr-none"
            : "bg-brand-card border border-brand-border text-brand-text rounded-tl-none"}`}
        dangerouslySetInnerHTML={{ __html: format(content) }}
      />
    </div>
  );
}