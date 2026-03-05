// frontend/src/app/page.tsx
"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import clsx from "clsx";

const API_URL = process.env.NEXT_PUBLIC_AGENT_API_URL ?? "http://localhost:8001";

interface Message {
  id:        string;
  role:      "user" | "assistant";
  content:   string;
  tool_hint?: string;
  ts:        number;
}

const EXAMPLES = [
  "What is the current unemployment trend over the last 12 months?",
  "Forecast GDP growth for next quarter given current conditions",
  "What did the Fed say about inflation in the most recent FOMC minutes?",
  "Compare current CPI to the 2008 financial crisis period",
  "What's the probability the Fed cuts rates in the next 2 months?",
];

// ── Markdown renderer (minimal, no deps) ─────────────────────────────────────
function renderMarkdown(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g,     "<em>$1</em>")
    .replace(/`(.+?)`/g,       "<code>$1</code>")
    .replace(/^#{1,3} (.+)$/gm, "<h3>$1</h3>")
    .replace(/^- (.+)$/gm,     "<li>$1</li>")
    .replace(/\n\n/g,           "</p><p>")
    .replace(/\[(\d+)\] /g,    "<span class='citation'>[$1]</span> ")
    .trim();
}

// ── Message bubble ────────────────────────────────────────────────────────────
function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div className={clsx("flex gap-3 animate-fadeIn", isUser && "flex-row-reverse")}>
      {/* Avatar */}
      <div className={clsx(
        "w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center font-mono text-xs font-bold mt-1",
        isUser ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
               : "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
      )}>
        {isUser ? "YOU" : "AI"}
      </div>

      {/* Bubble */}
      <div className={clsx(
        "max-w-[82%] rounded-2xl px-4 py-3 font-mono text-sm leading-relaxed",
        isUser
          ? "bg-surface-2 border border-amber-500/20 text-text-primary rounded-tr-sm"
          : "bg-surface-1 border border-cyan-500/10 text-text-primary rounded-tl-sm"
      )}>
        {msg.tool_hint && (
          <div className="flex items-center gap-1.5 mb-2 text-text-dim text-xs">
            <span className="animate-spin inline-block">⚙</span>
            <span>{msg.tool_hint}</span>
          </div>
        )}
        {isUser ? (
          <p>{msg.content}</p>
        ) : (
          <div
            className="prose-agent"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
          />
        )}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AgentChat() {
  const [messages,   setMessages]   = useState<Message[]>([]);
  const [input,      setInput]      = useState("");
  const [sessionId,  setSessionId]  = useState<string | null>(null);
  const [streaming,  setStreaming]  = useState(false);
  const [connected,  setConnected]  = useState<boolean | null>(null);
  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);
  const abortRef   = useRef<AbortController | null>(null);

  // Check API health
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(() => setConnected(true))
      .catch(() => setConnected(false));
  }, []);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || streaming) return;

    const userMsg: Message = {
      id: Date.now().toString(), role: "user", content: text.trim(), ts: Date.now(),
    };
    const asstMsg: Message = {
      id: (Date.now() + 1).toString(), role: "assistant", content: "", ts: Date.now(),
    };

    setMessages(prev => [...prev, userMsg, asstMsg]);
    setInput("");
    setStreaming(true);

    const abort = new AbortController();
    abortRef.current = abort;

    try {
      const resp = await fetch(`${API_URL}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ message: text.trim(), session_id: sessionId, stream: true }),
        signal:  abort.signal,
      });

      // Save session ID from header
      const sid = resp.headers.get("X-Session-ID");
      if (sid) setSessionId(sid);

      const reader  = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split("\n");

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") break;

          const chunk = data.replace(/\\n/g, "\n");

          if (chunk.includes("_Executing tools..._")) {
            setMessages(prev => prev.map(m =>
              m.id === asstMsg.id ? { ...m, tool_hint: "Running tools..." } : m
            ));
            continue;
          }

          buffer += chunk;
          setMessages(prev => prev.map(m =>
            m.id === asstMsg.id ? { ...m, content: buffer, tool_hint: undefined } : m
          ));
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setMessages(prev => prev.map(m =>
          m.id === asstMsg.id
            ? { ...m, content: "Error: Could not reach the agent API. Is it running?" }
            : m
        ));
      }
    } finally {
      setStreaming(false);
    }
  }, [streaming, sessionId]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const newSession = async () => {
    try {
      const r = await fetch(`${API_URL}/sessions`, { method: "POST" });
      const d = await r.json();
      setSessionId(d.session_id);
      setMessages([]);
    } catch {}
  };

  return (
    <div className="min-h-screen bg-surface flex flex-col font-mono">

      {/* Header */}
      <header className="border-b border-border px-6 py-4 flex items-center justify-between bg-surface-1/80 backdrop-blur sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 border border-cyan-500/60 rotate-45 relative">
            <div className="absolute inset-[3px] bg-cyan-500/20 rotate-0" />
          </div>
          <div>
            <span className="text-base font-bold tracking-tight text-text-primary font-display">
              ECON<span className="text-cyan-400">INTEL</span>
            </span>
            <span className="ml-2 text-xs text-text-dim">AI MACRO ANALYST</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {sessionId && (
            <span className="text-xs text-text-dim">
              session <span className="text-text-muted">{sessionId.slice(0, 8)}</span>
            </span>
          )}
          <div className={clsx(
            "w-2 h-2 rounded-full",
            connected === true  ? "bg-green-400 animate-pulse" :
            connected === false ? "bg-red-400" : "bg-yellow-400 animate-pulse"
          )} />
          <button
            onClick={newSession}
            className="text-xs text-text-muted border border-border px-2 py-1 rounded hover:border-cyan-500/40 hover:text-cyan-400 transition-colors"
          >
            NEW SESSION
          </button>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6 max-w-4xl mx-auto w-full">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-8 py-16">
            <div className="text-center">
              <h1 className="font-display text-3xl font-bold text-text-primary mb-2">
                Econ Intelligence Agent
              </h1>
              <p className="text-text-muted text-sm max-w-md">
                Ask anything about the US economy. I'll query your macro data,
                run ML forecasts, and search Fed & IMF documents to answer.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-2 w-full max-w-lg">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(ex)}
                  className="text-left text-xs text-text-muted border border-border rounded-lg px-4 py-3 hover:border-cyan-500/40 hover:text-cyan-400 transition-all hover:bg-surface-2"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {messages.map(msg => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}
            {streaming && messages.at(-1)?.content === "" && (
              <div className="flex gap-1.5 pl-11">
                {[0,1,2].map(i => (
                  <div key={i} className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 animate-bounce"
                    style={{ animationDelay: `${i * 150}ms` }} />
                ))}
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <footer className="border-t border-border px-4 py-4 bg-surface-1/80 backdrop-blur">
        <div className="max-w-4xl mx-auto flex gap-3 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the economy... (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={streaming}
            className={clsx(
              "flex-1 bg-surface-2 border border-border rounded-xl px-4 py-3",
              "text-sm text-text-primary placeholder-text-dim resize-none",
              "focus:outline-none focus:border-cyan-500/50 transition-colors",
              "min-h-[48px] max-h-[160px]",
              streaming && "opacity-50 cursor-not-allowed",
            )}
            style={{ height: "auto" }}
            onInput={e => {
              const t = e.target as HTMLTextAreaElement;
              t.style.height = "auto";
              t.style.height = Math.min(t.scrollHeight, 160) + "px";
            }}
          />
          <button
            onClick={() => streaming ? abortRef.current?.abort() : sendMessage(input)}
            className={clsx(
              "px-4 py-3 rounded-xl text-sm font-bold transition-all",
              streaming
                ? "bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30"
                : "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30",
            )}
          >
            {streaming ? "STOP" : "SEND"}
          </button>
        </div>
        <p className="text-center text-xs text-text-dim mt-2">
          Tools: PostgreSQL · ML Platform · Fed/IMF Documents
        </p>
      </footer>
    </div>
  );
}
