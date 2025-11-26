import { useState, useRef, useEffect } from "react";
import axios from "axios";

type Sender = "user" | "bot";

interface Message {
  sender: Sender;
  text: string;
}

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSending]);

  const sendMessage = async (messageText?: string) => {
    const textToSend = messageText || input.trim();
    if (!textToSend || isSending) return;

    const userMsg: Message = { sender: "user", text: textToSend };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsSending(true);

    try {
      const { data } = await axios.post<{ response: string }>(`${API_URL}/chat`, {
        message: textToSend,
      });
      setMessages((prev) => [...prev, { sender: "bot", text: data.response }]);
    } catch (error) {
      let errorMessage = "⚠️ Unable to reach the F1 backend. Check that it is running.";

      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED' || error.message.includes('Network Error')) {
          errorMessage = `⚠️ Cannot connect to backend at ${API_URL}. Make sure the server is running on port 8000.`;
        } else if (error.response) {
          errorMessage = `⚠️ Backend error: ${error.response.data?.detail || error.response.statusText}`;
        } else {
          errorMessage = `⚠️ Network error: ${error.message}`;
        }
      }

      console.error("Backend error:", error);
      setMessages((prev) => [...prev, { sender: "bot", text: errorMessage }]);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="flex h-screen bg-[#0d0d0d] text-white overflow-hidden">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-64" : "w-0"
        } bg-[#171717] border-r border-[#2d2d2d] transition-all duration-300 flex flex-col overflow-hidden`}
      >
        <div className="p-4 border-b border-[#2d2d2d]">
          <button
            onClick={clearChat}
            className="w-full px-4 py-2.5 bg-[#dc2626] hover:bg-[#b91c1c] rounded-lg transition-colors flex items-center gap-2 text-sm font-medium"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          <div className="text-xs text-[#8e8e93] px-3 py-2 font-medium">Recent</div>
          {messages.length === 0 && (
            <div className="text-sm text-[#8e8e93] px-3 py-2">No recent chats</div>
          )}
        </div>
        <div className="p-4 border-t border-[#2d2d2d] text-xs text-[#8e8e93] text-center">
          F1 Insight Chatbot
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-14 bg-[#171717] border-b border-[#2d2d2d] flex items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-[#2d2d2d] rounded-lg transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div>
              <h1 className="text-base font-semibold">F1 Insight</h1>
              <p className="text-xs text-[#8e8e93]">Powered by FastF1 & Ollama</p>
            </div>
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto bg-[#0d0d0d]">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-2xl px-4">
                <div className="mb-6">
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-[#dc2626] rounded-full mb-4">
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13 10V3L4 14h7v7l9-11h-7z"
                      />
                    </svg>
                  </div>
                  <h2 className="text-2xl font-semibold mb-2">F1 Insight Chatbot</h2>
                  <p className="text-[#8e8e93] mb-8">
                    Ask me anything about Formula 1 drivers, races, records, and statistics.
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div
                    onClick={() => sendMessage("Who won the last race?")}
                    className="p-4 bg-[#171717] border border-[#2d2d2d] rounded-lg hover:bg-[#1f1f1f] cursor-pointer transition-colors active:scale-95"
                  >
                    <div className="font-medium mb-1">Who won the last race?</div>
                    <div className="text-[#8e8e93] text-xs">Recent race results</div>
                  </div>
                  <div
                    onClick={() => sendMessage("Show me driver statistics")}
                    className="p-4 bg-[#171717] border border-[#2d2d2d] rounded-lg hover:bg-[#1f1f1f] cursor-pointer transition-colors active:scale-95"
                  >
                    <div className="font-medium mb-1">Driver statistics</div>
                    <div className="text-[#8e8e93] text-xs">Compare drivers</div>
                  </div>
                  <div
                    onClick={() => sendMessage("What are the fastest lap records?")}
                    className="p-4 bg-[#171717] border border-[#2d2d2d] rounded-lg hover:bg-[#1f1f1f] cursor-pointer transition-colors active:scale-95"
                  >
                    <div className="font-medium mb-1">Fastest lap records</div>
                    <div className="text-[#8e8e93] text-xs">Track records</div>
                  </div>
                  <div
                    onClick={() => sendMessage("What are the current championship standings?")}
                    className="p-4 bg-[#171717] border border-[#2d2d2d] rounded-lg hover:bg-[#1f1f1f] cursor-pointer transition-colors active:scale-95"
                  >
                    <div className="font-medium mb-1">Championship standings</div>
                    <div className="text-[#8e8e93] text-xs">Current season</div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`group mb-6 ${
                    msg.sender === "user" ? "bg-[#0d0d0d]" : "bg-[#0d0d0d]"
                  }`}
                >
                  <div className="flex gap-4">
                    <div
                      className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                        msg.sender === "user"
                          ? "bg-[#dc2626]"
                          : "bg-[#2d2d2d] group-hover:bg-[#3d3d3d] transition-colors"
                      }`}
                    >
                      {msg.sender === "user" ? (
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                        </svg>
                      ) : (
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M13 10V3L4 14h7v7l9-11h-7z"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium mb-1">
                        {msg.sender === "user" ? "You" : "F1 Insight"}
                      </div>
                      <div className="prose prose-invert max-w-none">
                        <div className="text-[#e5e5e5] whitespace-pre-wrap leading-relaxed">
                          {msg.text}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {isSending && (
                <div className="group mb-6">
                  <div className="flex gap-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[#2d2d2d] flex items-center justify-center">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 10V3L4 14h7v7l9-11h-7z"
                        />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium mb-1">F1 Insight</div>
                      <div className="flex gap-1 py-2">
                        <div className="w-2 h-2 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                        <div className="w-2 h-2 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                        <div className="w-2 h-2 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-[#2d2d2d] bg-[#171717] p-4">
          <div className="max-w-3xl mx-auto">
            <div className="relative flex items-end gap-2 bg-[#2d2d2d] rounded-2xl border border-[#3d3d3d] focus-within:border-[#dc2626] transition-colors p-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
                }}
                onKeyDown={handleKeyDown}
                placeholder="Message F1 Insight..."
                disabled={isSending}
                rows={1}
                className="flex-1 bg-transparent border-none outline-none resize-none text-[#e5e5e5] placeholder-[#8e8e93] text-sm py-2 px-2 max-h-[200px] overflow-y-auto"
                style={{ scrollbarWidth: "thin" }}
              />
              <button
                onClick={sendMessage}
                disabled={isSending || !input.trim()}
                className={`p-2 rounded-lg transition-all ${
                  input.trim() && !isSending
                    ? "bg-[#dc2626] hover:bg-[#b91c1c] text-white"
                    : "bg-[#2d2d2d] text-[#8e8e93] cursor-not-allowed"
                }`}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
            <p className="text-xs text-[#8e8e93] text-center mt-2">
              F1 Insight can make mistakes. Check important info.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
