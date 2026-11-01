"use client";

import { useRef, useState } from "react";

import { btn, btnSecondary, input, label } from "@/components/ui";
import { formatUsdExact } from "@/lib/money";
import { curlSnippet, MAX_TOKENS_CAP, parseSse, pythonSnippet, type ChatMessage, type StreamEvent } from "@/lib/playground";

export interface PlaygroundModel {
  id: string;
  label: string;
  maxContext: number;
  note: string | null;
}

type Usage = NonNullable<StreamEvent["usage"]>;

async function errorText(res: Response): Promise<string> {
  let message = `Request failed (HTTP ${res.status}).`;
  try {
    const body = (await res.json()) as { error?: { message?: string; code?: string } };
    if (body.error?.message) message = body.error.message;
    if (body.error?.code === "model_unavailable") {
      const retry = Number(res.headers.get("retry-after"));
      message = `This model is not running right now${retry ? `; try again in about ${Math.ceil(retry / 60)} min` : ""}. (${message})`;
    }
  } catch {
    // keep the generic message
  }
  return message;
}

export function Playground({ models, apiBase, gatewayDown }: { models: PlaygroundModel[]; apiBase: string; gatewayDown: boolean }) {
  const [model, setModel] = useState(models[0]?.id ?? "");
  const [system, setSystem] = useState("You are a helpful assistant.");
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1);
  const [maxTokens, setMaxTokens] = useState(512);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [showCode, setShowCode] = useState<"curl" | "python" | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const current = models.find((m) => m.id === model);
  const fullMessages = (extra: ChatMessage[]): ChatMessage[] => [
    ...(system.trim() ? [{ role: "system" as const, content: system }] : []),
    ...messages,
    ...extra,
  ];

  async function send() {
    const text = draft.trim();
    if (!text || streaming || !model) return;
    const userMsg: ChatMessage = { role: "user", content: text };
    const convo = fullMessages([userMsg]);
    setMessages((m) => [...m, userMsg, { role: "assistant", content: "" }]);
    setDraft("");
    setError(null);
    setUsage(null);
    setStreaming(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const append = (s: string) =>
      setMessages((m) => {
        const copy = m.slice();
        const last = copy[copy.length - 1];
        copy[copy.length - 1] = { ...last, content: last.content + s };
        return copy;
      });
    try {
      const res = await fetch("/api/playground/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, messages: convo, temperature, top_p: topP, max_tokens: maxTokens }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        setError(await errorText(res));
        setMessages((m) => m.slice(0, -1)); // drop the empty assistant turn
        return;
      }
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const { events, rest } = parseSse(buffer + value);
        buffer = rest;
        for (const ev of events) {
          if (ev.content) append(ev.content);
          if (ev.usage) setUsage(ev.usage);
          if (ev.error) setError(ev.error);
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError("The connection was lost.");
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  const snippetInput = {
    apiBase,
    model: model || "qwen3-8b",
    messages: fullMessages(draft.trim() ? [{ role: "user", content: draft.trim() }] : messages.length ? [] : [{ role: "user", content: "Hello!" }]),
    temperature,
    top_p: topP,
    max_tokens: maxTokens,
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[18rem_1fr]">
      <aside className="space-y-4">
        {gatewayDown ? (
          <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-900 ring-1 ring-amber-200">
            The Loom API is not reachable right now, so this list comes from the catalog and requests may fail.
          </p>
        ) : null}
        <div>
          <label htmlFor="model" className={label}>
            Model
          </label>
          <select id="model" value={model} onChange={(e) => setModel(e.target.value)} className={`${input} mt-1`}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          {current?.note ? <p className="mt-1 text-xs text-zinc-500">{current.note}</p> : null}
        </div>
        <div>
          <label htmlFor="system" className={label}>
            System prompt
          </label>
          <textarea id="system" rows={4} value={system} onChange={(e) => setSystem(e.target.value)} className={`${input} mt-1`} />
        </div>
        <Slider id="temperature" label="Temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} />
        <Slider id="top_p" label="Top P" value={topP} min={0.05} max={1} step={0.05} onChange={setTopP} />
        <div>
          <label htmlFor="max_tokens" className={label}>
            Max tokens
          </label>
          <input
            id="max_tokens"
            type="number"
            min={1}
            max={Math.min(MAX_TOKENS_CAP, current?.maxContext ?? MAX_TOKENS_CAP)}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Math.max(1, Math.min(MAX_TOKENS_CAP, Number(e.target.value) || 1)))}
            className={`${input} mt-1`}
          />
        </div>
        <div className="flex gap-2">
          <button type="button" className={btnSecondary} onClick={() => setShowCode(showCode ? null : "curl")}>
            {showCode ? "Hide code" : "View code"}
          </button>
          <button
            type="button"
            className={btnSecondary}
            disabled={streaming}
            onClick={() => {
              setMessages([]);
              setUsage(null);
              setError(null);
            }}
          >
            Clear
          </button>
        </div>
      </aside>

      <section className="flex min-h-[28rem] flex-col rounded-lg border border-zinc-200 bg-white">
        {showCode ? (
          <div className="border-b border-zinc-200 p-4">
            <div className="mb-2 flex gap-2 text-sm">
              {(["curl", "python"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setShowCode(t)}
                  className={`rounded px-2 py-1 ${showCode === t ? "bg-zinc-900 text-white" : "text-zinc-700 hover:bg-zinc-100"}`}
                >
                  {t === "curl" ? "curl" : "Python (OpenAI SDK)"}
                </button>
              ))}
            </div>
            <pre className="max-h-80 overflow-auto rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
              {showCode === "curl" ? curlSnippet(snippetInput) : pythonSnippet(snippetInput)}
            </pre>
            <p className="mt-2 text-xs text-zinc-500">Set LOOM_API_KEY to a key from the API keys page.</p>
          </div>
        ) : null}
        <div className="flex-1 space-y-4 overflow-y-auto p-4" aria-live="polite">
          {messages.length === 0 ? <p className="text-sm text-zinc-500">Send a message to start.</p> : null}
          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "ml-auto max-w-[85%]" : "max-w-[85%]"}>
              <div className="mb-1 text-xs text-zinc-500">{m.role === "user" ? "You" : model}</div>
              <div
                className={`whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${m.role === "user" ? "bg-zinc-900 text-white" : "bg-zinc-100 text-zinc-900"}`}
              >
                {m.content || (streaming && i === messages.length - 1 ? "…" : "")}
              </div>
            </div>
          ))}
          {error ? <p className="text-sm text-red-700">{error}</p> : null}
          {usage ? (
            <p className="text-xs text-zinc-500">
              {usage.prompt_tokens ?? 0} input + {usage.completion_tokens ?? 0} output tokens
              {typeof usage.cost_micros === "number" ? `, cost ${formatUsdExact(usage.cost_micros)}` : ""}
            </p>
          ) : null}
        </div>
        <form
          className="flex gap-2 border-t border-zinc-200 p-3"
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <textarea
            rows={2}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder="Message (Enter to send, Shift+Enter for a new line)"
            className={`${input} flex-1 resize-none`}
          />
          {streaming ? (
            <button type="button" className={btnSecondary} onClick={() => abortRef.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" className={btn} disabled={!draft.trim() || !model}>
              Send
            </button>
          )}
        </form>
      </section>
    </div>
  );
}

function Slider(props: { id: string; label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <div>
      <div className="flex justify-between">
        <label htmlFor={props.id} className={label}>
          {props.label}
        </label>
        <span className="text-sm tabular-nums text-zinc-600">{props.value.toFixed(2)}</span>
      </div>
      <input
        id={props.id}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
        className="mt-1 w-full"
      />
    </div>
  );
}
