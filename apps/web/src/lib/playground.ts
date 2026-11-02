// Playground request validation and the "View code" snippets. Pure: shared by the
// route handler (server) and the playground page (client).

import { z } from "zod";

export const MAX_TOKENS_CAP = 16_384; // LOOM_MAX_TOKENS_CAP default

export const chatMessage = z.object({
  role: z.enum(["system", "user", "assistant"]),
  content: z.string().max(100_000),
});

export const playgroundRequest = z.object({
  model: z.string().min(1).max(200).regex(/^[A-Za-z0-9._:/-]+$/),
  messages: z.array(chatMessage).min(1).max(200),
  temperature: z.number().min(0).max(2),
  top_p: z.number().gt(0).max(1),
  max_tokens: z.number().int().min(1).max(MAX_TOKENS_CAP),
});

export type PlaygroundRequest = z.infer<typeof playgroundRequest>;
export type ChatMessage = z.infer<typeof chatMessage>;

/** The exact body sent to the gateway: always streaming, always with a usage chunk. */
export function gatewayBody(req: PlaygroundRequest) {
  return {
    model: req.model,
    messages: req.messages,
    temperature: req.temperature,
    top_p: req.top_p,
    max_tokens: req.max_tokens,
    stream: true,
    stream_options: { include_usage: true },
  };
}

export interface SnippetInput {
  apiBase: string; // e.g. http://localhost:8080/v1
  model: string;
  messages: ChatMessage[];
  temperature: number;
  top_p: number;
  max_tokens: number;
}

function shellSingleQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

export function curlSnippet(i: SnippetInput): string {
  const body = JSON.stringify(
    {
      model: i.model,
      messages: i.messages,
      temperature: i.temperature,
      top_p: i.top_p,
      max_tokens: i.max_tokens,
      stream: true,
    },
    null,
    2,
  );
  return [
    `curl ${i.apiBase}/chat/completions \\`,
    `  -H "Authorization: Bearer $LOOM_API_KEY" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d ${shellSingleQuote(body)}`,
  ].join("\n");
}

/** JSON string literals are valid Python string literals. */
function py(s: string): string {
  return JSON.stringify(s);
}

export function pythonSnippet(i: SnippetInput): string {
  const msgs = i.messages.map((m) => `        {"role": ${py(m.role)}, "content": ${py(m.content)}},`).join("\n");
  return `import os
from openai import OpenAI

client = OpenAI(
    base_url=${py(i.apiBase)},
    api_key=os.environ["LOOM_API_KEY"],
)

stream = client.chat.completions.create(
    model=${py(i.model)},
    messages=[
${msgs}
    ],
    temperature=${i.temperature},
    top_p=${i.top_p},
    max_tokens=${i.max_tokens},
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()`;
}

export interface StreamEvent {
  content?: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number; cost_micros?: number };
  error?: string;
  done?: boolean;
}

/**
 * Splits an SSE buffer into complete events. Returns the parsed events and the
 * unconsumed remainder (a partial event still arriving).
 */
export function parseSse(buffer: string): { events: StreamEvent[]; rest: string } {
  const events: StreamEvent[] = [];
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = parts.pop() ?? "";
  for (const part of parts) {
    for (const line of part.split("\n")) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") {
        events.push({ done: true });
        continue;
      }
      try {
        const obj = JSON.parse(data) as {
          choices?: { delta?: { content?: string | null } }[];
          usage?: StreamEvent["usage"] | null;
          error?: { message?: string };
        };
        if (obj.error) events.push({ error: obj.error.message ?? "The stream failed." });
        const content = obj.choices?.[0]?.delta?.content;
        if (content) events.push({ content });
        if (obj.usage) events.push({ usage: obj.usage });
      } catch {
        // ignore a malformed line
      }
    }
  }
  return { events, rest };
}
