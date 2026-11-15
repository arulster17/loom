import { describe, expect, it } from "vitest";

import { curlSnippet, gatewayBody, parseSse, playgroundRequest, pythonSnippet } from "./playground";

const valid = {
  model: "qwen3-8b",
  messages: [
    { role: "system", content: "Be brief." },
    { role: "user", content: "Hi" },
  ],
  temperature: 0.7,
  top_p: 1,
  max_tokens: 256,
};

describe("playground request", () => {
  it("accepts a normal chat and always streams with usage", () => {
    const req = playgroundRequest.parse(valid);
    expect(gatewayBody(req)).toEqual({ ...valid, stream: true, stream_options: { include_usage: true } });
  });

  it("refuses out-of-range settings and extra roles", () => {
    for (const bad of [
      { ...valid, temperature: 3 },
      { ...valid, top_p: 0 },
      { ...valid, max_tokens: 16_385 },
      { ...valid, max_tokens: 1.5 },
      { ...valid, messages: [] },
      { ...valid, messages: [{ role: "tool", content: "x" }] },
      { ...valid, model: "a b" },
    ]) {
      expect(playgroundRequest.safeParse(bad).success).toBe(false);
    }
  });

  it("drops fields the client should not control", () => {
    const req = playgroundRequest.parse({ ...valid, user: "x", n: 5, stream: false });
    expect(gatewayBody(req)).not.toHaveProperty("n");
    expect(gatewayBody(req).stream).toBe(true);
  });
});

describe("SSE parsing", () => {
  it("handles chunks split across reads, usage, errors and DONE", () => {
    const first = parseSse('data: {"choices":[{"delta":{"content":"Hel"}}]}\n\ndata: {"choices":[{"delta":{"con');
    expect(first.events).toEqual([{ content: "Hel" }]);
    const second = parseSse(
      first.rest +
        'tent":"lo"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"cost_micros":1}}\n\n' +
        'data: {"error":{"message":"upstream lost","code":"upstream_lost"}}\n\ndata: [DONE]\n\n',
    );
    expect(second.events).toEqual([
      { content: "lo" },
      { usage: { prompt_tokens: 3, completion_tokens: 2, cost_micros: 1 } },
      { error: "upstream lost" },
      { done: true },
    ]);
    expect(second.rest).toBe("");
  });

  it("accepts CRLF framing and ignores comments and junk", () => {
    const r = parseSse(': ping\r\n\r\ndata: not json\r\n\r\ndata: {"choices":[{"delta":{"content":"x"}}]}\r\n\r\n');
    expect(r.events).toEqual([{ content: "x" }]);
  });
});

describe("View code snippets", () => {
  const input = {
    apiBase: "http://localhost:8080/v1",
    model: "qwen3-8b",
    messages: [{ role: "user" as const, content: "It's \"quoted\"\nand multi-line" }],
    temperature: 0.5,
    top_p: 0.9,
    max_tokens: 100,
  };

  it("curl quotes the body safely for a POSIX shell and never embeds a key", () => {
    const s = curlSnippet(input);
    expect(s).toContain("$LOOM_API_KEY");
    expect(s).toContain("http://localhost:8080/v1/chat/completions");
    expect(s).toContain(`'\\''`); // the apostrophe in It's
    expect(s).not.toMatch(/loom_sk_/);
  });

  it("Python uses the OpenAI SDK with the Loom base URL", () => {
    const s = pythonSnippet(input);
    expect(s).toContain('base_url="http://localhost:8080/v1"');
    expect(s).toContain('os.environ["LOOM_API_KEY"]');
    expect(s).toContain('"content": "It\'s \\"quoted\\"\\nand multi-line"');
    expect(s).toContain("temperature=0.5");
  });
});
