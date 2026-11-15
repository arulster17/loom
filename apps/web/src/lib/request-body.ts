// Reading a request body with a hard byte cap, so an oversized or endless upload is
// refused before it is buffered: a declared Content-Length above the cap is refused
// without reading, and a chunked body is read only until it passes the cap.

export type LimitedBody = { ok: true; text: string } | { ok: false };

export async function readBodyLimited(req: Request, maxBytes: number): Promise<LimitedBody> {
  const declared = req.headers.get("content-length");
  if (declared !== null) {
    const n = Number(declared);
    if (!Number.isFinite(n) || n > maxBytes) {
      await req.body?.cancel().catch(() => {});
      return { ok: false };
    }
  }
  if (!req.body) return { ok: true, text: "" };
  const reader = req.body.getReader();
  const decoder = new TextDecoder();
  let text = "";
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      await reader.cancel().catch(() => {});
      return { ok: false };
    }
    text += decoder.decode(value, { stream: true });
  }
  return { ok: true, text: text + decoder.decode() };
}
