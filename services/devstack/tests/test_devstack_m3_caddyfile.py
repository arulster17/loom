"""`deploy/Caddyfile`: the public routes (M3 spec §6.7, §8.5; A37's static half).

Read statically; A15 checks the behaviour through a running Caddy. The file is parsed
the way Caddy reads it (whitespace-separated tokens, quotes, a block opened by a `{` at
the end of a line and closed by a `}` on its own line; `{$LOOM_DOMAIN}` and other
placeholders are tokens, not blocks), snippets `(name) { ... }` are expanded where a site
(or another snippet) `import`s them, site addresses are compared without `https://` and
`:443` (`http://` and `:80` sites are the redirects), and only what Caddy itself requires
is pinned:

- HSTS (`Strict-Transport-Security` with `max-age=31536000`) on all three sites;
- request body limits (`max_size`): `api.` 1 MiB + 1 KiB, `hooks.` 256 KiB (sizes in any
  unit Caddy accepts: bytes, KB/KiB, MB/MiB, ...);
- a rate-limit directive (`rate_limit`, the pinned module's) on `api.`;
- the apex proxies to `web:3000`;
- `hooks.` proxies to `billing:8090` only under a matcher that selects `/stripe/webhook`
  or `/stripe/health` (a path on the directive, on an enclosing `handle`/`route`, or a
  named matcher whose `path` values are those or whose `path_regexp` selects only those
  two paths, probed against a list of other paths), with `POST` and `GET` method matchers, and
  answers 404 otherwise (an unconditional `reverse_proxy billing:8090` fails).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CADDYFILE = ROOT / "deploy" / "Caddyfile"
STRIPE_PATHS = {"/stripe/webhook", "/stripe/health"}


class Node:
    def __init__(self, args: list[str], parent: Node | None) -> None:
        self.args = args
        self.parent = parent
        self.children: list[Node] = []

    def walk(self) -> list[Node]:
        out: list[Node] = []
        for c in self.children:
            out.append(c)
            out += c.walk()
        return out

    def tokens(self) -> list[str]:
        return self.args + [t for c in self.walk() for t in c.args]


def lines_of_tokens(text: str) -> list[list[str]]:
    """Caddy's lexer, simplified: tokens per line; `#` at a token's start comments out
    the rest of the line; "..." and `...` are one token."""
    out: list[list[str]] = []
    cur: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            if cur:
                out.append(cur)
            cur = []
            i += 1
        elif ch.isspace():
            i += 1
        elif ch == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif ch in '"`':
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" and ch == '"' else 1
            cur.append(text[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace():
                j += 1
            cur.append(text[i:j])
            i = j
    if cur:
        out.append(cur)
    return out


def parse(text: str) -> Node:
    root = Node([], None)
    stack = [root]
    for toks in lines_of_tokens(text):
        if toks == ["}"]:
            assert len(stack) > 1, "the Caddyfile closes a block it never opened"
            stack.pop()
        elif toks[-1] == "{":
            node = Node(toks[:-1], stack[-1])
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].children.append(Node(toks, stack[-1]))
    assert len(stack) == 1, "the Caddyfile leaves a block open"
    return root


def expand(node: Node, snippets: dict[str, Node], depth: int = 0) -> None:
    """Replace `import <snippet>` lines with the snippet's lines (recursively: also an
    `import` at a snippet's own top level, and inside nested blocks)."""
    assert depth < 20, "snippet imports nest too deep"

    def inline(c: Node, level: int) -> list[Node]:
        assert level < 10, "snippet imports nest too deep"
        if c.args[:1] == ["import"] and len(c.args) > 1 and c.args[1] in snippets:
            body = [clone(s, node) for s in snippets[c.args[1]].children]
            return [n for s in body for n in inline(s, level + 1)]
        return [c]

    node.children = [n for c in node.children for n in inline(c, 0)]
    for c in node.children:
        expand(c, snippets, depth + 1)


def clone(node: Node, parent: Node) -> Node:
    copy = Node(list(node.args), parent)
    copy.children = [clone(c, copy) for c in node.children]
    return copy


def sites() -> dict[str, Node]:
    """HTTPS site address (without `https://` and `:443`) -> its block, snippets expanded;
    `http://` and `:80` sites (redirects) are not the three sites."""
    assert CADDYFILE.is_file(), "deploy/Caddyfile does not exist"
    root = parse(CADDYFILE.read_text())
    snippets = {
        c.args[0][1:-1]: c
        for c in root.children
        if c.args and c.args[0].startswith("(") and c.args[0].endswith(")")
    }
    out: dict[str, Node] = {}
    for c in root.children:
        if not c.args or c.args[0].startswith("("):
            continue  # global options, snippets
        expand(c, snippets)
        for addr in " ".join(c.args).replace(",", " ").split():
            if addr.startswith("http://") or addr.endswith(":80"):
                continue
            out[addr.removeprefix("https://").removesuffix(":443")] = c
    return out


def site(prefix: str) -> Node:
    want = f"{prefix}{{$LOOM_DOMAIN}}"
    found = sites()
    assert want in found, f"no site block for {want}: {sorted(found)}"
    return found[want]


SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "ki": 1024,
    "kib": 1024,
    "m": 1000**2,
    "mb": 1000**2,
    "mi": 1024**2,
    "mib": 1024**2,
    "g": 1000**3,
    "gb": 1000**3,
    "gi": 1024**3,
    "gib": 1024**3,
}


def size_bytes(text: str) -> int:
    """A size as Caddy parses it (go-humanize ParseBytes)."""
    m = re.fullmatch(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*([A-Za-z]*)", text.strip())
    assert m and m.group(2).lower() in SIZE_UNITS, f"not a size: {text!r}"
    return int(float(m.group(1).replace(",", "")) * SIZE_UNITS[m.group(2).lower()])


# ---------------------------------------------------------------- HSTS


@pytest.mark.parametrize("prefix", ["", "api.", "hooks."])
def test_hsts_for_a_year_on_every_site(prefix: str) -> None:
    hsts = [
        n
        for n in site(prefix).walk()
        if any(a.lower().lstrip("+>?") == "strict-transport-security" for a in n.args)
        and any("max-age=31536000" in a for a in n.args)
    ]
    assert hsts, f"{prefix}{{$LOOM_DOMAIN}} sets no Strict-Transport-Security max-age=31536000"


# ---------------------------------------------------------------- body limits


@pytest.mark.parametrize(
    ("prefix", "limit"), [("api.", 1024 * 1024 + 1024), ("hooks.", 256 * 1024)]
)
def test_request_body_limit(prefix: str, limit: int) -> None:
    sizes = [size_bytes(n.args[1]) for n in site(prefix).walk() if n.args[:1] == ["max_size"]]
    assert sizes, f"{prefix}{{$LOOM_DOMAIN}} has no request body limit (max_size)"
    assert limit in sizes and max(sizes) <= limit, (
        f"{prefix}{{$LOOM_DOMAIN}} body limits {sizes}, want {limit} bytes"
    )


# ---------------------------------------------------------------- api.


def test_api_is_rate_limited_per_client() -> None:
    found = [n for n in site("api.").walk() if n.args[:1] in (["rate_limit"], ["ratelimit"])]
    assert found, "api.{$LOOM_DOMAIN} has no rate-limit directive (§15 D2)"


def test_api_answers_404_outside_its_routes() -> None:
    assert not_found(site("api.")), "api.{$LOOM_DOMAIN} has no 404 for other paths"


# ---------------------------------------------------------------- apex


def proxies(node: Node) -> list[Node]:
    return [n for n in node.walk() if n.args[:1] == ["reverse_proxy"]]


def upstreams(proxy: Node) -> list[str]:
    toks = proxy.args[1:] + [t for c in proxy.walk() if c.args[:1] == ["to"] for t in c.args[1:]]
    return [re.sub(r"^\w+://", "", t) for t in toks if not t.startswith(("@", "/", "*"))]


def test_the_apex_proxies_to_the_web_app() -> None:
    ps = proxies(site(""))
    assert ps, "{$LOOM_DOMAIN} proxies nowhere"
    ups = [u for p in ps for u in upstreams(p)]
    assert "web:3000" in ups, f"{{$LOOM_DOMAIN}} does not proxy to web:3000: {ups}"
    assert set(ups) == {"web:3000"}, f"{{$LOOM_DOMAIN}} proxies elsewhere too: {ups}"


# ---------------------------------------------------------------- hooks.


def not_found(node: Node) -> bool:
    return any(n.args[:1] in (["respond"], ["error"]) and "404" in n.args for n in node.walk())


def matcher_lines(scope: Node, name: str) -> list[list[str]]:
    """The matcher lines (`<type> <args...>`) of every definition of the named matcher
    `name` in `scope`: the single-line form `@name <type> <args...>` and the block form."""
    out: list[list[str]] = []
    for n in scope.walk():
        if n.args[:1] == [name]:
            if len(n.args) > 1:
                out.append(n.args[1:])
            out += [c.args for c in n.walk() if c.args]
    return out


# Paths a matcher restricted to the two Stripe routes must not select.
OTHER_PATHS = (
    "/",
    "/health",
    "/metrics",
    "/internal/credits",
    "/internal/stripe/webhook",
    "/stripe",
    "/stripe/other",
    "/stripe/webhooks",
    "/stripe/healthz",
    "/stripe/webhook/x",
    "/stripe/health/x",
    "/x/stripe/webhook",
    "/v1/chat/completions",
)


def regexp_paths(pattern: str) -> set[str] | None:
    """The Stripe paths a `path_regexp` selects (Caddy: unanchored RE2 search on the
    path), or None if it also selects another path (or does not compile here)."""
    try:
        rx = re.compile(pattern)
    except re.error:
        return None
    if any(rx.search(p) for p in OTHER_PATHS):
        return None
    return {p for p in STRIPE_PATHS if rx.search(p)} or None


def stripe_paths(matcher: str, scope: Node) -> set[str] | None:
    """The Stripe paths a matcher token selects, or None if it selects anything else:
    exact `path` matchers, or a `path_regexp` that matches only those paths (other
    matcher types, e.g. `method` or `header`, only narrow it further)."""
    if matcher.startswith("/"):
        return {matcher} if matcher in STRIPE_PATHS else None
    if not matcher.startswith("@"):
        return None
    paths: set[str] = set()
    for kind, *args in matcher_lines(scope, matcher):
        if kind == "not":
            return None
        if kind == "path":
            if not args or not set(args) <= STRIPE_PATHS:
                return None
            paths |= set(args)
        elif kind == "path_regexp":
            sel = regexp_paths(args[-1]) if args else None
            if sel is None:
                return None
            paths |= sel
        elif kind == "expression":
            found = set(re.findall(r"/[\w/*.{}-]*", " ".join(args)))
            if not found or not found <= STRIPE_PATHS:
                return None
            paths |= found
    return paths or None


def guard(proxy: Node, scope: Node) -> set[str] | None:
    """The Stripe paths that reach `proxy`: from a matcher on it or on an enclosing
    `handle`/`handle_path`/`route`; None when nothing restricts it to them."""
    chain: list[str] = []
    if len(proxy.args) > 1 and proxy.args[1].startswith(("@", "/")):
        chain.append(proxy.args[1])
    node = proxy.parent
    while node is not None and node is not scope:
        if node.args[:1] in (["handle"], ["handle_path"], ["route"]) and len(node.args) > 1:
            chain.append(node.args[1])
        node = node.parent
    selected = [p for m in chain if (p := stripe_paths(m, scope)) is not None]
    return set.intersection(*selected) if selected else None


def test_hooks_proxies_to_billing_only_the_two_stripe_routes() -> None:
    hooks = site("hooks.")
    to_billing = [p for p in proxies(hooks) if "billing:8090" in upstreams(p)]
    assert to_billing, "hooks.{$LOOM_DOMAIN} does not proxy to billing:8090"
    reached: set[str] = set()
    for p in to_billing:
        paths = guard(p, hooks)
        assert paths, (
            f"`{' '.join(p.args)}` sends more than /stripe/webhook and /stripe/health to "
            "billing (an unrestricted route reaches /internal/* and /health)"
        )
        reached |= paths
    assert reached == STRIPE_PATHS, f"hooks. routes {sorted(reached)} to billing"
    for p in proxies(hooks):
        assert set(upstreams(p)) <= {"billing:8090"}, f"hooks. proxies elsewhere: {p.args}"


def test_hooks_restricts_the_methods() -> None:
    text = " ".join(site("hooks.").tokens())
    assert re.search(r"method\W*[^@]*?\bPOST\b", text), "hooks. matches no POST method"
    assert re.search(r"method\W*[^@]*?\bGET\b", text), "hooks. matches no GET method"


def test_hooks_answers_404_otherwise() -> None:
    assert not_found(site("hooks.")), "hooks.{$LOOM_DOMAIN} has no 404 for other requests"


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
