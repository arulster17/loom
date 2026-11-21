"""No product name in user-facing text (M3 spec §6.11, A36).

The name is undecided: it comes from `LOOM_BRAND_NAME` at request time, or a sentence is
reworded without one. A scan of the string literals and JSX text of `apps/web/src`
(tests excluded), billing's `stripe_api.py` and `loom_platform/errors.py` finds no
`Loom` other than in `X-Loom-*` header names. Not user-facing, so not scanned: comments,
docstrings and identifiers (`LoomEnv`, `loom_sk_`, `LOOM_*`). The web half (the title
template, the sign-in heading, the headers with and without the name) is in
`apps/web/src/app/brand-name.test.ts`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WEB_SRC = ROOT / "apps" / "web" / "src"
PY_FILES = [
    ROOT / "services" / "billing" / "src" / "loom_billing" / "stripe_api.py",
    ROOT / "services" / "platform_common" / "src" / "loom_platform" / "errors.py",
]
NAME = re.compile(r"(?<![A-Za-z0-9_-])Loom(?![A-Za-z0-9_])")
HEADER = re.compile(r"x-loom-[a-z0-9-]+", re.I)


def without_header_names(text: str) -> str:
    return HEADER.sub("", text)


def strip_ts_comments(src: str) -> str:
    """Source with `//` and `/* */` comments blanked (JSX `{/* */}` included); strings,
    template literals and JSX text are kept, so a `//` inside a URL string stays.

    An apostrophe in JSX text (`Don't`) is not a string's start: a `'` right after a
    letter or digit cannot open a string in TypeScript, so it is skipped, and a `'`/`"`
    string ends at the end of its line in any case (only template literals span lines).
    A stray `'` after a space in JSX text (`'90s`) is not handled here; the web app's
    lint (`react/no-unescaped-entities`) refuses it there anyway."""
    out: list[str] = []
    i, n = 0, len(src)
    quote: str | None = None
    while i < n:
        ch = src[i]
        if quote:
            if ch == "\n" and quote != "`":
                quote = None
                out.append(ch)
                i += 1
                continue
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j == -1 else j + 2
            out.append(" ")
            continue
        if ch in '"`' or (ch == "'" and not (i and src[i - 1].isalnum())):
            quote = ch
        out.append(ch)
        i += 1
    return "".join(out)


def web_files() -> list[Path]:
    files = [
        p
        for p in sorted(WEB_SRC.rglob("*"))
        if p.suffix in (".ts", ".tsx")
        and not re.search(r"\.test\.tsx?$", p.name)
        and "__tests__" not in p.parts
    ]
    assert len(files) > 20, f"found only {len(files)} web source files under {WEB_SRC}"
    return files


def test_no_product_name_in_web_text() -> None:
    hits = []
    for path in web_files():
        code = without_header_names(strip_ts_comments(path.read_text(encoding="utf-8")))
        for m in NAME.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            text = code.splitlines()[line - 1].strip()
            hits.append(f"{path.relative_to(ROOT)}:{line}: {text[:120]}")
    assert not hits, "user-facing `Loom` in the web app (§6.11, A36):\n" + "\n".join(hits)


def docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def python_hits(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = docstring_nodes(tree)
    return [
        f"{path.relative_to(ROOT)}:{node.lineno}: {node.value[:120]!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
        and NAME.search(without_header_names(node.value))
    ]


def test_no_product_name_in_platform_errors() -> None:
    """`loom_platform/errors.py` (the deploy worker's, §6.11)."""
    hits = python_hits(PY_FILES[1])
    assert not hits, "user-facing `Loom` in loom_platform/errors.py:\n" + "\n".join(hits)


def test_no_product_name_in_billing_stripe_api() -> None:
    """`loom_billing/stripe_api.py` (the Stripe worker's, §6.11)."""
    hits = python_hits(PY_FILES[0])
    assert not hits, "user-facing `Loom` in loom_billing/stripe_api.py:\n" + "\n".join(hits)


def test_the_scan_sees_jsx_text_and_skips_comments_and_headers() -> None:
    """The scanner itself: JSX text and strings count, comments, identifiers and the
    X-Loom-* header names do not."""
    src = (
        "// Loom comment\n/* Loom block */\nconst LoomEnv = 1;\n"
        'const h = "X-Loom-Request-Id";\nconst u = "https://x.test//Loom";\n'
        "export const A = () => <h1>{/* Loom */}Welcome to Loom</h1>;\n"
        "export const B = () => <p>Don't panic</p>; // Loom note\n"
        'const v = "Loom";\n'
    )
    code = without_header_names(strip_ts_comments(src))
    found = [code.count("\n", 0, m.start()) + 1 for m in NAME.finditer(code)]
    assert found == [5, 6, 8], found


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
