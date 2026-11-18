"""Helpers for the B3 review tests (test_platform_m3_review_*.py): a Postgres TCP proxy
that records every statement a client sends, and a service subprocess whose output is
read as it runs.

`PgProxy` holds a port that refuses connections (bound, not listening: Postgres is
"unreachable") until `open()`, then forwards each connection to the real server. It
answers SSL/GSSAPI encryption requests with "no" itself, so the client's protocol
messages stay readable, and records the SQL of every simple query ('Q') and every
Parse ('P') per connection, with the connection's user, *before* forwarding it. That is
the seam for "no work query ran": everything a service asks Postgres to run passes
through the proxy and is recorded."""

from __future__ import annotations

import contextlib
import os
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field

import psycopg

_SSL_REQUEST = 80877103
_GSSENC_REQUEST = 80877104
_CANCEL_REQUEST = 80877102


@dataclass
class Session:
    user: str
    database: str
    statements: list[str] = field(default_factory=list)


def _shutdown(s: socket.socket) -> None:
    with contextlib.suppress(OSError):
        s.shutdown(socket.SHUT_RDWR)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class PgProxy:
    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        self.upstream = (upstream_host, upstream_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))  # bound, not listening: connections refused
        self.port: int = self._sock.getsockname()[1]
        self._lock = threading.Lock()
        self._sessions: list[Session] = []
        self._socks: list[socket.socket] = []
        self._closed = False

    # ------------------------------------------------------------------ control

    def open(self) -> None:
        """Postgres becomes reachable through the proxy."""
        self._sock.listen(64)
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            socks = [self._sock, *self._socks]
        for s in socks:
            _shutdown(s)
            s.close()

    def sessions(self) -> list[Session]:
        with self._lock:
            return [Session(s.user, s.database, list(s.statements)) for s in self._sessions]

    def statements(self) -> list[tuple[str, str]]:
        """(user, sql) of every statement sent, all connections."""
        return [(s.user, q) for s in self.sessions() for q in s.statements]

    # ------------------------------------------------------------------ forwarding

    def _track(self, s: socket.socket) -> None:
        with self._lock:
            self._socks.append(s)

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self._sock.accept()
            except OSError:
                return
            self._track(client)
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            while True:  # startup phase: untyped messages
                head = _recv_exact(client, 4)
                if head is None:
                    return
                (length,) = struct.unpack("!i", head)
                body = _recv_exact(client, length - 4)
                if body is None:
                    return
                (code,) = struct.unpack("!i", body[:4])
                if code in (_SSL_REQUEST, _GSSENC_REQUEST):
                    client.sendall(b"N")
                    continue
                upstream = socket.create_connection(self.upstream, timeout=10)
                upstream.settimeout(None)
                self._track(upstream)
                upstream.sendall(head + body)
                if code == _CANCEL_REQUEST:
                    return
                parts = body[4:].split(b"\0")
                params = dict(zip(parts[0::2], parts[1::2], strict=False))
                session = Session(
                    params.get(b"user", b"").decode(errors="replace"),
                    params.get(b"database", b"").decode(errors="replace"),
                )
                with self._lock:
                    self._sessions.append(session)
                break
            threading.Thread(target=self._pump, args=(upstream, client), daemon=True).start()
            while True:  # typed messages
                kind = _recv_exact(client, 1)
                if kind is None:
                    return
                head = _recv_exact(client, 4)
                if head is None:
                    return
                (length,) = struct.unpack("!i", head)
                body = _recv_exact(client, length - 4) if length > 4 else b""
                if body is None:
                    return
                sql: bytes | None = None
                if kind == b"Q":
                    sql = body.split(b"\0", 1)[0]
                elif kind == b"P":
                    sql = body.split(b"\0", 2)[1]
                if sql is not None:
                    with self._lock:
                        session.statements.append(sql.decode(errors="replace"))
                upstream.sendall(kind + head + body)
        except OSError:
            return
        finally:
            for s in (client, upstream):
                if s is not None:
                    _shutdown(s)
                    s.close()

    @staticmethod
    def _pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                _shutdown(s)


def proxied_url(proxy: PgProxy, *, dbname: str, user: str, password: str) -> str:
    q = urllib.parse.quote
    return f"postgresql://{q(user, safe='')}:{q(password, safe='')}@127.0.0.1:{proxy.port}/{dbname}"


# Keywords of statements (or clauses) that change something, take locks, or talk to
# other sessions, wherever they appear as a keyword (not inside a string literal, a
# quoted identifier or a comment): `WITH x AS (DELETE ...)`, `EXPLAIN ANALYZE INSERT`,
# `SELECT ... FOR UPDATE` / `FOR SHARE`, a second statement in one simple query.
_WORK_KEYWORDS = frozenset(
    "INSERT UPDATE DELETE MERGE TRUNCATE COPY CREATE ALTER DROP GRANT REVOKE LISTEN "
    "NOTIFY CALL DO LOCK SHARE VACUUM ANALYZE REFRESH CLUSTER REINDEX COMMENT SECURITY "
    "IMPORT REASSIGN EXECUTE".split()
)
# Built-in functions with an effect beyond reading: sequences, NOTIFY, advisory locks,
# large objects, other backends.
_WORK_FUNCTIONS = re.compile(
    r"^(?:nextval|setval|pg_notify|pg_(?:try_)?advisory_(?:xact_)?lock(?:_shared)?|lo_\w+|"
    r"pg_terminate_backend|pg_cancel_backend)$"
)
_TOKEN = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<line_comment>--[^\n]*)"
    r"|(?P<block_comment>/\*.*?\*/)"
    r"|(?P<dollar>\$(?P<tag>[A-Za-z_][A-Za-z_0-9]*)?\$)"
    r"|(?P<estring>[eE]'(?:[^'\\]|\\.|'')*')"
    r"|(?P<string>'(?:[^']|'')*')"
    r"|(?P<qident>\"(?:[^\"]|\"\")*\")"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9$]*)"
    r"|(?P<param>\$\d+)"
    r"|(?P<number>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"
    r"|(?P<cast>::)"
    r"|(?P<op>.)",
    re.DOTALL,
)


@dataclass(frozen=True)
class _Tok:
    kind: str  # ident (lower-cased), qident (as written), string (its text), op, other
    text: str


def sql_tokens(sql: str) -> list[_Tok]:
    """The statement's tokens, comments and whitespace dropped. Unquoted identifiers and
    keywords are lower-cased (as Postgres folds them); a quoted identifier keeps its case;
    a string literal (standard, E'' or dollar-quoted) is one `string` token."""
    out: list[_Tok] = []
    pos = 0
    while pos < len(sql):
        m = _TOKEN.match(sql, pos)
        if m is None:  # an unterminated literal or comment: the rest is one literal
            out.append(_Tok("string", sql[pos:]))
            break
        kind = m.lastgroup
        if kind == "tag":
            kind = "dollar"
        text = m.group()
        pos = m.end()
        if kind in ("space", "line_comment", "block_comment"):
            continue
        if kind == "dollar":
            end = sql.find(text, pos)
            if end < 0:
                out.append(_Tok("string", sql[pos:]))
                break
            out.append(_Tok("string", sql[pos:end]))
            pos = end + len(text)
        elif kind == "estring":
            out.append(_Tok("string", text[2:-1]))
        elif kind == "string":
            out.append(_Tok("string", text[1:-1].replace("''", "'")))
        elif kind == "qident":
            out.append(_Tok("qident", text[1:-1].replace('""', '"')))
        elif kind == "ident":
            out.append(_Tok("ident", text.lower()))
        elif kind in ("cast", "op"):
            out.append(_Tok("op", text))
        else:
            out.append(_Tok("other", text))
    return out


# The object-identifier types a literal is read as an object name by: `'orgs'::regclass`,
# `CAST('orgs' AS regclass)`, `regclass 'orgs'`.
_REG_TYPES = frozenset(
    "regclass regcollation regconfig regdictionary regnamespace regoper regoperator regproc "
    "regprocedure regrole regtype".split()
)
# Words a parenthesis follows that make it no function call: `x IN ('orgs')`,
# `VALUES ('orgs')`, `= ANY ('{orgs}')`, `ARRAY('orgs')`, a sub-expression after a
# keyword. Anything else directly before `(` is a function name.
_PAREN_KEYWORDS = frozenset(
    "in values any all some exists array row as over filter within using on and or not "
    "where select from then else when case by is like ilike similar between with returning "
    "into set join lateral distinct group order having limit offset union except intersect "
    "recursive materialized escape".split()
)


def _reg_type_at(toks: list[_Tok], j: int) -> bool:
    """toks[j:] starts with an object-identifier type name, maybe `pg_catalog.`-qualified."""
    if j < len(toks) and toks[j].kind == "ident" and toks[j].text in _REG_TYPES:
        return True
    return (
        j + 2 < len(toks)
        and toks[j] == _Tok("ident", "pg_catalog")
        and toks[j + 1] == _Tok("op", ".")
        and toks[j + 2].kind == "ident"
        and toks[j + 2].text in _REG_TYPES
    )


def _literal_reads_as_object(toks: list[_Tok], i: int) -> bool:
    """Whether the string literal toks[i] is read as an object's name: cast to a reg type
    (`'x'::regclass`, `CAST('x' AS regclass)`, `regclass 'x'`) or passed directly as a
    function's argument (`to_regclass('x')`, `has_table_privilege(u, 'x', 'INSERT')`,
    `pg_relation_size('x')`)."""
    prev = toks[i - 1] if i else None
    nxt = toks[i + 1] if i + 1 < len(toks) else None
    if nxt == _Tok("op", "::") and _reg_type_at(toks, i + 2):
        return True
    if prev is not None and prev.kind == "ident" and prev.text in _REG_TYPES:
        return True  # a typed literal: regclass 'x'
    if (
        prev == _Tok("op", "(")
        and i >= 2
        and toks[i - 2] == _Tok("ident", "cast")
        and nxt == _Tok("ident", "as")
        and _reg_type_at(toks, i + 2)
    ):
        return True
    if prev not in (_Tok("op", "("), _Tok("op", ",")) or nxt not in (
        _Tok("op", ")"),
        _Tok("op", ","),
    ):
        return False
    depth = 0
    for j in range(i - 1, -1, -1):  # the parenthesis the literal is directly inside
        if toks[j] == _Tok("op", ")"):
            depth += 1
        elif toks[j] == _Tok("op", "("):
            if depth == 0:
                before = toks[j - 1] if j else None
                return before is not None and (
                    before.kind == "qident"
                    or (before.kind == "ident" and before.text not in _PAREN_KEYWORDS)
                )
            depth -= 1
    return False


def _literal_names(text: str) -> list[tuple[str | None, str]]:
    """(schema, name) for the object a `'...'::regclass` style literal names."""
    base = text.split("(", 1)[0].strip()
    parts = [p.strip() for p in base.split(".")]
    names = [p[1:-1] if p.startswith('"') and p.endswith('"') else p.lower() for p in parts]
    if len(names) == 1:
        return [(None, names[0])]
    return [(names[-2], names[-1])]


def schema_object_names(conn: psycopg.Connection) -> list[str]:
    """Names of the schema's own tables, partitions, views, sequences and functions (not
    an extension's): a statement that refers to one of them is service work, not a
    catalog check."""
    rows = conn.execute(
        "SELECT c.relname::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f') "
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = 'pg_class'::regclass "
        "  AND d.objid = c.oid AND d.deptype = 'e') "
        "UNION SELECT p.proname::text FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = 'public' "
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = 'pg_proc'::regclass "
        "  AND d.objid = p.oid AND d.deptype = 'e')"
    ).fetchall()
    return sorted({r[0] for r in rows})


def work_classifier(names: list[str]) -> Callable[[str], bool]:
    """Whether a statement is service work against the schema (`names`: its objects,
    from `schema_object_names`). Work is a statement that

    - refers to one of the schema's objects: an identifier (unquoted, folded to lower
      case, or quoted, as written) that is one of `names`, unqualified or qualified by
      `public`; or a string literal naming one that is read as an object name: cast to
      a reg type (`'orgs'::regclass`, `CAST('orgs' AS regclass)`, `regclass 'orgs'`) or
      passed directly as a function's argument (`to_regclass('public.orgs')`,
      `has_table_privilege(current_user, 'orgs', 'INSERT')`, `pg_relation_size('orgs')`);
    - or has a keyword that writes, locks, listens or notifies (anywhere: CTEs,
      EXPLAIN ANALYZE, FOR UPDATE / FOR SHARE, a second statement in the same message);
    - or calls a built-in with an effect (nextval/setval, pg_notify, advisory locks...).

    A mere mention is not work: a schema name inside a string literal compared in a
    catalog query (`relname = 'orgs'`, `relname IN ('orgs')`, cast to a type that is no
    reg type), in a comment, as a column alias (`AS users`),
    or as a column of another relation or schema (`s.sessions`, `pg_catalog.x`).
    Catalog reads (pg_roles, pg_class, current_user, pg_has_role ...), SET/SHOW and
    transaction control are not work. `test_platform_m3_work_classifier.py` pins both
    sides."""
    # An unquoted identifier is folded to lower case before lookup (as `sql_tokens`
    # does), a quoted one is not: either way it names an object only if the folded or
    # quoted text is that object's name exactly.
    exact = set(names)

    def schema_name(tok: _Tok) -> bool:
        return tok.kind in ("ident", "qident") and tok.text in exact

    def is_work(sql: str) -> bool:
        toks = sql_tokens(sql)
        for i, tok in enumerate(toks):
            prev = toks[i - 1] if i else None
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            qualified = prev is not None and prev == _Tok("op", ".")
            qualifier = toks[i - 2] if qualified and i >= 2 else None
            if tok.kind == "ident" and not qualified and tok.text.upper() in _WORK_KEYWORDS:
                return True
            if tok.kind in ("ident", "qident"):
                called = nxt == _Tok("op", "(")
                in_catalog = qualifier is None or qualifier.text == "pg_catalog"
                if called and in_catalog and _WORK_FUNCTIONS.match(tok.text):
                    return True
                if qualified and (qualifier is None or qualifier.text != "public"):
                    continue  # a column of an alias, or another schema's object
                if prev == _Tok("ident", "as") and not called:
                    continue  # an output alias
                if schema_name(tok):
                    return True
            if tok.kind == "string" and _literal_reads_as_object(toks, i):
                for schema, name in _literal_names(tok.text):
                    if schema in (None, "public") and name in exact:
                        return True
        return False

    return is_work


class Service:
    """A service subprocess; stdout and stderr are read together, line by line."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._lock:
                self._lines.append(line)

    @property
    def output(self) -> str:
        with self._lock:
            return "".join(self._lines)

    @property
    def returncode(self) -> int | None:
        return self.proc.poll()

    def _lines_from(self, start: int) -> list[str]:
        with self._lock:
            return self._lines[start:]

    def wait_line(
        self, pattern: re.Pattern[str], timeout_s: float, *, after: int = 0
    ) -> tuple[int, str] | None:
        """The first output line at index `after` or later that matches `pattern`, as
        (index, line), once it appears within `timeout_s`; None when it does not (early
        when the process exits without one). Each line is read and matched once, as it
        arrives: the wait never re-reads the output it already checked."""
        deadline = time.monotonic() + timeout_s
        checked = after
        while True:
            exited = self.proc.poll() is not None
            if exited:
                self._reader.join(5)  # the lines written before the exit
            for line in self._lines_from(checked):
                if pattern.search(line):
                    return checked, line.rstrip("\n")
                checked += 1
            if exited or time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def wait_for(self, pattern: re.Pattern[str], timeout_s: float) -> bool:
        """Whether a line matching `pattern` appears within `timeout_s` (False early if
        the process exits without one)."""
        return self.wait_line(pattern, timeout_s) is not None

    def lines_matching(self, pattern: re.Pattern[str], *, after: int = 0) -> list[str]:
        """The output lines so far, from index `after` on, that match `pattern`."""
        return [line.rstrip("\n") for line in self._lines_from(after) if pattern.search(line)]

    def wait_exit(self, timeout_s: float) -> int | None:
        try:
            code = self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            return None
        self._reader.join(5)
        return code

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        self._reader.join(5)


def service_env(**overrides: str) -> dict[str, str]:
    """The caller's environment without any LOOM_* variable, plus `overrides`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOOM_")}
    env.update(overrides)
    return env
