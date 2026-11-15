"""The pod proxy: mTLS in front of the engine on a serving pod.

phase1-runpod-plan.md §4 and docs/phase1-contracts.md §9. One process per pod:

- accepts TCP only from allowlisted source addresses (when configured), checked before
  the TLS handshake, so anyone else never reaches the TLS stack;
- TLS 1.3 only, and the client must present a certificate signed by Loom's CA;
- every request needs `X-Loom-Pod-Token` (403 otherwise);
- forwards only `POST /v1/chat/completions`, `POST /v1/completions`, `POST /tokenize`,
  `GET /v1/models`, `GET /health`, `GET /metrics` (404 otherwise) to the engine on
  loopback, with only the `Content-Type` and `Accept` request headers;
- streams the engine's response body through byte for byte, each chunk written and
  flushed as it arrives;
- when the client goes away mid-response, closes the engine connection at once, so the
  engine aborts the request; when the engine goes away mid-response, drops the client
  connection without ending the body, so the client sees an incomplete response, never
  a clean end;
- logs method, path, status, bytes and time per request; never bodies.

HTTP/1.1 (h11) with keep-alive; ALPN offers only `http/1.1`. Python and two small
dependencies (h11, httpx) rather than a reverse-proxy binary: it runs in the test suite
and in the compose stack as is, and its tests are the behaviour any replacement (e.g. a
pinned Caddy on RunPod pods) must pass.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import ipaddress
import json
import logging
import os
import signal
import socket
import ssl
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h11
import httpx

log = logging.getLogger("loom_podproxy")

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

TOKEN_HEADER = b"x-loom-pod-token"
ROUTES: frozenset[tuple[bytes, bytes]] = frozenset(
    {
        (b"POST", b"/v1/chat/completions"),
        (b"POST", b"/v1/completions"),
        (b"POST", b"/tokenize"),  # the gateway fits an omitted chat max_tokens (§6)
        (b"GET", b"/v1/models"),
        (b"GET", b"/health"),
        (b"GET", b"/metrics"),
    }
)
REQUEST_HEADERS = frozenset({b"content-type", b"accept"})
RESPONSE_HEADERS = frozenset({b"content-type", b"content-encoding"})
READ_SIZE = 65536


@dataclass(frozen=True)
class ProxyConfig:
    token: str
    upstream: str = "http://127.0.0.1:8000"
    # Empty: any source address (TLS client auth still applies).
    allowed_networks: tuple[IPNetwork, ...] = ()
    max_body_bytes: int = 4 * 1024 * 1024
    handshake_timeout_s: float = 10.0
    # Keep-alive: close a connection that sends no new request for this long.
    idle_timeout_s: float = 75.0
    upstream_connect_timeout_s: float = 5.0
    # Longest gap between two chunks from the engine (the gateway's own request timeout
    # is the real bound; this only stops a hung engine from pinning connections forever).
    upstream_read_timeout_s: float = 660.0
    max_upstream_connections: int = 1024
    _token_bytes: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("the pod token must not be empty")
        object.__setattr__(self, "_token_bytes", self.token.encode())

    def ip_allowed(self, host: str) -> bool:
        if not self.allowed_networks:
            return True
        try:
            addr = ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            return False
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        return any(addr in net for net in self.allowed_networks)

    def token_ok(self, presented: bytes | None) -> bool:
        return presented is not None and hmac.compare_digest(presented, self._token_bytes)


def parse_networks(values: Sequence[str]) -> tuple[IPNetwork, ...]:
    """CIDRs or single addresses, comma- or argument-separated."""
    out: list[IPNetwork] = []
    for value in values:
        for part in value.split(","):
            if part.strip():
                out.append(ipaddress.ip_network(part.strip(), strict=False))
    return tuple(out)


def server_ssl_context(cert: Path, key: Path, client_ca: Path) -> ssl.SSLContext:
    """TLS 1.3 server context requiring a client certificate that chains to `client_ca`."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.verify_flags |= ssl.VERIFY_X509_STRICT
    ctx.load_cert_chain(cert, key)
    ctx.load_verify_locations(cafile=client_ca)
    ctx.set_alpn_protocols(["http/1.1"])
    return ctx


def _error_body(status: int, message: str, code: str) -> bytes:
    kind = "server_error" if status >= 500 else "invalid_request_error"
    if status == 403:
        kind = "permission_error"
    body = {"error": {"message": message, "type": kind, "param": None, "code": code}}
    return json.dumps(body).encode()


class _ClientGone(Exception):
    pass


class _UpstreamLost(Exception):
    pass


class PodProxy:
    def __init__(
        self,
        config: ProxyConfig,
        ssl_context: ssl.SSLContext,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.ssl_context = ssl_context
        self._http = httpx.AsyncClient(
            base_url=config.upstream,
            transport=transport,
            timeout=httpx.Timeout(
                connect=config.upstream_connect_timeout_s,
                read=config.upstream_read_timeout_s,
                write=30.0,
                pool=config.upstream_connect_timeout_s,
            ),
            limits=httpx.Limits(
                max_connections=config.max_upstream_connections,
                max_keepalive_connections=min(256, config.max_upstream_connections),
            ),
            trust_env=False,
        )
        self._sockets: list[socket.socket] = []
        self._tasks: set[asyncio.Task[None]] = set()
        self._accepting: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------ lifecycle

    async def start(self, host: str, port: int) -> int:
        """Listen on (host, port); returns the bound port (useful with port 0)."""
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(512)
        sock.setblocking(False)
        self._sockets.append(sock)
        self._accepting.append(asyncio.create_task(self._accept_loop(sock)))
        return int(sock.getsockname()[1])

    async def close(self) -> None:
        for task in self._accepting:
            task.cancel()
        for sock in self._sockets:
            sock.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._accepting, *self._tasks, return_exceptions=True)
        await self._http.aclose()

    async def _accept_loop(self, sock: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                conn, addr = await loop.sock_accept(sock)
            except OSError as e:  # EMFILE and friends: back off instead of spinning
                log.warning("accept failed: %s", e)
                await asyncio.sleep(0.1)
                continue
            peer = str(addr[0])
            if not self.config.ip_allowed(peer):
                log.warning("refused connection from %s: source address not allowed", peer)
                conn.close()
                continue
            task = asyncio.create_task(self._connection(conn, peer))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _connection(self, conn: socket.socket, peer: str) -> None:
        loop = asyncio.get_running_loop()
        conn.setblocking(False)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        reader = asyncio.StreamReader(limit=READ_SIZE)
        protocol = asyncio.StreamReaderProtocol(reader)
        try:
            transport, _ = await loop.connect_accepted_socket(
                lambda: protocol,
                conn,
                ssl=self.ssl_context,
                ssl_handshake_timeout=self.config.handshake_timeout_s,
            )
        except (ConnectionError, TimeoutError) as e:
            # Port probes and clients that hang up before or during the handshake.
            log.debug("connection from %s closed during the TLS handshake: %s", peer, _short(e))
            conn.close()
            return
        except (ssl.SSLError, OSError) as e:
            log.warning("TLS handshake with %s failed: %s", peer, _short(e))
            conn.close()
            return
        writer = asyncio.StreamWriter(transport, protocol, reader, loop)
        try:
            await self._serve(reader, writer, peer)
        except (ConnectionError, ssl.SSLError, _ClientGone, _UpstreamLost):
            transport.abort()
        except Exception:
            log.exception("connection from %s failed", peer)
            transport.abort()
        finally:
            if not transport.is_closing():
                writer.close()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(writer.wait_closed(), 5)

    # ------------------------------------------------------------ HTTP

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, peer: str
    ) -> None:
        conn = h11.Connection(h11.SERVER, max_incomplete_event_size=64 * 1024)
        while True:
            try:
                request = await asyncio.wait_for(
                    self._next_event(conn, reader), self.config.idle_timeout_s
                )
            except TimeoutError:
                return
            except h11.RemoteProtocolError:
                await self._send_simple(conn, writer, 400, "Malformed request.", "bad_request")
                return
            if isinstance(request, h11.ConnectionClosed):
                return
            if not isinstance(request, h11.Request):
                return
            pipelined = await self._handle(conn, reader, writer, request, peer)
            if conn.our_state is h11.MUST_CLOSE or conn.their_state is h11.MUST_CLOSE:
                return
            try:
                conn.start_next_cycle()
            except h11.LocalProtocolError:
                return
            if pipelined:
                conn.receive_data(bytes(pipelined))

    async def _next_event(self, conn: h11.Connection, reader: asyncio.StreamReader) -> object:
        while True:
            event = conn.next_event()
            if event is h11.NEED_DATA:
                conn.receive_data(await reader.read(READ_SIZE))
                continue
            return event

    async def _read_body(self, conn: h11.Connection, reader: asyncio.StreamReader) -> bytes | None:
        """The request body, or None when it exceeds `max_body_bytes`."""
        parts: list[bytes] = []
        size = 0
        while True:
            event = await self._next_event(conn, reader)
            if isinstance(event, h11.Data):
                size += len(event.data)
                if size > self.config.max_body_bytes:
                    return None
                parts.append(bytes(event.data))
            elif isinstance(event, h11.EndOfMessage):
                return b"".join(parts)
            else:
                raise _ClientGone

    async def _handle(
        self,
        conn: h11.Connection,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request: h11.Request,
        peer: str,
    ) -> bytearray | None:
        started = time.monotonic()
        method = bytes(request.method)
        path = bytes(request.target).split(b"?", 1)[0]
        headers = {bytes(k).lower(): bytes(v) for k, v in request.headers}
        status, sent, pipelined = await self._route(conn, reader, writer, method, path, headers)
        quiet = status == 200 and path in (b"/health", b"/metrics")  # probes, every few s
        log.log(
            logging.DEBUG if quiet else logging.INFO,
            "%s %s %s -> %d %dB %.1fms",
            peer,
            method.decode(),
            path.decode(errors="replace"),
            status,
            sent,
            (time.monotonic() - started) * 1000,
        )
        return pipelined

    async def _route(
        self,
        conn: h11.Connection,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        method: bytes,
        path: bytes,
        headers: dict[bytes, bytes],
    ) -> tuple[int, int, bytearray | None]:
        """(status, body bytes sent, pipelined bytes read from the client meanwhile)."""
        if not self.config.token_ok(headers.get(TOKEN_HEADER)):
            await self._send_simple(conn, writer, 403, "Pod token missing or wrong.", "forbidden")
            return 403, 0, None
        if (method, path) not in ROUTES:
            await self._send_simple(conn, writer, 404, "Not found.", "not_found")
            return 404, 0, None
        try:
            body = await asyncio.wait_for(self._read_body(conn, reader), self.config.idle_timeout_s)
        except TimeoutError:
            raise _ClientGone from None
        if body is None:
            await self._send_simple(conn, writer, 413, "Request body too large.", "too_large")
            return 413, 0, None
        upstream_headers = {k.decode(): v.decode("latin-1") for k, v in headers.items()}
        upstream_headers = {
            k: v for k, v in upstream_headers.items() if k.encode() in REQUEST_HEADERS
        }
        upstream_headers["accept-encoding"] = "identity"
        upstream = self._http.build_request(
            method.decode(), path.decode(), content=body or None, headers=upstream_headers
        )
        try:
            response = await self._http.send(upstream, stream=True)
        except httpx.HTTPError as e:
            log.warning("engine unreachable: %s", _short(e))
            status = 503 if path == b"/health" else 502
            await self._send_simple(
                conn, writer, status, "Engine unreachable.", "upstream_unavailable"
            )
            return status, 0, None
        try:
            return await self._relay(conn, reader, writer, response)
        finally:
            await response.aclose()

    async def _relay(
        self,
        conn: h11.Connection,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        response: httpx.Response,
    ) -> tuple[int, int, bytearray | None]:
        out_headers = [(k, v) for k, v in response.headers.raw if k.lower() in RESPONSE_HEADERS]
        if "content-length" in response.headers and "transfer-encoding" not in response.headers:
            out_headers.append((b"content-length", response.headers["content-length"].encode()))
        writer.write(conn.send(h11.Response(status_code=response.status_code, headers=out_headers)))
        await writer.drain()

        sent = 0

        async def pump() -> None:
            nonlocal sent
            try:
                async for chunk in response.aiter_raw():
                    if chunk:
                        writer.write(conn.send(h11.Data(data=chunk)))
                        await writer.drain()
                        sent += len(chunk)
            except httpx.HTTPError as e:
                # Never end the body cleanly: a chunked terminator would tell the client
                # the stream completed.
                log.warning("engine lost mid-response: %s", _short(e))
                raise _UpstreamLost from e
            writer.write(conn.send(h11.EndOfMessage()))
            await writer.drain()

        pipelined = bytearray()

        async def watch() -> None:
            """Returns when the client closes its side (EOF or error) mid-response."""
            while True:
                try:
                    data = await reader.read(READ_SIZE)
                except Exception:
                    return
                if not data:
                    return
                pipelined.extend(data)

        pump_task = asyncio.create_task(pump())
        watch_task = asyncio.create_task(watch())
        try:
            done, _ = await asyncio.wait(
                {pump_task, watch_task}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            pump_task.cancel()
            watch_task.cancel()
            raise
        if pump_task not in done:
            # The client went away: stop reading from the engine and close that connection
            # (response.aclose() in the caller), so the engine aborts the request.
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task
            log.info("client closed mid-response after %dB; engine request cancelled", sent)
            raise _ClientGone
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task
        pump_task.result()  # re-raise _UpstreamLost / ConnectionError
        return response.status_code, sent, (pipelined or None)

    async def _send_simple(
        self, conn: h11.Connection, writer: asyncio.StreamWriter, status: int, msg: str, code: str
    ) -> None:
        body = _error_body(status, msg, code)
        # Refusals and failures close the connection: the request body may be unread, and a
        # refused client has no reason to keep a connection.
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", b"%d" % len(body)),
            (b"connection", b"close"),
        ]
        try:
            writer.write(conn.send(h11.Response(status_code=status, headers=headers)))
            writer.write(conn.send(h11.Data(data=body)))
            writer.write(conn.send(h11.EndOfMessage()))
            await writer.drain()
        except h11.LocalProtocolError:
            raise _ClientGone from None


def _short(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:300]


# ---------------------------------------------------------------- CLI


def _read_token(token_file: Path | None) -> str:
    if token_file is not None:
        return token_file.read_text().strip()
    token = os.environ.get("LOOM_POD_TOKEN", "").strip()
    if not token:
        raise SystemExit("give --token-file or set LOOM_POD_TOKEN")
    return token


def build_parser() -> argparse.ArgumentParser:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="loom-podproxy", description=__doc__.splitlines()[0])
    p.add_argument("--host", default=env("LOOM_PODPROXY_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(env("LOOM_PODPROXY_PORT", "8443")))
    p.add_argument("--upstream", default=env("LOOM_PODPROXY_UPSTREAM", "http://127.0.0.1:8000"))
    p.add_argument("--cert", type=Path, default=env("LOOM_PODPROXY_CERT"), required=False)
    p.add_argument("--key", type=Path, default=env("LOOM_PODPROXY_KEY"), required=False)
    p.add_argument(
        "--client-ca",
        type=Path,
        default=env("LOOM_PODPROXY_CLIENT_CA"),
        help="CA certificate client certificates must chain to (Loom's CA)",
    )
    p.add_argument(
        "--token-file",
        type=Path,
        default=env("LOOM_PODPROXY_TOKEN_FILE"),
        help="file holding the pod token (else $LOOM_POD_TOKEN)",
    )
    p.add_argument(
        "--allow-ip",
        action="append",
        default=[v for v in [env("LOOM_PODPROXY_ALLOW_IPS")] if v],
        metavar="CIDR",
        help="source addresses allowed to connect (repeatable or comma-separated; default any)",
    )
    p.add_argument("--log-level", default=env("LOOM_PODPROXY_LOG_LEVEL", "info"))
    return p


async def serve(args: argparse.Namespace) -> None:
    for name in ("cert", "key", "client_ca"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required")
    config = ProxyConfig(
        token=_read_token(args.token_file),
        upstream=args.upstream,
        allowed_networks=parse_networks(args.allow_ip),
    )
    proxy = PodProxy(config, server_ssl_context(args.cert, args.key, args.client_ca))
    port = await proxy.start(args.host, args.port)
    log.info(
        "listening on %s:%d -> %s (allowed sources: %s)",
        args.host,
        port,
        config.upstream,
        ", ".join(map(str, config.allowed_networks)) or "any",
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        await proxy.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
