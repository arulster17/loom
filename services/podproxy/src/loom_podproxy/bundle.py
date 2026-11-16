"""`loom-podproxy-bundle`: the pinned pod proxy bundle (docs/phase1-m2-spec.md §9).

Builds one tarball holding the `loom-podproxy` wheel and the wheels of its dependencies as
`uv.lock` pins them (h11, httpx and what they need), and records the tarball's sha256 in
`config/serving-artifacts.lock`. Bootstrap ships the tarball to a serving pod, checks it
against that sha256 and installs it with `pip install --no-index --find-links` into a
venv of the image's own Python; nothing comes from the engine image's site-packages.

Reproducible by construction, so the committed lock names what any checkout builds:

- every input is a committed git blob from `git archive HEAD` (with `core.autocrlf` off,
  so a CRLF Windows checkout and a Linux one read the same bytes); uncommitted changes
  are not in the bundle;
- the `loom-podproxy` wheel is written here, from those blobs and its `pyproject.toml`,
  with fixed timestamps, modes and order and no compression, rather than by a build
  backend whose version and zlib would leak into the bytes;
- the dependency wheels are the exact files `uv.lock` names (url and sha256), for the
  pod's target: CPython 3.12 on x86_64 Linux;
- the tarball is an uncompressed ustar archive with fixed metadata and sorted members.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from base64 import urlsafe_b64encode
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PACKAGE = "loom-podproxy"
PACKAGE_DIR = "services/podproxy"  # in the repository
LOCK_DEFAULT = "config/serving-artifacts.lock"  # relative to the repository root
LOCK_TABLE = "podproxy-bundle"
BUNDLE_DIR = "loom-podproxy-bundle"  # the one directory inside the tarball
GENERATOR = "loom-podproxy-bundle"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

# The serving pod's interpreter (vLLM images ship CPython 3.12 on x86_64 Linux), for
# `uv.lock` markers and wheel tags.
TARGET_PYTHON = "3.12"
TARGET_ENV: Mapping[str, str] = {
    "python_version": "3.12",
    "python_full_version": "3.12.3",
    "implementation_name": "cpython",
    "implementation_version": "3.12.3",
    "platform_python_implementation": "CPython",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "os_name": "posix",
    "extra": "",
}
_PY_TAGS = frozenset({"py3", "py312", "cp312"})
_ABI_TAGS = frozenset({"none", "abi3", "cp312"})
_LINUX_X86_64 = re.compile(r"^(manylinux\d*_\d+_\d+|manylinux\d+|musllinux_\d+_\d+)_x86_64$")


class BundleError(RuntimeError):
    pass


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- git


def git(repo: Path, *args: str, blobs: bool = True) -> bytes:
    """`blobs`: read files exactly as committed, with no line-ending conversion."""
    raw = ["-c", "core.autocrlf=false", "-c", "core.eol=lf"] if blobs else []
    done = subprocess.run(["git", *raw, *args], cwd=repo, capture_output=True)
    if done.returncode != 0:
        detail = done.stderr.decode(errors="replace").strip()
        raise BundleError(f"git {' '.join(args)} failed: {detail}")
    return done.stdout


def repo_root(start: Path) -> Path:
    return Path(git(start, "rev-parse", "--show-toplevel").decode().strip())


def committed_files(repo: Path, *paths: str) -> dict[str, bytes]:
    """{path: blob} for every regular file under `paths` at HEAD, via `git archive`."""
    raw = git(repo, "archive", "--format=tar", "HEAD", "--", *paths)
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                assert f is not None
                out[member.name] = f.read()
    return out


def warn_if_dirty(repo: Path) -> None:
    args = ("--no-optional-locks", "status", "--porcelain", "--", PACKAGE_DIR, "uv.lock")
    status = git(repo, *args, blobs=False).decode().strip()
    if status:
        print(
            "note: uncommitted changes are not in the bundle (built from HEAD):\n" + status,
            file=sys.stderr,
        )


# ---------------------------------------------------------------- markers (PEP 508)

_TOKEN = re.compile(
    r"\s*(?:(?P<str>'[^']*'|\"[^\"]*\")|(?P<op>===|==|!=|<=|>=|~=|<|>|\(|\))"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_.]*))"
)
_VERSION_VARS = frozenset({"python_version", "python_full_version", "implementation_version"})


def _tokens(marker: str) -> list[str]:
    out: list[str] = []
    pos = 0
    marker = marker.strip()
    while pos < len(marker):
        m = _TOKEN.match(marker, pos)
        if m is None or m.end() == pos:
            raise BundleError(f"cannot parse marker {marker!r} at {pos}")
        out.append(m.group(m.lastgroup or "word"))
        pos = m.end()
    return out


def _version(text: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(\.\d+)*", text):
        raise BundleError(f"unsupported version in a marker: {text!r}")
    return tuple(int(p) for p in text.split("."))


def _compare_versions(left: str, op: str, right: str) -> bool:
    if op in ("==", "!=") and right.endswith(".*"):
        prefix = _version(right[:-2])
        equal = _version(left)[: len(prefix)] == prefix
        return equal if op == "==" else not equal
    a, b = _version(left), _version(right)
    width = max(len(a), len(b))
    a, b = a + (0,) * (width - len(a)), b + (0,) * (width - len(b))
    if op == "~=":
        if len(_version(right)) < 2:
            raise BundleError(f"~= needs two version parts: {right!r}")
        stem = _version(right)[:-1]
        return a >= b and a[: len(stem)] == stem
    ops = {
        "==": a == b,
        "===": left == right,
        "!=": a != b,
        "<": a < b,
        "<=": a <= b,
        ">": a > b,
        ">=": a >= b,
    }
    if op not in ops:
        raise BundleError(f"unsupported version operator {op!r}")
    return ops[op]


class _MarkerParser:
    def __init__(self, marker: str, env: Mapping[str, str]) -> None:
        self.marker = marker
        self.tokens = _tokens(marker)
        self.pos = 0
        self.env = env

    def parse(self) -> bool:
        value = self._or()
        if self.pos != len(self.tokens):
            raise BundleError(f"trailing tokens in marker {self.marker!r}")
        return value

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise BundleError(f"marker {self.marker!r} ends early")
        self.pos += 1
        return token

    def _or(self) -> bool:
        value = self._and()
        while self._peek() == "or":
            self._take()
            rhs = self._and()
            value = value or rhs
        return value

    def _and(self) -> bool:
        value = self._atom()
        while self._peek() == "and":
            self._take()
            rhs = self._atom()
            value = value and rhs
        return value

    def _atom(self) -> bool:
        if self._peek() == "(":
            self._take()
            value = self._or()
            if self._take() != ")":
                raise BundleError(f"unbalanced parentheses in {self.marker!r}")
            return value
        left = self._take()
        op = self._take()
        if op == "not":
            if self._take() != "in":
                raise BundleError(f"expected 'not in' in {self.marker!r}")
            op = "not in"
        right = self._take()
        return self._compare(left, op, right)

    def _value(self, token: str) -> tuple[str, bool]:
        """(value, whether it is a version variable)."""
        if token[:1] in ("'", '"'):
            return token[1:-1], False
        if token not in self.env:
            raise BundleError(f"unknown marker variable {token!r} in {self.marker!r}")
        return self.env[token], token in _VERSION_VARS

    def _compare(self, left_token: str, op: str, right_token: str) -> bool:
        left, left_is_version = self._value(left_token)
        right, right_is_version = self._value(right_token)
        if op == "in":
            return left in right
        if op == "not in":
            return left not in right
        if left_is_version or right_is_version:
            return _compare_versions(left, op, right)
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        raise BundleError(f"operator {op!r} on a string in {self.marker!r}")


def marker_holds(marker: str | None, env: Mapping[str, str] = TARGET_ENV) -> bool:
    return True if not marker else _MarkerParser(marker, env).parse()


# ---------------------------------------------------------------- uv.lock


@dataclass(frozen=True)
class LockedWheel:
    name: str
    version: str
    filename: str
    url: str
    sha256: str


def _wheel_fits(filename: str) -> tuple[bool, bool]:
    """(installable on the target, pure Python)."""
    parts = filename.removesuffix(".whl").split("-")
    if len(parts) not in (5, 6):
        return False, False
    py, abi, plat = parts[-3:]
    py_ok = bool(set(py.split(".")) & _PY_TAGS)
    abi_ok = bool(set(abi.split(".")) & _ABI_TAGS)
    plats = plat.split(".")
    pure = plats == ["any"]
    plat_ok = pure or any(_LINUX_X86_64.match(p) for p in plats)
    return py_ok and abi_ok and plat_ok, pure


def _pick_wheel(pkg: Mapping[str, Any]) -> LockedWheel:
    candidates: list[tuple[int, str, Mapping[str, Any]]] = []
    for wheel in pkg.get("wheels", []):
        filename = str(wheel["url"]).rsplit("/", 1)[-1]
        fits, pure = _wheel_fits(filename)
        if fits:
            candidates.append((0 if pure else 1, filename, wheel))
    if not candidates:
        raise BundleError(
            f"uv.lock has no wheel of {pkg['name']} {pkg['version']} for CPython "
            f"{TARGET_PYTHON} on x86_64 Linux"
        )
    _, filename, wheel = min(candidates, key=lambda c: (c[0], c[1]))
    digest = str(wheel.get("hash", ""))
    if not digest.startswith("sha256:"):
        raise BundleError(f"uv.lock records no sha256 for {filename}")
    return LockedWheel(
        normalize(pkg["name"]), str(pkg["version"]), filename, str(wheel["url"]), digest[7:]
    )


def _requirement_name(requirement: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if m is None:
        raise BundleError(f"cannot read requirement {requirement!r}")
    return normalize(m.group(1))


def locked_dependencies(lock_text: str, project: Mapping[str, Any]) -> list[LockedWheel]:
    """The wheels of everything `PACKAGE` needs on the target, as `uv.lock` pins them."""
    lock = tomllib.loads(lock_text)
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for pkg in lock.get("package", []):
        by_name.setdefault(normalize(pkg["name"]), []).append(pkg)

    def entry(name: str, version: str | None) -> Mapping[str, Any]:
        found = by_name.get(name, [])
        if version is not None:
            found = [p for p in found if str(p.get("version")) == version]
        if len(found) != 1:
            raise BundleError(f"uv.lock: expected one entry for {name} {version or ''}")
        return found[0]

    root = entry(normalize(PACKAGE), None)
    declared = {_requirement_name(r) for r in project.get("dependencies", [])}
    locked = {normalize(r["name"]) for r in root.get("metadata", {}).get("requires-dist", [])}
    if declared != locked:
        raise BundleError(
            f"uv.lock is out of date for {PACKAGE} (pyproject: {sorted(declared)}, "
            f"uv.lock: {sorted(locked)}); run `uv lock` and commit it"
        )

    wheels: dict[str, LockedWheel] = {}
    seen: set[tuple[str, str]] = set()
    todo: list[tuple[Mapping[str, Any], tuple[str, ...]]] = [(root, ())]
    while todo:
        pkg, extras = todo.pop()
        key = (normalize(pkg["name"]), ",".join(sorted(extras)))
        if key in seen:
            continue
        seen.add(key)
        deps = list(pkg.get("dependencies", []))
        for extra in extras:
            deps += pkg.get("optional-dependencies", {}).get(extra, [])
        for dep in deps:
            if not marker_holds(dep.get("marker")):
                continue
            child = entry(normalize(dep["name"]), dep.get("version"))
            todo.append((child, tuple(dep.get("extra", []))))
        if pkg is not root:
            wheel = _pick_wheel(pkg)
            if wheels.setdefault(wheel.name, wheel) != wheel:
                raise BundleError(f"two versions of {wheel.name} needed")
    return sorted(wheels.values(), key=lambda w: w.filename)


def fetch(wheel: LockedWheel, cache: Path | None) -> bytes:
    """The wheel's bytes, checked against the sha256 in uv.lock."""
    cached = cache / wheel.sha256 / wheel.filename if cache is not None else None
    if cached is not None and cached.is_file():
        data = cached.read_bytes()
        if sha256_hex(data) == wheel.sha256:
            return data
    request = urllib.request.Request(wheel.url, headers={"User-Agent": GENERATOR})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if sha256_hex(data) != wheel.sha256:
        raise BundleError(f"{wheel.filename}: sha256 differs from uv.lock")
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(cached, data)
    return data


# ---------------------------------------------------------------- the proxy's wheel


def _zip_entry(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3  # Unix, whatever builds it (the default depends on the OS)
    info.external_attr = 0o100644 << 16
    return info


def _record_hash(data: bytes) -> str:
    digest = urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"sha256={digest}"


def build_wheel(files: Mapping[str, bytes]) -> tuple[str, bytes]:
    """(file name, bytes) of the `loom-podproxy` wheel from its committed files."""
    pyproject = tomllib.loads(files[f"{PACKAGE_DIR}/pyproject.toml"].decode())
    project = pyproject["project"]
    if normalize(project["name"]) != PACKAGE:
        raise BundleError(f"{PACKAGE_DIR}/pyproject.toml is not {PACKAGE}")
    if project.get("optional-dependencies"):
        raise BundleError("extras are not supported in the bundle's wheel")
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    dist = f"{project['name'].replace('-', '_')}-{project['version']}"

    contents: dict[str, bytes] = {}
    for package in packages:
        src = PurePosixPath(PACKAGE_DIR) / package
        for path, data in files.items():
            pure = PurePosixPath(path)
            if pure.is_relative_to(src):
                contents[str(PurePosixPath(src.name) / pure.relative_to(src))] = data
    if not contents:
        raise BundleError(f"no package files under {PACKAGE_DIR}")

    meta = [
        "Metadata-Version: 2.1",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project.get('description', '')}",
    ]
    if "license" in project:
        meta.append(f"License: {project['license']}")
    if "requires-python" in project:
        meta.append(f"Requires-Python: {project['requires-python']}")
    meta += [f"Requires-Dist: {r}" for r in project.get("dependencies", [])]
    scripts = project.get("scripts", {})
    entry_points = ["[console_scripts]", *(f"{k} = {v}" for k, v in sorted(scripts.items()))]
    wheel_meta = [
        "Wheel-Version: 1.0",
        f"Generator: {GENERATOR}",
        "Root-Is-Purelib: true",
        "Tag: py3-none-any",
    ]
    contents[f"{dist}.dist-info/METADATA"] = ("\n".join(meta) + "\n").encode()
    contents[f"{dist}.dist-info/WHEEL"] = ("\n".join(wheel_meta) + "\n").encode()
    contents[f"{dist}.dist-info/entry_points.txt"] = ("\n".join(entry_points) + "\n").encode()

    record_name = f"{dist}.dist-info/RECORD"
    record = [f"{name},{_record_hash(data)},{len(data)}" for name, data in sorted(contents.items())]
    record.append(f"{record_name},,")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in sorted(contents, key=lambda n: (".dist-info/" in n, n)):
            zf.writestr(_zip_entry(name), contents[name])
        zf.writestr(_zip_entry(record_name), ("\n".join(record) + "\n").encode())
    return f"{dist}-py3-none-any.whl", buf.getvalue()


# ---------------------------------------------------------------- the tarball and lock


def build_tarball(wheels: Iterable[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for filename, data in sorted(wheels):
            info = tarfile.TarInfo(f"{BUNDLE_DIR}/{filename}")
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def render_lock(filename: str, digest: str, wheels: Iterable[str]) -> str:
    lines = [
        "# Pinned serving artifacts (docs/phase1-m2-spec.md section 9). Written by",
        "# `loom-podproxy-bundle` from the committed tree; do not edit by hand. Bootstrap",
        "# checks the shipped proxy bundle against this sha256.",
        "",
        f"[{LOCK_TABLE}]",
        f'file = "{filename}"',
        f'sha256 = "{digest}"',
        f'python = "{TARGET_PYTHON}"',
        "wheels = [",
        *(f'    "{w}",' for w in sorted(wheels)),
        "]",
    ]
    return "\n".join(lines) + "\n"


def read_lock(path: Path) -> dict[str, Any]:
    """The `[podproxy-bundle]` table of a lock file (`file`, `sha256`, ...)."""
    table = tomllib.loads(path.read_text(encoding="utf-8")).get(LOCK_TABLE)
    if not isinstance(table, dict) or not isinstance(table.get("sha256"), str):
        raise BundleError(f"{path} has no [{LOCK_TABLE}] sha256")
    return table


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@dataclass(frozen=True)
class Bundle:
    filename: str
    data: bytes
    sha256: str
    wheels: tuple[str, ...]


def build(repo: Path, cache: Path | None) -> Bundle:
    files = committed_files(repo, PACKAGE_DIR, "uv.lock")
    if "uv.lock" not in files or f"{PACKAGE_DIR}/pyproject.toml" not in files:
        raise BundleError("HEAD has no uv.lock or no podproxy pyproject.toml")
    own_name, own = build_wheel(files)
    project = tomllib.loads(files[f"{PACKAGE_DIR}/pyproject.toml"].decode())["project"]
    deps = locked_dependencies(files["uv.lock"].decode(), project)
    wheels = [(own_name, own), *((w.filename, fetch(w, cache)) for w in deps)]
    data = build_tarball(wheels)
    filename = f"{BUNDLE_DIR}-{project['version']}.tar"
    return Bundle(filename, data, sha256_hex(data), tuple(sorted(n for n, _ in wheels)))


def _default_cache() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "loom" / "podproxy-bundle-wheels"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loom-podproxy-bundle",
        description="Build the pinned pod proxy bundle from the committed tree (git HEAD) "
        "and record its sha256 in the lock.",
    )
    p.add_argument("--out", type=Path, required=True, metavar="DIR", help="where to write it")
    p.add_argument(
        "--lock",
        type=Path,
        metavar="FILE",
        help=f"lock file to record the sha256 in (default: <repo>/{LOCK_DEFAULT})",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="do not write the lock; exit 1 when it does not record this build",
    )
    p.add_argument(
        "--repo", type=Path, default=None, help="repository (default: the one around cwd)"
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"downloaded wheel cache, checked against uv.lock (default: {_default_cache()})",
    )
    p.add_argument("--no-cache", action="store_true", help="download every wheel")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        repo = repo_root(args.repo or Path.cwd())
        warn_if_dirty(repo)
        cache = None if args.no_cache else (args.cache_dir or _default_cache())
        bundle = build(repo, cache)
        lock = args.lock or repo / LOCK_DEFAULT
        lock_text = render_lock(bundle.filename, bundle.sha256, bundle.wheels)
        if args.check:
            recorded = read_lock(lock) if lock.is_file() else {}
            if recorded.get("sha256") != bundle.sha256:
                raise BundleError(
                    f"{lock} records {recorded.get('sha256')}, this tree builds {bundle.sha256}"
                )
        args.out.mkdir(parents=True, exist_ok=True)
        _write_atomic(args.out / bundle.filename, bundle.data)
        if not args.check:
            lock.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(lock, lock_text.encode())
    except (BundleError, OSError) as e:
        raise SystemExit(f"loom-podproxy-bundle: {e}") from None
    print(f"{bundle.sha256}  {args.out / bundle.filename}")
    for name in bundle.wheels:
        print(f"  {name}")


if __name__ == "__main__":
    main()
