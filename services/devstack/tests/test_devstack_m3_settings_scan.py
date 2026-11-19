"""The domain is a setting, not a literal (M3 spec §6.11, A35), and the RDS password is
never in the Terraform configuration (§5.2, A4's grep half).

- No domain literal in `infra/aws/platform/*.tf` or `deploy/**`: every host name there is
  a third party's (AWS, Let's Encrypt, Stripe, GitHub, the image registries, ...) or a
  reserved test name; the product's domain comes from `var.domain` / `{$LOOM_DOMAIN}`.
- The Caddyfile names its three sites `{$LOOM_DOMAIN}`, `api.{$LOOM_DOMAIN}` and
  `hooks.{$LOOM_DOMAIN}` (§6.7), and `hooks.` routes only `POST /stripe/webhook` and
  `GET /stripe/health` (§8.5; A37 checks the behaviour through Caddy in A15).
- No `password = ...` attribute in `infra/aws/platform` (the master password is
  RDS-managed: `manage_master_user_password = true`).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLATFORM = ROOT / "infra" / "aws" / "platform"
DEPLOY = ROOT / "deploy"
CADDYFILE = DEPLOY / "Caddyfile"

# Host names that may appear: third parties the stack talks to or pulls from. Subdomains
# count (`ends with .<name>`), so this covers e.g. the AWS service endpoints
# (`*.amazonaws.com`, ECR's `<account>.dkr.ecr.<region>.amazonaws.com`), SES's MAIL FROM
# records (SPF `include:amazonses.com`, MX `feedback-smtp.<region>.amazonses.com`) and
# DKIM CNAMEs (`*.dkim.amazonses.com`), the Let's Encrypt ACME directories
# (`acme-v02.api.letsencrypt.org`, `acme-staging-v02.api.letsencrypt.org`), and the IAM
# OIDC provider for GitHub (`token.actions.githubusercontent.com`).
THIRD_PARTY = (
    "amazonaws.com",
    "amazon.com",
    "amazonses.com",
    "amazontrust.com",
    "awsstatic.com",
    "aws.dev",
    "letsencrypt.org",
    "lencr.org",
    "stripe.com",
    "github.com",
    "githubusercontent.com",
    "docker.io",
    "docker.com",
    "ghcr.io",
    "quay.io",
    "gcr.io",
    "grafana.com",
    "prometheus.io",
    "caddyserver.com",
    "golang.org",
    "go.dev",
    "googlesource.com",
    "google.com",
    "googleapis.com",
    "gstatic.com",
    "grafana.net",
    "hashicorp.com",
    "terraform.io",
    "python.org",
    "pypi.org",
    "npmjs.org",
    "npmjs.com",
    "nodejs.org",
    "alpinelinux.org",
    "debian.org",
    "json-schema.org",
    "w3.org",
)
RESERVED_TLDS = ("test", "example", "invalid", "localhost", "local", "internal")
RESERVED_DOMAINS = ("example.com", "example.net", "example.org")  # RFC 2606
# A word.tld host name, also inside a URL (`https://x.com/...`) or an e-mail address
# (`no-reply@x.com`); the TLDs are those a product domain would plausibly use (so
# Terraform references such as `aws_ssm_parameter.site` or file names do not match).
TLDS = "com|net|org|io|ai|co|dev|app|cloud|tech|xyz|gg|us|uk|de|eu|info|biz|ly"
HOST = re.compile(rf"(?<![\w.-])((?:[a-z0-9-]+\.)+(?:{TLDS}))(?![\w-])", re.I)
TF_NAMESPACES = {"var", "local", "data", "module", "each", "count", "self", "path", "terraform"}
SKIP_FILES = {"go.sum", "go.mod"}  # Go module paths of the Caddy build, not settings
PLACEHOLDER = re.compile(r"\{(\$?[A-Za-z_][\w.:/-]*)\}")


def strip_comments(text: str, suffix: str) -> str:
    if suffix in (".tf", ".hcl"):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        return "\n".join(re.sub(r"\s(#|//).*$", "", " " + line)[1:] for line in text.splitlines())
    if suffix in (".yml", ".yaml", "") or suffix == ".caddyfile":
        return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in text.splitlines())
    return text


def third_party(host: str) -> bool:
    h = host.lower()
    if h.split(".")[-1] in RESERVED_TLDS:
        return True
    return any(h == d or h.endswith("." + d) for d in (*THIRD_PARTY, *RESERVED_DOMAINS))


def domain_literals(files: list[Path]) -> list[str]:
    hits = []
    for path in files:
        suffix = ".caddyfile" if path.name == "Caddyfile" else path.suffix
        text = strip_comments(path.read_text(errors="replace"), suffix)
        for m in HOST.finditer(text):
            host = m.group(1)
            if host.split(".")[0] in TF_NAMESPACES:
                continue
            if not third_party(host):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}: {host}")
    return hits


def test_no_domain_literal_in_deploy() -> None:
    """deploy/** (the deploy worker's files)."""
    assert DEPLOY.is_dir(), "deploy/ does not exist"
    files = [
        p
        for p in sorted(DEPLOY.rglob("*"))
        if p.is_file() and p.name not in SKIP_FILES and b"\0" not in p.read_bytes()[:4096]
    ]
    assert files, "deploy/ is empty"
    hits = domain_literals(files)
    assert not hits, "domain literals (the domain is a setting, §6.11):\n" + "\n".join(hits)


def test_no_domain_literal_in_the_platform_terraform() -> None:
    """infra/aws/platform/*.tf (the infra worker's files)."""
    files = sorted(PLATFORM.glob("*.tf"))
    assert files, "infra/aws/platform has no .tf files"
    hits = domain_literals(files)
    assert not hits, "domain literals (the domain is a setting, §6.11):\n" + "\n".join(hits)


def test_platform_variables_domain_and_brand_name_have_no_default() -> None:
    text = (PLATFORM / "variables.tf").read_text()
    for name in ("domain", "brand_name"):
        m = re.search(rf'variable\s+"{name}"\s*\{{(.*?)\n\}}', text, re.S)
        assert m, f'variables.tf has no variable "{name}"'
        assert not re.search(r"^\s*default\s*=", m.group(1), re.M), f"{name} has a default"


def caddy_sites() -> dict[str, str]:
    """Top-level site blocks of the Caddyfile: address line -> body."""
    assert CADDYFILE.is_file(), "deploy/Caddyfile does not exist"
    text = strip_comments(CADDYFILE.read_text(), ".caddyfile")
    # Placeholders ({$LOOM_DOMAIN}, {remote_host}, ...) are not blocks: mask their braces.
    masked = PLACEHOLDER.sub(lambda m: "\x01" + m.group(1) + "\x02", text)

    def unmask(s: str) -> str:
        return s.replace("\x01", "{").replace("\x02", "}")

    sites: dict[str, str] = {}
    depth, start, header = 0, 0, ""
    for i, ch in enumerate(masked):
        if ch == "{":
            if depth == 0:
                header = masked[masked.rfind("\n", 0, i) + 1 : i].strip()
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                sites[unmask(header)] = unmask(masked[start:i])
    # Snippets `(name) { ... }`: expand `import name [args]` lines (also inside a snippet).
    snippets = {h[1:-1]: b for h, b in sites.items() if h.startswith("(") and h.endswith(")")}

    def expand(body: str, depth: int = 0) -> str:
        assert depth < 10, "snippet imports nest too deep"

        def one(m: re.Match[str]) -> str:
            name = m.group(1)
            return expand(snippets[name], depth + 1) if name in snippets else m.group(0)

        return re.sub(r"^[ \t]*import[ \t]+(\S+).*$", one, body, flags=re.M)

    return {h: expand(b) for h, b in sites.items() if h not in {f"({s})" for s in snippets}}


def site_addresses(header: str) -> list[str]:
    """The HTTPS addresses of a site block's header, without `https://` and `:443`;
    `http://` and `:80` addresses (the redirects) are not the three sites."""
    out = []
    for a in re.split(r"[,\s]+", header):
        if not a or a.startswith("http://") or a.endswith(":80"):
            continue
        out.append(a.removeprefix("https://").removesuffix(":443"))
    return out


def site(prefix: str) -> str:
    sites = caddy_sites()
    want = f"{prefix}{{$LOOM_DOMAIN}}"
    found = [body for addr, body in sites.items() if want in site_addresses(addr)]
    assert found, f"no site block for {want} in the Caddyfile: {list(sites)}"
    return found[0]


def test_the_caddyfile_names_the_domain_only_through_loom_domain() -> None:
    for prefix in ("", "api.", "hooks."):
        site(prefix)
    text = CADDYFILE.read_text()
    assert "{$LOOM_DOMAIN:" not in text, "the Caddyfile gives LOOM_DOMAIN a default domain"


def test_hooks_routes_only_the_webhook_and_its_health_check_to_billing() -> None:
    body = site("hooks.")
    # Exact paths or a `path_regexp` such as `^/stripe/(webhook|health)$` (the routing
    # itself is checked in test_devstack_m3_caddyfile.py).
    for route in ("webhook", "health"):
        assert re.search(rf"/stripe/\S*\b{route}\b", body), f"hooks. names no /stripe/{route}"
    assert "billing:8090" in body, "hooks. does not proxy to billing:8090"
    for path in ("/internal", "/health ", "/metrics"):
        assert f"handle {path}" not in body and f"path {path}" not in body, path


def test_api_routes_v1_to_the_gateway() -> None:
    body = site("api.")
    assert "/v1/" in body and "gateway:8080" in body, body
    assert "flush_interval -1" in body, "api. buffers streamed responses"
    assert "lb_try_duration" in body, "api. has no lb_try_duration for gateway restarts"


def test_no_password_attribute_in_the_platform_configuration() -> None:
    files = sorted(PLATFORM.rglob("*.tf")) + sorted(PLATFORM.rglob("*.tfvars*"))
    assert files, "infra/aws/platform has no Terraform files"
    hits = []
    for path in files:
        text = strip_comments(path.read_text(), ".tf")
        for n, line in enumerate(text.splitlines(), 1):
            # A4: the `password` (or write-only `password_wo`) attribute; names such as
            # `manage_master_user_password` are not it.
            if re.match(r"^\s*password(_wo)?\s*=", line):
                hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not hits, "password attributes in Terraform (§5.2: RDS-managed):\n" + "\n".join(hits)
    rds = PLATFORM / "rds.tf"
    assert rds.is_file(), "infra/aws/platform/rds.tf does not exist"
    assert re.search(r"manage_master_user_password\s*=\s*true", rds.read_text()), (
        "rds.tf does not set manage_master_user_password = true"
    )
