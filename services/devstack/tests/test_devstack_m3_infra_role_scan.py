"""Nothing outside the listed grants gives the platform's IAM roles permissions
(docs/phase1-m3-spec.md §13 A6, §5.2 iam, §5.6).

A6's "exactly" (§5.6: §5.2 plus the §5.6 table, nothing else) is checked by the
tftests on the roles' own policies (`infra/aws/platform/tests/platform_iam.tftest.hcl`).
A plan cannot list resources a test does not name, so a second inline policy or an
extra managed-policy attachment on a role would slip past them. This static scan of
the stack's `*.tf` files closes that gap:

- instance role: its inline policy `aws_iam_role_policy.instance` and the
  `AmazonSSMManagedInstanceCore` attachment (§5.2), nothing else;
- GitHub OIDC role: its inline policy `aws_iam_role_policy.github_images` only;
- serving reaper's role: its inline policy `aws_iam_role_policy.serving_reaper` only;
- no role block carries `inline_policy` or `managed_policy_arns`;
- every role-grant resource names one of those roles directly (by reference or by its
  literal name), so a grant cannot hide behind an indirection.

The patterns assume `terraform fmt` layout (CI checks it): top-level blocks start at
column 0 and end with a `}` alone at column 0; top-level attributes are indented two
spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STACK = ROOT / "infra" / "aws" / "platform"

ROLES = ("instance", "github_images", "serving_reaper")

# Resource types that attach permissions to a role.
GRANT_TYPE_RE = re.compile(r"^aws_iam_(role_polic(y|ies)|policy_attachment)")

BLOCK_RE = re.compile(
    r'(?ms)^resource\s+"(?P<type>[A-Za-z0-9_]+)"\s+"(?P<name>[A-Za-z0-9_-]+)"\s*\{'
    r"(?P<body>[^\n]*\}$|.*?^\})"
)

# The one managed policy §5.2 lists for the instance role.
SSM_CORE_RE = re.compile(
    r'(?m)^  policy_arn\s*=\s*"arn:[^"]*:iam::aws:policy/AmazonSSMManagedInstanceCore"\s*$'
)


@dataclass(frozen=True)
class Block:
    file: str
    type: str
    name: str
    body: str

    @property
    def address(self) -> str:
        return f"{self.type}.{self.name}"


def parse_blocks(sources: dict[str, str]) -> list[Block]:
    out: list[Block] = []
    for fname in sorted(sources):
        for m in BLOCK_RE.finditer(sources[fname]):
            out.append(Block(fname, m["type"], m["name"], m["body"]))
    return out


def stack_sources() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(STACK.glob("*.tf"))}


def role_names(blocks: list[Block]) -> dict[str, str]:
    """Role resource name -> the role's literal `name` attribute (if set)."""
    out: dict[str, str] = {}
    for b in blocks:
        if b.type == "aws_iam_role":
            m = re.search(r'(?m)^  name\s*=\s*"([^"]+)"', b.body)
            if m:
                out[b.name] = m[1]
    return out


def role_targets(block: Block, names: dict[str, str]) -> set[str] | None:
    """Which of the known roles a grant resource attaches to; None if it names none."""
    values: list[str] = []
    for m in re.finditer(r"(?m)^  (?:role|role_name)\s*=\s*(.+)$", block.body):
        values.append(m[1])
    for m in re.finditer(r"(?ms)^  roles\s*=\s*\[(.*?)\]", block.body):
        values.append(m[1])
    found: set[str] = set()
    for v in values:
        for ref in re.findall(r"\baws_iam_role\.([A-Za-z0-9_-]+)\b", v):
            found.add(ref)
        for role, literal in names.items():
            if re.search(r'"' + re.escape(literal) + r'"', v):
                found.add(role)
    return found or None


def violations(sources: dict[str, str]) -> list[str]:
    blocks = parse_blocks(sources)
    names = role_names(blocks)
    problems: list[str] = []

    for b in blocks:
        if (
            b.type == "aws_iam_role"
            and b.name in ROLES
            and re.search(r"(?m)^  (inline_policy\s*\{|managed_policy_arns\s*=)", b.body)
        ):
            problems.append(
                f"{b.address} ({b.file}) carries inline_policy/managed_policy_arns: "
                "grants belong in the listed policy resources only"
            )

    for b in blocks:
        if not GRANT_TYPE_RE.match(b.type):
            continue
        targets = role_targets(b, names)
        if targets is None:
            problems.append(
                f"{b.address} ({b.file}) attaches permissions to a role this scan cannot "
                "identify (name the role directly: aws_iam_role.<name> or its literal name)"
            )
            continue
        for role in sorted(targets):
            if role == "instance":
                if b.address == "aws_iam_role_policy.instance":
                    continue
                if b.type == "aws_iam_role_policy_attachment" and SSM_CORE_RE.search(b.body):
                    continue
                problems.append(
                    f"{b.address} ({b.file}) grants the instance role more than §5.2/§5.6 list "
                    "(allowed: aws_iam_role_policy.instance and the AmazonSSMManagedInstanceCore "
                    "attachment)"
                )
            elif role == "github_images":
                if b.address != "aws_iam_role_policy.github_images":
                    problems.append(
                        f"{b.address} ({b.file}) grants the OIDC role more than "
                        "aws_iam_role_policy.github_images"
                    )
            elif role == "serving_reaper":
                if b.address != "aws_iam_role_policy.serving_reaper":
                    problems.append(
                        f"{b.address} ({b.file}) grants the serving reaper's role more than "
                        "aws_iam_role_policy.serving_reaper"
                    )
            # Other roles (e.g. the DLM service role of §6.10) are outside A6.
    return problems


def test_stack_files_are_found_and_parsed() -> None:
    sources = stack_sources()
    assert "iam.tf" in sources, f"no iam.tf in {STACK}"
    addresses = {b.address for b in parse_blocks(sources)}
    for role in ROLES:
        assert f"aws_iam_role.{role}" in addresses
        assert f"aws_iam_role_policy.{role}" in addresses


def test_the_instance_role_has_its_policy_and_the_ssm_attachment() -> None:
    """The scan sees the grants §5.2 lists (so the next test is not vacuous)."""
    blocks = parse_blocks(stack_sources())
    names = role_names(blocks)
    instance_grants = [
        b
        for b in blocks
        if GRANT_TYPE_RE.match(b.type) and "instance" in (role_targets(b, names) or set())
    ]
    assert "aws_iam_role_policy.instance" in {b.address for b in instance_grants}
    ssm = [
        b
        for b in instance_grants
        if b.type == "aws_iam_role_policy_attachment"
        and re.search(r"(?m)^  policy_arn\s*=.*AmazonSSMManagedInstanceCore", b.body)
    ]
    assert len(ssm) == 1, "the instance role has the AmazonSSMManagedInstanceCore attachment (§5.2)"


def test_nothing_else_grants_the_roles_permissions() -> None:
    problems = violations(stack_sources())
    assert not problems, "\n".join(problems)


EXTRA_INLINE = """
resource "aws_iam_role_policy" "instance_extra" {
  name = "extra"
  role = aws_iam_role.instance.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "s3:*", Resource = "*" }]
  })
}
"""

ADMIN_ATTACHMENT = """
resource "aws_iam_role_policy_attachment" "instance_admin" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}
"""

LITERAL_NAME_ATTACHMENT = """
resource "aws_iam_role_policy_attachment" "by_name" {
  role       = "loom-platform-instance"
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}
"""

REAPER_BASIC_EXECUTION = """
resource "aws_iam_role_policy_attachment" "reaper_basic" {
  role       = aws_iam_role.serving_reaper.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
"""

OIDC_EXTRA = """
resource "aws_iam_role_policy" "github_more" {
  role   = aws_iam_role.github_images.id
  policy = "{}"
}
"""

INDIRECT = """
resource "aws_iam_role_policy_attachment" "indirect" {
  role       = local.some_role
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}
"""

LEGACY_ATTACHMENT = """
resource "aws_iam_policy_attachment" "legacy" {
  name       = "legacy"
  roles      = [aws_iam_role.instance.name]
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}
"""


@pytest.mark.parametrize(
    "extra",
    [
        EXTRA_INLINE,
        ADMIN_ATTACHMENT,
        LITERAL_NAME_ATTACHMENT,
        REAPER_BASIC_EXECUTION,
        OIDC_EXTRA,
        INDIRECT,
        LEGACY_ATTACHMENT,
    ],
    ids=[
        "extra_inline_s3_star",
        "admin_attachment",
        "attachment_by_literal_role_name",
        "reaper_managed_policy",
        "oidc_second_inline_policy",
        "role_through_a_local",
        "aws_iam_policy_attachment",
    ],
)
def test_the_scan_catches_an_extra_grant(extra: str) -> None:
    sources = stack_sources()
    sources["zz_extra.tf"] = extra.lstrip("\n")
    assert violations(sources), "the scan must flag this extra grant"


def test_the_scan_catches_managed_policy_arns_on_a_role() -> None:
    sources = stack_sources()
    iam = sources["iam.tf"]
    marker = 'resource "aws_iam_role" "instance" {\n'
    assert marker in iam
    sources["iam.tf"] = iam.replace(
        marker,
        marker + '  managed_policy_arns = ["arn:aws:iam::aws:policy/AdministratorAccess"]\n',
    )
    assert violations(sources)


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
