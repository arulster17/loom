"""Static checks on infra/aws/bench (terraform itself is not installed in CI)."""

import re
from pathlib import Path

import pytest

from loom_bench.providers import aws_ec2, aws_reaper, runpod_reaper_lambda
from loom_bench.providers.aws_ec2 import AwsSettings

TF_DIR = Path(__file__).resolve().parents[3] / "infra" / "aws" / "bench"
TF_FILES = sorted(TF_DIR.glob("*.tf"))


def tf_text() -> str:
    return "\n".join(p.read_text() for p in TF_FILES)


def test_terraform_files_present() -> None:
    names = {p.name for p in TF_FILES}
    assert {"versions.tf", "variables.tf", "main.tf", "reaper.tf", "outputs.tf"} <= names
    assert "runpod_reaper.tf" in names


@pytest.mark.parametrize("path", TF_FILES, ids=lambda p: p.name)
def test_braces_balance(path: Path) -> None:
    text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', path.read_text())
    assert text.count("{") == text.count("}")
    assert text.count("[") == text.count("]")
    assert text.count("(") == text.count(")")


def test_no_account_ids_or_secrets() -> None:
    text = tf_text()
    assert not re.search(r"\b\d{12}\b", text)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)


def test_aws_settings_yaml_output_matches_settings_fields() -> None:
    outputs = (TF_DIR / "outputs.tf").read_text()
    m = re.search(r"aws_settings_yaml.*?yamlencode\(\{(.*?)\}\)", outputs, re.DOTALL)
    assert m
    keys = set(re.findall(r"^\s*(\w+)\s*=", m.group(1), re.MULTILINE))
    assert keys <= set(AwsSettings.model_fields)
    required = {n for n, f in AwsSettings.model_fields.items() if f.is_required()}
    assert required <= keys


def test_reaper_lambda_points_at_the_module() -> None:
    reaper_tf = (TF_DIR / "reaper.tf").read_text()
    src = re.search(r'source_file\s*=\s*"\$\{path\.module\}/([^"]+)"', reaper_tf)
    assert src
    assert (TF_DIR / src.group(1)).resolve() == Path(aws_reaper.__file__).resolve()
    assert 'handler          = "aws_reaper.lambda_handler"' in reaper_tf
    assert 'schedule_expression = "rate(15 minutes)"' in reaper_tf


def test_tag_conditions_match_provider_tags() -> None:
    text = tf_text()
    for tag in (aws_ec2.MANAGED_TAG, aws_ec2.TTL_TAG, aws_ec2.EXPERIMENT_TAG, aws_ec2.OWNER_TAG):
        assert f"aws:RequestTag/{tag}" in text
    assert f"ec2:ResourceTag/{aws_reaper.MANAGED_TAG}" in text
    assert "parameter/aws/service/deeplearning/*" in text
    assert aws_ec2.DLAMI_PARAMETER.startswith("/aws/service/deeplearning/")


def test_security_group_has_no_ingress() -> None:
    text = tf_text()
    assert not re.search(r"^\s*ingress\s*\{", text, re.MULTILINE)
    assert "ingress_rule" not in text
    assert "aws_security_group_rule" not in text


def test_runpod_reaper_lambda_points_at_its_module() -> None:
    tf = (TF_DIR / "runpod_reaper.tf").read_text()
    src = re.search(r'source_file\s*=\s*"\$\{path\.module\}/([^"]+)"', tf)
    assert src
    assert (TF_DIR / src.group(1)).resolve() == Path(runpod_reaper_lambda.__file__).resolve()
    assert 'handler          = "runpod_reaper_lambda.lambda_handler"' in tf
    assert 'schedule_expression = "rate(15 minutes)"' in tf
    for var in (
        "LOOM_RUNPOD_KEY_SECRET_ID",
        "LOOM_RUNPOD_REAPER_PREFIX",
        "LOOM_RUNPOD_REAPER_DRY_RUN",
        "LOOM_RUNPOD_REAPER_MAX_PER_RUN",
    ):
        assert var in tf and var in Path(runpod_reaper_lambda.__file__).read_text()


def test_runpod_reaper_secret_has_no_value_and_only_its_role_reads_it() -> None:
    text = tf_text()
    assert "aws_secretsmanager_secret_version" not in text
    assert "secret_string" not in text
    tf = (TF_DIR / "runpod_reaper.tf").read_text()
    # The role's own policy: its log group and GetSecretValue on that one secret.
    policy = re.search(r'data "aws_iam_policy_document" "runpod_reaper" \{(.*?)\n\}', tf, re.DOTALL)
    assert policy
    actions = re.findall(r'"([a-z0-9]+:[A-Za-z]+)"', policy.group(1))
    assert sorted(actions) == [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "secretsmanager:GetSecretValue",
    ]
    assert "aws_secretsmanager_secret.runpod_reaper_key.arn" in policy.group(1)
    assert "ec2:" not in policy.group(1)
    # The secret's resource policy denies everyone else.
    assert '"ArnNotEquals"' in tf and "aws_iam_role.runpod_reaper.arn" in tf
    # The EC2 reaper's role cannot read it.
    assert "secretsmanager" not in (TF_DIR / "reaper.tf").read_text()


def test_runpod_reaper_starts_in_dry_run() -> None:
    variables = (TF_DIR / "variables.tf").read_text()
    block = re.search(r'variable "runpod_reaper_dry_run" \{(.*?)\n\}', variables, re.DOTALL)
    assert block and "default     = true" in block.group(1)
