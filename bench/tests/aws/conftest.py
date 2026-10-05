from collections.abc import Iterator
from typing import Any

import boto3
import pytest
from moto import mock_aws

REGION = "us-east-1"


@pytest.fixture(autouse=True)
def fake_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


@pytest.fixture
def moto_aws() -> Iterator[None]:
    with mock_aws():
        yield


@pytest.fixture
def ec2(moto_aws: None) -> Any:
    return boto3.client("ec2", region_name=REGION)


def amazon_ami(ec2: Any) -> str:
    # Filtered: moto serializes ~1200 Amazon AMIs (0.5 s) for an unfiltered listing.
    images = ec2.describe_images(
        Owners=["amazon"], Filters=[{"Name": "name", "Values": ["amzn2-ami-hvm-*"]}]
    )
    return str(images["Images"][0]["ImageId"])


def launch(ec2: Any, tags: dict[str, str]) -> str:
    resp = ec2.run_instances(
        ImageId=amazon_ami(ec2),
        InstanceType="g6e.xlarge",
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[
            {"ResourceType": "instance", "Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}
        ],
    )
    return str(resp["Instances"][0]["InstanceId"])


def state(ec2: Any, instance_id: str) -> str:
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    return str(resp["Reservations"][0]["Instances"][0]["State"]["Name"])
