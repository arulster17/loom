"""`site/config.yaml`: settings for the public results site that are not results."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, field_validator

from loom_bench.registry import REPO_ROOT, read_yaml

DEFAULT_SITE_DIR = REPO_ROOT / "site"
DEFAULT_SITE_CONFIG = DEFAULT_SITE_DIR / "config.yaml"
DEFAULT_SNAPSHOT_DIR = DEFAULT_SITE_DIR / "data"
DEFAULT_BUILD_DIR = DEFAULT_SITE_DIR / "_build"

FieldName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_.\-]*$")]


class WaitlistConfig(BaseModel):
    """Where the waitlist form posts. With no `action_url` the site renders no form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_url: HttpUrl | None = None
    method: Literal["GET", "POST"] = "POST"
    field: FieldName = "email"
    # Spam trap: a field humans never see. `_gotcha` is the name Formspree checks.
    honeypot: FieldName = "_gotcha"
    extra_fields: dict[FieldName, str] = Field(default_factory=dict)

    @field_validator("method", mode="before")
    @classmethod
    def _upper(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

    @property
    def enabled(self) -> bool:
        return self.action_url is not None


class SiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = "Loom"
    repo_url: HttpUrl = HttpUrl("https://github.com/arulster17/loom")
    waitlist: WaitlistConfig = Field(default_factory=WaitlistConfig)

    @property
    def repo(self) -> str:
        return str(self.repo_url).rstrip("/")


def load_site_config(path: Path | str | None = None) -> SiteConfig:
    data = read_yaml(Path(path or DEFAULT_SITE_CONFIG))
    return SiteConfig.model_validate(data or {})
