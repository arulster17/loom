"""python -m loom_bench.site {export,build}; see docs/site.md."""

from __future__ import annotations

import argparse
from pathlib import Path

from loom_bench.site.build import build_site
from loom_bench.site.config import DEFAULT_BUILD_DIR, DEFAULT_SITE_CONFIG, DEFAULT_SNAPSHOT_DIR
from loom_bench.site.snapshot import export_snapshot
from loom_bench.slo import Slo
from loom_bench.store.db import session_scope


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m loom_bench.site")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="write a results snapshot from the results store")
    export.add_argument("--out", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    export.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        metavar="ID",
        help="experiment id to publish (repeatable); default: latest completed of each name",
    )
    export.add_argument("--database-url", help="default: $LOOM_DATABASE_URL")
    export.add_argument("--slo", type=Path, help="SLO YAML, if the runs do not record one")

    build = sub.add_parser("build", help="render the static site from a snapshot")
    build.add_argument("--data", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    build.add_argument("--config", type=Path, default=DEFAULT_SITE_CONFIG)
    build.add_argument("--out", type=Path, default=DEFAULT_BUILD_DIR)

    args = parser.parse_args(argv)
    if args.command == "export":
        slo = Slo.from_yaml(args.slo.read_text(encoding="utf-8")) if args.slo else None
        with session_scope(args.database_url) as session:
            manifest = export_snapshot(session, args.out, args.experiments or "latest", slo=slo)
        configs = sum(m.configs for m in manifest.models)
        print(
            f"wrote {args.out}: {len(manifest.experiment_ids)} experiments, "
            f"{configs} configs, {manifest.run_count} runs"
        )
    else:
        pages = build_site(args.data, args.out, args.config)
        print(f"wrote {len(pages)} pages to {args.out}")


if __name__ == "__main__":
    main()
