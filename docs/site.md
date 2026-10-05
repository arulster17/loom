# Public results site

The Phase 0 results page: methodology, the exact configs, the open-source harness,
and a waitlist form. It is a static site built from a **results snapshot** committed
to git, so every published number is reviewable in a pull request and the site needs
no server or database.

```
results store (Postgres) ──export──▶ site/data/  (committed JSON snapshot)
                                         │
                                       build
                                         ▼
                                    site/_build/  (static HTML, gitignored) ──▶ GitHub Pages
```

Code: `bench/src/loom_bench/site/`. Public functions: `export_snapshot`, `build_site`,
`load_snapshot`, `waitlist_count`, `record_count`.

## Pages

| Page | Shows |
|---|---|
| `index.html` | What Loom is, the headline table (best ranked config per model and workload), the waitlist |
| `models/<id>.html` | Leaderboard per workload with CIs, quality gate, cold start, per-config details: config YAML, load sweep, price source, `bench reproduce` command, links to every run's provenance JSON |
| `methodology.html` | Open and closed loop, metric definitions, goodput at SLO, repetitions and CIs, cost formula with a worked example, price sources, quality gate rule, synthetic vs realistic content, what we don't do |
| `pricing.html` | Our cost at SLO, planned price and margin per model, next to public list prices with sources |
| `harness.html` | Repository link, license, quickstart with the mock backend |

Every registry model gets a page. Until a model has published results its pages say
"No published results yet — first runs pending" and show the registry configuration
queued for benchmarking; no number is shown that was not measured. The methodology
page's worked example uses the first ranked published result; before there is one it
uses a real instance price with a round, clearly labelled illustrative throughput.

## Publishing results after runs

1. Export a snapshot from the results store. By default it takes the newest completed
   experiment of each name; pass `--experiment <id>` (repeatable) to choose.

   ```bash
   LOOM_DATABASE_URL=postgresql+psycopg://... \
     uv run bench site export --out site/data      # or --db URL
   ```

   The export analyses the runs with the report module (`analyze_runs`,
   `with_quality`, `cold_starts_by_config`, `build_competitiveness`): goodput at SLO,
   cost from `bench/prices.yaml`, quality gate, cold starts; the same analysis as
   `bench report` and the summary after `bench run`. It replaces the previous
   snapshot. The SLO comes from the experiment specs and run summaries; if they
   disagree, pass `--slo slo.yaml` or publish the experiments separately.

2. Build and check locally:

   ```bash
   uv run bench site build        # site/data -> site/_build
   python -m http.server 8000 --directory site/_build
   # open http://localhost:8000
   ```

3. Commit `site/data/` and open a pull request. Merging to `main` deploys.

From Python, the same steps are
`export_snapshot(session, "site/data", "latest" | [experiment ids], slo=..., allocation=...)`
and `build_site("site/data", "site/_build", "site/config.yaml")`.

## What gets published

Everything in the snapshot is copied to `data/` on the site and linked from the pages:

| File | Contents |
|---|---|
| `manifest.json` | Generation time, git commit of the export, loom-bench version, experiment ids, SLO, cost allocation, model index |
| `models/<id>.json` | The model's registry entry and its analysed results (`ConfigResult`s, with every load point and its CIs; each estimate names its interval `method`: `log_t` geometric mean with a log-scale t-interval, `t_clipped` or `t`) and cold starts |
| `competitiveness.json` | Our cost at SLO, planned price and margin, competitors' public list prices with sources and flags |
| `prices.json` | The price book used for every cost |
| `experiments/<id>.json` | Each experiment's record, including its full spec and spend |
| `provenance/<run id>.json` | Every run a result uses: run metadata, summary, and the provenance record exactly as stored. `provenance_digest` is the sha256 of the record's canonical JSON |

Experiment specs and provenance are published in full; keep credentials and other
secrets out of them.

The site loads no external scripts, fonts or images. Its only external links are to
the GitHub repository, model weights on Hugging Face, and the price and dataset
sources named in the data.

## Deployment

`.github/workflows/site.yml` builds and deploys on pushes to `main` that touch
`site/**`, `bench/src/loom_bench/site/**`, `bench/src/loom_bench/report/**` or the
workflow itself, and on manual dispatch (Actions → site → Run workflow).

**One-time setup, not done yet:** in the repository's Settings → Pages, set
**Source: GitHub Actions**, then add the repository variable `LOOM_PAGES_ENABLED=true`
(Settings → Secrets and variables → Actions → Variables). Until then the workflow only
builds the site. The site is then served at
`https://arulster17.github.io/loom/`; all links are relative, so the `/loom/` base path
needs no configuration.

## Waitlist endpoint

`site/config.yaml`:

```yaml
waitlist:
  action_url: null      # where the form posts
  method: POST
  field: email          # name of the email field the endpoint expects
  honeypot: _gotcha     # hidden spam-trap field
  extra_fields: {}      # extra hidden fields, e.g. {source: results-page}
```

While `action_url` is null, the page shows "Waitlist opens soon" with the repository
link and no form, so nobody submits into nothing. To open the waitlist, set
`action_url` and push:

- **Formspree**: `action_url: https://formspree.io/f/<form id>`. It drops submissions
  that fill the `_gotcha` honeypot and answers cross-origin requests with JSON, so the
  inline success message works.
- **Another form service** (Buttondown, ...): use its form-post URL, set `field` and
  `honeypot` to the names it expects, and check that it answers a cross-origin POST
  with CORS headers and a 2xx status. If it only redirects, the inline script reports
  an error even when the signup went through; remove the script or use Formspree.
- **A Loom endpoint**: accept a form POST with `email`, return 2xx with CORS headers
  for the site's origin, drop requests whose honeypot field is filled, and store with
  `loom_bench.store.repo.add_waitlist_signup`.

The form works without JavaScript (a normal POST); with JavaScript it submits in place
and shows success or the error inline. It carries a required `type=email` field, the
honeypot, and the line "We only use this to tell you when Loom opens; no sharing."

Record the signup count as a demand signal in [waitlist.md](waitlist.md).
