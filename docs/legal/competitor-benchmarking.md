# Competitor benchmarking

Loom benchmarks only infrastructure it runs itself. It ships no tooling, credentials,
endpoint presets or scripts for load-testing or evaluating third-party inference APIs.

## Why

Several providers' terms restrict benchmarking or competitive use of their services,
including Together AI, DeepInfra, and Groq's service agreement. Running Loom's load
generator or eval harness against those endpoints could breach those terms, so the
repository does not make it easy to do.

## What we use instead

Competitor data comes only from public price pages. Each figure is entered by hand in
`bench/competitors.yaml` with its source URL and the date it was checked, and is used
only to compare list prices in the competitiveness view. Nothing in the repository
calls a competitor's API.

## Comparing named third-party endpoints

Measuring the latency, throughput or quality of a named third-party endpoint, or
publishing such a comparison, requires that provider's written permission and a
legal review first.
