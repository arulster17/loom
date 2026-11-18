# Agent retros

## 2026-11-16 — Phase 1 M2 (pool manager), docs/phase1-m2-spec.md §14

- **Task / size:** Large. 104 files, ~19.8k lines, 40 commits on `phase1-m2-pool`.
- **Setup:** orchestrator plus per-wave test-writer → reviewer (tests) → implementer → reviewer (code). Wave 2 ran in two worktrees in parallel (`m2-manager`, `m2-proxy-reaper`). A final whole-branch reviewer, then a verifier. About 30 agent runs in total.
- **What review caught:**
  - **Test reviews:** a harness that inflated fake time (it would have been flaky on 4 CPUs); concurrent ticks the spec never required; a bundle sha that depended on CRLF checkouts; a vacuous allowlist test; a missing lease-dir creation test (every pod would have died at its first lease); an acceptance timeout shorter than `idle_stop_s`.
  - **Code reviews:** listing-lag false "lost" pods and ambiguous creates that could run two paid pods; adopted pods drained right after a restart; a dev-mode placeholder bundle that would have bootstrapped paid RunPod pods; one hung SSH call stalling the tick.
- **What went wrong:**
  1. **The stop gate can't wait.** It runs only on the main session's turn end and in the main checkout. Waiting on background agents used up its 3 tries while a wave's tests were red by design. It doesn't gate subagents or worktrees, so implementers self-verified with `loom-verify`.
  2. **`loom_poolmgr/testing.py` wasn't on `protect.paths`.** The hook blocked the test-writer, so the fakes live in `services/poolmgr/tests/poolmgr_fakes.py`.
  3. **`pytest.ini` testpaths lacks `services/poolmgr/tests`.** It's locked, so CI got an explicit step instead.
  4. **WSL/Git-Bash quoting accidents:**
     - a stray `C:` dir of throwaway certs inside the repo;
     - a test file run as a shell script, which ran `docker compose up` and stopped the user's `loom-postgres-1`/`loom-redis-1` (restored, volumes kept);
     - `uv run` from WSL on `/mnt/c` replacing the Windows `.venv`.
  5. **Devcerts files owned by the container's root** broke `loom-verify`'s rsync (fixed: `--owner`).
  6. **The safety classifier was briefly unavailable;** I verified the phase B scope with read-only tools and an independent reviewer.
- **Worth changing:**
  - Protect future `*/testing.py` fake modules up front.
  - Add new service test dirs to `pytest.ini` before locking.
  - Give agents a WSL runner script instead of hand-quoted `wsl ... bash -lc`.
  - `loom-verify` could exclude `/.dev`.

## 2026-11-16 — this turn, short form (21 agents)

- **Task:** all of M2 from the spec's §14 plan, end to end.
- **Size:** Large.
- **Agents:**
  - test-writers ×5 (wave 1, 2a scripts, 2a lifecycle, 2b, wave 3);
  - test reviewers ×5;
  - implementers ×4 (wave 1, 2a in phases A and B, 2b, wave 3);
  - code reviewers ×4 plus a final whole-branch review;
  - a verifier.
- **Worktrees:** 2a and 2b ran in their own worktrees.
- **Went wrong:**
  - the stop gate gave up during waits for agents;
  - test fakes were blocked from the unprotected `testing.py`;
  - a stray `C:` dir appeared in the repo;
  - an agent briefly stopped the user's Postgres and Redis containers (restored);
  - devcerts file ownership broke `loom-verify`'s rsync;
  - the safety classifier was briefly unavailable.
- **Caught by review and verification:**
  - pods that could run twice or never be terminated (listing lag, ambiguous creates, task exceptions);
  - the dev placeholder bundle reaching paid pods;
  - a hung SSH call stalling the tick;
  - flaky test-harness time;
  - a vacuous allowlist test;
  - the missing lease-dir creation.
- **Next time:**
  - protect fake modules and add test paths to `pytest.ini` before locking;
  - use one WSL runner script for all agents;
  - start implementers only after their tests' review, unless the gate would otherwise give up.

## 2026-11-17 — before-M3 review fixes (plan §13 R1-R6, R8, R9, R11-R14), test phase

- **Setup:** Medium x3 (gateway, billing, web), each on its own branch in worktrees; spec `docs/m3-review-fixes-spec.md` approved first. Agents so far (all Opus): 4 test-writers (one per area, plus one for gateway review gaps), 3 test reviewers, 2 test revisers (billing, web). Implementers not started (waiting for the user's lock).
- **Review caught (one major per area):** no test of the G4 clamp to the reservation; the B3 concurrent role-setup test could never hit "role already exists" because roles are server-wide and never dropped (fixed with configurable role names plus a barrier, user decision); the W1 limit tests skipped in CI (no Redis in the `web` job), so deleting the limit would have passed.
- **Went wrong:** the billing test writer left a `loom_app` role on the user's dev Postgres, which made 13 B3 tests pass vacuously (dropped); existing tests contradicted the spec (gateway timeout, web signin-action/auth) and had to be changed before the lock; `apps/web/src/**/*.test.ts` was not in `protect.paths`; Node is not installed (web runs in a throwaway Docker runner); a PowerShell commit message with quotes failed (use `-F file`); `git add -A` in a worktree was refused by the guard (name files).
- **Next time:** check `protect.paths` covers every test location before writing tests; have test writers drop server-wide state they create; check CI has the services a new test needs.

## 2026-11-17 — before-M3 fixes, implementer start (blocked)

- **Task:** start 4 implementers (gateway Sonnet and Opus arms, billing, web) after the user locked the tests.
- **Setup:** 6 implementer agents, each with worktree isolation and told to `git switch` to its branch: 4 at first, then fresh gateway arms for fairness.
- **Went wrong:** with the tests locked, the guard refuses `git switch` in a worktree, because it moves the locked test files. All 4 stopped correctly without working around it. Worktrees of agents that end without changes are auto-removed, so "stop, then the user switches, then resume" cannot work. The lock was first hand-edited over Remote Control; the edit was held until a typed `/agent-setup` command, and a probe file showed the lock was not in force.
- **Next time:** put each agent on its branch before locking (switch while unlocked, then lock, then resume), or lock only after the worktrees exist on their branches. Probe the lock before relying on it.

## 2026-11-17 — before-M3 review fixes: Sonnet vs Opus implementer (gateway, same task)

- **Design:** both models built G1-G5 from the same locked tests (edc2357) with word-for-word identical briefs (only branch, cores 0-3 / 4-7 and Redis DB differed). Each build got an identical blind Opus review (neutral worktree name, diff only). All other agents were Opus; no effort overrides.
- **Results (first implementation):**

| | Sonnet (f1809be) | Opus (197783c) |
|---|---|---|
| Harness tokens | 153,636 | 194,624 |
| Tokens summed over requests (all / without failed runner calls) | 5,573,465 / 5,332,806 | 7,291,364 / 7,124,759 |
| Output tokens | 37,449 | 55,162 |
| Runner retries, exit 124 / 125 (124 = my brief's bad timeoutMs) | 2 / 0 (240,659 tokens) | 1 / 0 (166,605 tokens) |
| Tool uses / diff | 71 / +215 −55 | 92 / +323 −97 |
| Tests | 307 passed, no stops | 307 passed, no stops, ran its own 3× repeat under load |
| First blind review | changes needed; excluding the status-lines major (out of both briefs): 3 doc/comment issues of its own + shared ones | changes needed; excluding the same: 1 code-behaviour issue of its own (shutdown during the final settle drops `[DONE]`) + shared ones |
| Fix rounds after the pick | 2 (G2a + review items: 80,015 harness tokens; retry pin: 50,972) | none (not picked) |
| Verifier | partial: one flaky G2a test, caused by the test harness (uvicorn keep-alive), compose M1 14/14, M2 5/5 | not run (not picked) |
| Wall time | 526 s (unreliable: shared runner queue) | 1189 s (unreliable) |

- **What decided the pick:** one finding that the reviewer found only by reading code, not reproduced at review time. A scratch test then reproduced it 10/10 on the Opus build and 0/10 on the Sonnet build. Without that repro the builds were close: neither had a correctness blocker. Sonnet used ~25% fewer tokens and wrote a smaller diff. Opus wrote more complete docs and checked itself more.
- **Caveats:** one task, one run per model; two different reviewer instances; reviewers were Opus (same family as one arm); billing's unpinned compose stack may have loaded either arm's test runs; wall times include runner queueing.
- **Process notes:** a locked `git switch` blocks worktree agents (fix: `git worktree add` by the orchestrator, then agents run without isolation, with `run_in_wsl` cwd set); the spec's mechanisms I wrote failed review three times on billing B1 (each fix found by a fresh reviewer with a concrete Redis trace), so give designs with concurrency a review before tests are written; three test-review rounds each caught a test no correct code could pass or one that could never fail.

## 2026-11-17 — before-M3 fixes, final stretch (12 agents this turn)

- **Task:** finish billing B1/B3, verify gateway and billing, merge the spec, open PRs #12-#14 (not merged).
- **Setup:** Medium x3 in git worktrees made by the orchestrator. Agents (Opus unless noted): 2 billing test-writers, 1 billing fix round (B1 token, retry-all, api exit), 2 billing code reviewers, 2 billing verifiers (full compose as loom_app, M1/M2), 1 gateway fix round (Sonnet, retry pin), 1 gateway fix-round reviewer, 1 gateway verifier, 1 billing test reviewer.
- **Went wrong / redone:** the B1 heal mechanism in my spec failed review three times (different-P double heal, re-heal after a reload, busy org never healed) before the random heal token. B3 first made misconfiguration an outage, then a short retry list made DNS/recovery fatal. A verifier ran on a HEAD that moved under it (had to re-verify). The gateway cap test is flaky (test-harness keep-alive); fixing it needs an unlock.
- **Caught by review/verify:** every B1/B3 flaw above, with a concrete Redis or psycopg trace; the api exiting 3 not 2; compose-level checks (loom_app sessions, ledger guards refused, prod start with Postgres down).
- **Next time:** review concurrency designs before writing tests; give verifiers a fixed commit; probe the lock and worktree setup before starting implementers.

## 2026-11-18 — before-M3 fixes, second session: locked-test fixes, follow-on code, merges #14, #13, #12

- **Task / size:** Medium x3, on the existing worktrees. Handoff steps 2, 3 and the stale-folder part of 5, plus the protect.paths commit.
- **Agents (model named on every spawn):**
  - Opus test-writers ×9: 3 first passes, 2 gateway flake fixes, 2 billing review rounds, the exit-2 regression test plus its fix, and the G3 flake.
  - Sonnet implementers ×2 (billing marker list, web no-token path).
  - Opus implementers ×2 (web W1 unhandled rejection, billing exit-2 precedence).
  - Opus reviewers ×13 and Opus verifiers ×6.
- **Caught by review/verify:**
  - Two flakes inside the new gateway tests: a proxy port rebind race, and a NOSCRIPT retry counted as a second release. The second showed only on CI's fresh Redis, in session 1.
  - A billing race test that timed out under load and leaked roles.
  - A W1 unhandled rejection from @auth/core's late `Promise.all`.
  - A failed role gate exiting 3 when uvicorn's bind failed.
  - A held-port test that didn't check the shutdown ran.
  - A merge that made the contracts doc name two 503 codes for one case.
  - A test made flaky by the merge (it relied on a warm auth cache).
- **Went wrong:**
  - My `pkill -f` pattern matched its own command line. Use `yes` hogs and `pkill -x yes`.
  - `run_in_wsl` refuses `timeoutMs` above 600000.
  - In a `cp -r` copy of the WSL mirror, `uv run pytest` runs the mirror's code (venv shebangs). Use `uv run python -m pytest`.
  - The docs have CRLF endings in the worktrees, which broke a conflict-resolver script twice.
  - The guard (correctly) refuses agent edits of `.claude/agent-setup.json`; the user made them.
- **Next time:**
  - Run each new test file under the flake-check recipe before the first push, and once more after merging main into a branch.
  - Write merge-resolution scripts CRLF-aware from the start.
