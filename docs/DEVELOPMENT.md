# Developer Guide

This document covers local setup, testing, candidate verification, updating the mlc-cli pin, CI workflows, and contribution notes.
For the API endpoint reference, see [API_ENDPOINTS.md](API_ENDPOINTS.md).

---

## Development setup

**Requirements:**

- Docker + Docker Compose v2.x
- Python 3.10+
- NVIDIA GPU + drivers + NVIDIA Container Toolkit (required for GPU-backed flows)
- `git`

**Local Python environment (for tests only):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running tests

### Unit and integration tests

```bash
pytest tests/unit/ tests/integration/ -q
```

With coverage:

```bash
pytest tests/unit/ tests/integration/ -v --cov=app --cov-report=term-missing
```

**Test layers:**

| Layer | What it covers | Requires Docker/GPU? |
|---|---|---|
| `tests/unit/` | Service logic, command builders, helpers | No |
| `tests/integration/` | API lifecycle, chat flow, architecture contract | No (mocked) |
| Smoke/full integration scripts | Real API server validation | Yes (live container) |
| Docker/GPU validation | Real build, quantize, compile, run | Yes |

---

## Smoke and full integration tests

These are standalone scripts that run against a **live API server**.
They are not collected by pytest — run them directly.

**Smoke test** (quick integration validation):

```bash
API_URL=http://localhost:8000 python tests/integration/test_smoke.py
```

**Full pipeline test** (broader validation including quantize, compile, run):

```bash
API_URL=http://localhost:8000 python tests/integration/test_full_pipeline.py
```

Set `API_URL` to point at any running API container. The default is `http://localhost:8000`.

Useful environment variables for the full pipeline test:

| Variable | Purpose |
|---|---|
| `FULL_RAW_MODEL` | Path to raw model weights (skips auto-download if set) |
| `FULL_CONV_TEMPLATE` | Conversation template (auto-detected from model name if omitted) |
| `FULL_QUANT` | Quantization format (default: `q4f16_1`) |
| `FULL_DEVICE` | Target device (default: `cuda`) |
| `FULL_BUILD_ACTION` | Build action (default: `install-wheels`) |
| `DOWNLOAD_RUN_MODEL_IF_MISSING` | Set to `1` to auto-download TinyLlama if no local model is found |
| `CLEANUP_FULL_MODEL` | Set to `1` to delete the auto-downloaded raw model after the test |

Both tests remain valid and are not obsolete. **The candidate verification flow runs both before allowing promotion.**

---

## Candidate mlc-cli verification

Use this flow to safely test a newer mlc-cli commit before updating `docker/mlc-cli.lock`.

```bash
python scripts/verify_mlc_cli_candidate.py
```

**What it does:**

1. Reads `MLC_CLI_REPO` and `MLC_CLI_REF` from `docker/mlc-cli.lock`
2. Fetches the configured repo's current `HEAD` (no clone)
3. If HEAD matches the pinned ref, exits cleanly — nothing to do
4. If HEAD differs:
   - Builds a candidate Docker image with the new ref
   - Starts an isolated candidate container on a separate port (default 8001)
   - Runs smoke and full integration tests against the candidate API
5. If both pass, offers a series of optional promotion steps
6. If either fails, leaves the candidate container running for debugging and prints connection details

**Post-pass options (prompted interactively):**

| Prompt | Default | What it does |
|---|---|---|
| Promote this tested ref? | N | Updates `MLC_CLI_REF` in `docker/mlc-cli.lock` to the tested SHA |
| Use candidate container as main local image? | Y | `docker commit` snapshots the tested container to `fastapi-mlc-docker-web:latest` |
| Import tested artifacts into main workspace? | Y | Replaces `dist/`, `wheels/`, `mlc-llm/`, `tvm/` in the main workspace with candidate artifacts |
| Commit and push `docker/mlc-cli.lock`? | N | Stages and commits only the lock file, then pushes if an upstream is set |

Each prompt accepts `?` to print a full explanation before answering.

**Re-open post-pass options without rerunning tests:**

```bash
python scripts/verify_mlc_cli_candidate.py finalize
```

This reads the saved state from `.mlc-cli-verify/last-pass.json` and re-opens the promotion prompts, provided the candidate image/container/volume still exist.

**Candidate isolation:**

- The candidate container runs on port 8001 (or the next free port) — separate from the main container
- The candidate workspace volume (`fastapi_mlc_verify_<sha12>`) is separate from the main workspace
- Nothing in the main container or main workspace is touched until after tests pass and you explicitly approve

---

## Updating `docker/mlc-cli.lock` safely

`docker/mlc-cli.lock` is the single source of truth for which mlc-cli source is included in the Docker image.

```
MLC_CLI_REPO=https://github.com/ballinyouup/mlc-cli.git
MLC_CLI_REF=<pinned commit SHA>
```

**Update process:**

1. Check the weekly CI summary (see below) or run the candidate script manually
2. Run `python scripts/verify_mlc_cli_candidate.py`
3. If both smoke and full tests pass, approve the lock file update when prompted
4. Review the diff: `git diff docker/mlc-cli.lock`
5. If committing manually: `git add docker/mlc-cli.lock && git commit -m "chore: update pinned mlc-cli ref"`
6. Rebuild the Docker image: `docker compose build`
7. Run `/repo-status` and `/setup-check` against the new image

**Never update `docker/mlc-cli.lock` without a passing candidate verification run.**

---

## CI workflows

### Fast CI (`ci.yml`)

Runs on every push and pull request.

```
pytest tests/ -v --cov=app --cov-report=term-missing
```

This covers unit tests, integration lifecycle tests, and architecture contract tests.
It does not require Docker or a GPU.

### Weekly mlc-cli update check (`upstream-drift.yml`)

Runs automatically every Monday at 10 AM EDT. Can also be triggered manually from the Actions tab.

**What it does:**

1. Reads `MLC_CLI_REPO` and `MLC_CLI_REF` from `docker/mlc-cli.lock`
2. Fetches the configured repo's current `HEAD` using `git ls-remote`
3. If HEAD matches the pinned ref: logs that no update is available and exits
4. If HEAD differs:
   - Clones the configured repo at HEAD
   - Runs `tests/upstream/check_cli_contract.py` against the clone

**Contract check results:**

| Result | What it means | Action |
|---|---|---|
| **Pass** | The CLI interface appears unchanged (commands, flags, scripts) | Run local candidate verification before updating the lock |
| **Fail** | The configured repo changed in ways that may break the wrapper | Investigate the diff, update wrapper code if needed, do not update lock yet |
| **Inconclusive** | The contract checker could not complete | Check the workflow logs; treat with caution |

**Important:** A contract check pass is an early signal only. It does **not** prove the candidate is safe.
You must still run the full local candidate verification before updating `docker/mlc-cli.lock`.

**On contract failure:** A GitHub issue is opened with the diff link and a checklist of what to investigate.

---

## Contract check

The contract check (`tests/upstream/check_cli_contract.py`) inspects the configured mlc-cli repo for:

- Expected subcommands (`build`, `quantize`, `compile`, `run`)
- Expected CLI flags used by the wrapper
- Expected script files and their structure

It is a lightweight static check. It does not run the mlc-cli tool or build anything.

A pass means the surface-level interface is recognizable. The actual behavior may still have changed in ways the contract check cannot detect. That is why local candidate verification (which runs the real Docker build and smoke/full integration tests) is required before promotion.

---

## Docker image and workspace notes

**Image layout:**

```
/opt/mlc-cli          — mlc-cli source included at Docker build time (read-only)
/opt/mlc-cli-ref.txt  — pinned commit SHA
/opt/mlc-cli-repo.txt — pinned repo URL
/workspace/mlc-cli    — writable runtime workspace (synced from /opt/mlc-cli by entrypoint)
```

**Artifact directories (preserved across container restarts):**

```
/workspace/mlc-cli/models/    — raw downloaded model weights
/workspace/mlc-cli/dist/      — quantized models and compiled libraries
/workspace/mlc-cli/wheels/    — built Python wheels
/workspace/mlc-cli/mlc-llm/   — MLC-LLM source tree
/workspace/mlc-cli/tvm/       — TVM source tree
```

The entrypoint syncs `/opt/mlc-cli` → `/workspace/mlc-cli` on startup, preserving the above directories.
This means source files are refreshed from the pinned image source, while expensive artifacts are kept.

**Updating mlc-cli** requires editing `docker/mlc-cli.lock` and rebuilding the Docker image.
The container does not fetch, pull, or clone mlc-cli at runtime.

**Checking source status:**

```bash
curl -s http://localhost:8000/repo-status | python3 -m json.tool
```

The `workspace_matches_baked` field confirms that the runtime workspace matches the included image source.

---

## Artifact import (candidate verification)

After a candidate verification pass, the script can optionally import tested artifacts from the candidate workspace into the main workspace:

- `dist/` — replaces (not merges) with candidate artifacts
- `wheels/` — replaces with candidate wheels
- `mlc-llm/` — replaces with candidate MLC-LLM build
- `tvm/` — replaces with candidate TVM build
- `models/` — skipped by default (large and source-independent)

Before replacing, you are prompted to optionally back up the existing main artifact directories.
If a backup is made and the import fails, automatic rollback is attempted.

After a successful import, the candidate volume is no longer needed.
The script prompts to clean it up (default: remove if artifacts were imported).

---

## Cleanup guidance

**After a successful candidate verification and promotion:**

- Candidate container: removed automatically if you chose to snapshot it as the main image
- Candidate image (`fastapi-mlc-docker-web:verify-<sha12>`): kept by default for local reference
- Candidate volume (`fastapi_mlc_verify_<sha12>`): removed by default if artifacts were imported

**After a failed candidate verification:**

- Candidate container is left running for debugging
- Connect with: `docker exec -it fastapi-mlc-verify-<sha12> bash`
- API is accessible at: `http://localhost:<port>`
- Clean up manually when done:
  ```bash
  docker rm -f fastapi-mlc-verify-<sha12>
  docker volume rm fastapi_mlc_verify_<sha12>
  docker rmi fastapi-mlc-docker-web:verify-<sha12>
  ```

**Candidate verification state** is saved to `.mlc-cli-verify/last-pass.json` (gitignored).
Use `python scripts/verify_mlc_cli_candidate.py finalize` to re-open post-pass prompts without rerunning tests.

---

## Contribution checklist

Before opening a pull request:

- [ ] Add or update tests for changed behavior
- [ ] Run `pytest tests/unit/ tests/integration/ -q` and confirm all pass
- [ ] Update `docs/API_ENDPOINTS.md` if any endpoint signature, field, or behavior changed
- [ ] Update this file if any dev workflow changed
- [ ] For Docker or mlc-cli changes: run the full local candidate verification
- [ ] For `docker/mlc-cli.lock` updates: candidate verification must pass before merging
- [ ] For GPU-dependent changes: document the manual validation you performed
