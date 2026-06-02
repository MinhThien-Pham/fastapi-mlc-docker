# Architecture Note: Build-time Baked `mlc-cli` Runtime

## Purpose

This note records the architecture direction for making `fastapi-mlc-docker` a clean Docker wrapper around `mlc-cli`.

The core decision is:

```text
`mlc-cli` is the primary tool dependency of `fastapi-mlc-docker`.
Therefore, `mlc-cli` should be cloned and pinned during Docker image build, not fetched later by the running API service.
```

The Docker image should contain the exact `mlc-cli` source version it was built against, create the runtime environment from the version information declared by `mlc-cli`, and start FastAPI with that environment already available.

Generated artifacts should remain outside the immutable image and persist in Docker volumes.

---

## Final Direction

The desired architecture is:

```text
Docker build:
  clone mlc-cli at a pinned SHA
  read mlc-cli/scripts/config/versions.sh
  create the conda runtime using the Python version declared by mlc-cli
  install FastAPI dependencies into that same environment

Container runtime:
  start FastAPI in the mlc-cli runtime environment
  do not clone mlc-cli from the network
  use a persistent writable workspace for generated artifacts
  keep build outputs, models, wheels, TVM, and MLC-LLM state across image rebuilds
```

This makes the wrapper deterministic and keeps the dependency relationship clear:

```text
fastapi-mlc-docker = API wrapper
mlc-cli            = pinned tool/runtime dependency
artifacts          = user-generated state
```

---

## Why the Architecture Changed

The initial wrapper design populated `mlc-cli` at runtime because the early goal was to quickly expose CLI behavior through HTTP endpoints.

At that stage, the project was still proving the basic flow:

```text
/build
/quantize
/compile
/run
```

The runtime-populated model made iteration convenient while `mlc-cli` itself was still being debugged.

The deeper architecture issue became visible when the in-process chat path was tested.

The command-driven `/run` path worked because it executed through the `mlc-cli` runtime environment. The `/chat/load` path required FastAPI itself to import `mlc_llm`. That exposed a mismatch: FastAPI and `mlc-cli` were not using the same Python runtime.

The fix was to run FastAPI inside the same conda environment used by `mlc-cli`. That established the real runtime contract:

```text
FastAPI and mlc-cli must share one runtime environment.
```

After that, the next issue became clear: the Dockerfile still needed to know which Python version `mlc-cli` expects.

`mlc-cli` already owns that source of truth in:

```text
mlc-cli/scripts/config/versions.sh
```

The wrapper should consume that source of truth directly. It should not duplicate or hardcode the Python version independently.

That requirement cannot be cleanly satisfied if `mlc-cli` only appears after the container starts. The Dockerfile needs `mlc-cli` during image build so it can read `versions.sh` before creating the conda environment.

This is the reason for moving from runtime-populated `mlc-cli` to build-time baked `mlc-cli`.

---

## Problems with Runtime-populated `mlc-cli`

The runtime-populated model has several structural problems.

### 1. The Docker image is incomplete

A Docker image should contain its primary application dependencies.

If the image starts without `mlc-cli` and the API later clones it through an endpoint, then the image is not fully defined at build time.

That makes the service less reproducible and harder to reason about.

### 2. Runtime clone creates unnecessary network dependency

The API service should not need internet access just to obtain its own tool dependency after startup.

Runtime network access is acceptable for operations that inherently fetch external data, such as downloading a new model or building missing upstream dependencies. It should not be required simply to make the API service usable.

### 3. Python/toolchain versions can drift

If the Dockerfile independently creates a Python environment and `mlc-cli` later expects a different Python version, the failure happens too late.

Example:

```text
mlc-cli updates its expected Python version
Docker image still creates the old Python version
mlc-cli builds wheels for the new ABI
FastAPI runs with the old ABI
pip install or import fails during runtime build/chat flow
```

The correct fix is to create the environment from `mlc-cli`'s own version file during image build.

### 4. A persistent repo volume can hide source updates

If the entire `mlc-cli` repo lives in a Docker volume, that volume can outlive image rebuilds.

A new FastAPI image may still execute an old `mlc-cli` checkout from the volume.

That breaks the expectation that updating the image updates the tool dependency.

### 5. Guardrails detect symptoms, not the root cause

Runtime checks can report Python mismatch, missing `mlc_llm`, or stale wheels.

Those checks are still useful, but they do not fix the underlying inversion:

```text
The Docker image does not own the mlc-cli dependency at build time.
```

The final architecture should remove that inversion.

---

## Desired Build-time Model

The image build should be responsible for the source and environment.

```text
docker compose build
  -> clone mlc-cli from MLC_CLI_REPO
  -> checkout MLC_CLI_REF
  -> read scripts/config/versions.sh
  -> create conda env with mlc-cli's declared Python version
  -> install FastAPI requirements into that env
```

After build succeeds, the container has:

```text
a pinned mlc-cli source
a matching Python runtime
FastAPI installed in the same runtime
no need to clone mlc-cli at startup
```

This makes build failure honest. If the image cannot clone `mlc-cli` or cannot create the required runtime, the image build should fail.

That is preferable to producing an image that starts but later fails when the API tries to clone or align its main dependency.

---

## Runtime Model

At runtime, the container should already have the required `mlc-cli` source and environment.

Runtime should manage only:

```text
workspace state
model files
build outputs
wheels
TVM source/build state
MLC-LLM source/build state
compiled libraries
```

The API should not clone `mlc-cli` at runtime.

Normal local operations should work without network after the needed artifacts exist:

```text
inspect setup state
compile an existing quantized model
run an existing compiled model
load chat engine
serve chat completions
```

Network may still be needed when the user explicitly performs network-dependent tasks:

```text
download a new model
build missing MLC-LLM/TVM dependencies
fetch external packages or sources not already present
```

This is acceptable because those operations are user-triggered and inherently depend on external resources.

---

## Directory Model

Use two separate concepts:

```text
baked source
persistent workspace
```

### Baked Source

```text
/opt/mlc-cli
```

This is cloned during Docker build from the pinned `MLC_CLI_REF`.

It represents the source version used to create the image and runtime environment.

It should not be hidden by a Docker volume in the default user workflow.

### Persistent Workspace

```text
/workspace/mlc-cli
```

This is the writable runtime workspace.

It should persist in a Docker volume and contain generated or downloaded state:

```text
/workspace/mlc-cli/models
/workspace/mlc-cli/dist
/workspace/mlc-cli/wheels
/workspace/mlc-cli/mlc-llm
/workspace/mlc-cli/tvm
```

The wrapper should run `mlc-cli` commands from this workspace so behavior stays close to native CLI usage.

---

## Why Use a Persistent Workspace

If generated artifacts are written only into the container writable layer, they can disappear when the container is removed or recreated.

That is not acceptable because these artifacts are expensive to regenerate:

```text
downloaded model files
quantized model outputs
compiled CUDA libraries
built Python wheels
TVM source/build output
MLC-LLM source/build output
```

Docker images should define source and environment.

Docker volumes should preserve user-generated state.

---

## Startup Sync

At container startup, the baked source should be synchronized into the persistent workspace.

Conceptually:

```text
sync /opt/mlc-cli -> /workspace/mlc-cli
preserve artifact directories
start FastAPI
```

The sync must preserve:

```text
models/
dist/
wheels/
mlc-llm/
tvm/
```

It must not run destructive cleanup such as:

```text
git clean -fdx
rm -rf models dist wheels mlc-llm tvm
```

It must not clone `mlc-cli` from the network.

The goal is:

```text
source files match the baked image version
artifact directories remain untouched
FastAPI starts with a local mlc-cli workspace ready to use
```

This keeps the runtime close to native usage while still letting Docker define the source and environment.

---

## Why Not Mount the Whole Repo as a Volume

Mounting a volume over the entire `mlc-cli` repo makes the volume version win over the image version.

That creates the same problem as the old runtime-populated model:

```text
image says it represents one mlc-cli version
volume may contain another
runtime executes the volume version
```

The image must retain control over the pinned source.

A persistent workspace is acceptable only if it is synchronized from the baked source and artifact directories are preserved intentionally.

---

## Why Not Store Artifacts Only Inside the Image

Artifacts should not be baked into the image because they are user state.

They may be large, hardware-specific, model-specific, or generated after the image is built.

Examples:

```text
models/
dist/
wheels/
mlc-llm/
tvm/
```

The image should be rebuildable without deleting these directories.

---

## Why Not Rely on Runtime Repo Management

Runtime repo management makes the API responsible for fetching and aligning its own dependency.

That design creates unclear responsibility:

```text
Dockerfile creates one environment
runtime endpoint clones another repo
volume may contain a third state
```

The final design should be simpler:

```text
MLC_CLI_REF changes -> rebuild image
FastAPI code changes -> rebuild image
Artifacts remain -> Docker volume
```

---

## Update Workflow: FastAPI Wrapper

When only the FastAPI wrapper changes:

```text
git pull
docker compose build web
docker compose up -d web
```

If `MLC_CLI_REF` does not change:

```text
mlc-cli source version remains the same
existing models remain
existing quantized outputs remain
existing wheels remain
existing compiled libraries remain
```

A new image may contain a fresh conda environment. If `mlc_llm` is not installed in that fresh environment but a compatible wheel exists in the workspace, reinstalling from existing wheels should be enough.

No model download, quantization, or compilation should be required unless the existing artifact is missing or incompatible.

---

## Update Workflow: `mlc-cli`

`fastapi-mlc-docker` should expose the pinned `mlc-cli` version through build configuration:

```text
MLC_CLI_REPO=https://github.com/MinhThien-Pham/mlc-cli.git
MLC_CLI_REF=<full 40-character commit SHA>
```

When updating `mlc-cli`:

```text
update MLC_CLI_REF
docker compose build web
docker compose up -d web
```

The new image will:

```text
clone the new mlc-cli source
read the new versions.sh
create the correct runtime environment
sync source into the persistent workspace
preserve artifacts
```

After a `mlc-cli` update, artifacts should be evaluated as follows:

| Artifact | Reuse expectation | Notes |
|---|---|---|
| `models/` | Reusable | Raw model files are usually independent of wrapper/tooling changes. |
| `dist/<model>-MLC/` | Usually reusable | Re-quantize if MLC-LLM changes format expectations. |
| `wheels/` | Conditional | Must match Python ABI, platform, and active MLC-LLM ref. |
| `dist/libs/*.so` | Conditional / often rebuild | Compiled libraries are tied to TVM/MLC runtime and should be recompiled after toolchain changes. |
| `tvm/` | Conditional | May need rebuild if TVM ref changes. |
| `mlc-llm/` | Conditional | May need rebuild if MLC-LLM ref changes. |

The system should detect and report stale or incompatible artifacts.

It should not delete or rebuild them automatically.

---

## Endpoint Direction

### Replace Runtime Repo Clone

The API should not need a write endpoint whose job is to clone `mlc-cli`.

Preferred direction:

```text
GET /repo-status
```

It should report:

```text
baked mlc-cli ref
workspace mlc-cli ref
baked source path
workspace path
whether workspace matches baked source
whether dev mode is active
```

If compatibility requires keeping the old endpoint temporarily, it should return a deprecation message explaining that `mlc-cli` is baked into the image and can be changed only by updating `MLC_CLI_REF` and rebuilding the image.

### Keep Command Endpoints

The main command endpoints remain:

```text
/build
/quantize
/compile
/run
/chat/load
/chat/completions
```

They should run against:

```text
/workspace/mlc-cli
```

This keeps behavior close to native `mlc-cli`.

### Strengthen Setup Diagnostics

`/setup-check` should be read-only and report:

```text
FastAPI wrapper commit
baked mlc-cli ref
workspace mlc-cli ref
Python runtime version
Python expected version from versions.sh
whether Python matches
whether mlc_llm is importable
available wheels
wheel ABI match
artifact directories present
compiled libraries present
CUDA availability
Go version
Conda version
warnings for stale wheels or compiled libraries
suggested user actions
```

It should never delete, rebuild, clone, or mutate state.

---

## Behavior Compared with Native `mlc-cli`

Native usage:

```text
user clones mlc-cli
user creates environment
user builds dependencies
user runs commands
artifacts stay on the host filesystem
```

Docker usage with this architecture:

```text
image clones mlc-cli
image creates environment
container runs commands
artifacts stay in a Docker volume
```

The goal is to preserve the useful behavior of native usage while making source, environment, and service runtime reproducible across machines.

Docker does not make `mlc-cli` more powerful. It makes the `mlc-cli` + FastAPI runtime more consistent, portable, and easier to reproduce.

---

## Development Workflow

Default user workflow should use the pinned baked `mlc-cli` source.

For active `mlc-cli` development, use an explicit dev mode.

A dev compose file may bind-mount a local `mlc-cli` checkout, but it must be opt-in and clearly reported by `/setup-check`.

Default mode should remain reproducible and pinned.

---

## Linux and macOS Runtime Profiles

This Docker architecture targets:

```text
Linux CUDA
Windows WSL2 with NVIDIA Docker support
GPU servers with NVIDIA Container Toolkit
```

macOS Apple Silicon Metal should be treated as a separate native profile.

Docker Desktop on macOS does not provide the same Metal GPU path that NVIDIA Docker provides for CUDA.

Do not try to make the Linux CUDA Dockerfile also serve as the macOS Metal runtime.

---

## Implementation Principles

The implementation should follow these rules:

```text
Do not modify mlc-cli for wrapper convenience.
Do not clone mlc-cli at runtime.
Do not hardcode Python version independently from mlc-cli.
Do not mount a volume over the baked source path in default mode.
Do not delete artifacts automatically.
Do not use git clean -fdx on the workspace.
Do not silently rebuild expensive artifacts.
Do not mix macOS Metal support into the Linux CUDA Docker architecture.
```

The service should detect mismatches and report clear suggested actions.

Examples:

```text
mlc_llm not installed but compatible wheel exists -> suggest install-wheels
wheel ABI mismatch -> suggest full build
compiled library may be stale -> suggest recompile
missing model -> suggest download/quantize path
```

---

## Validation Before Implementation

Before the wrapper architecture is refactored, the direct `mlc-cli` UX should be fully validated.

Validate:

```text
direct build flow
direct install-wheels flow
direct quantize flow
direct compile flow
direct run flow
interactive menu flow if supported
non-interactive command flow
missing wheel error behavior
missing compiled library error behavior
TVM_SOURCE=bundled behavior
TVM_SOURCE=relax/custom behavior if supported
clean rebuild behavior
```

The wrapper should be built on top of a stable CLI contract.

---

## Final Decision

The final architecture is:

```text
mlc-cli is cloned and pinned during Docker build
Python/runtime environment is created from mlc-cli versions.sh
FastAPI runs inside that environment
runtime clone is removed
workspace/artifacts persist in Docker volumes
updates happen through image rebuilds
diagnostics report state without mutating it
```

This design makes `fastapi-mlc-docker` a real Dockerized wrapper around `mlc-cli` instead of a runtime repo manager.
