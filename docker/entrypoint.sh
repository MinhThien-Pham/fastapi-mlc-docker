#!/usr/bin/env bash
set -euo pipefail

BAKED="${BAKED_MLC_CLI_PATH:-/opt/mlc-cli}"
WORK="${MLC_CLI_PATH:-/workspace/mlc-cli}"

if [ ! -d "$BAKED/.git" ]; then
  echo "[BOOT] ERROR: baked mlc-cli repo missing at $BAKED" >&2
  exit 1
fi

mkdir -p "$WORK"

for dir in models dist wheels; do
  mkdir -p "$WORK/$dir"
done

echo "[BOOT] Syncing baked mlc-cli source from $BAKED to $WORK"

rsync -a --delete \
  --exclude '/models/' \
  --exclude '/dist/' \
  --exclude '/wheels/' \
  --exclude '/mlc-llm/' \
  --exclude '/tvm/' \
  "$BAKED/" "$WORK/"

for dir in models dist wheels; do
  mkdir -p "$WORK/$dir"
done

echo "[BOOT] Baked mlc-cli ref: $(cat /opt/mlc-cli-ref.txt 2>/dev/null || echo unknown)"
echo "[BOOT] Workspace mlc-cli head: $(git -C "$WORK" rev-parse HEAD 2>/dev/null || echo unknown)"

export MLC_CLI_PATH="${MLC_CLI_PATH:-/workspace/mlc-cli}"
export TVM_SOURCE="${TVM_SOURCE:-bundled}"

case "$TVM_SOURCE" in
  bundled)
    export TVM_HOME="$MLC_CLI_PATH/mlc-llm/3rdparty/tvm"
    export PYTHONPATH="$TVM_HOME/python${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$MLC_CLI_PATH/mlc-llm/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  relax|custom)
    export TVM_HOME="$MLC_CLI_PATH/tvm"
    export PYTHONPATH="$TVM_HOME/python${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$TVM_HOME/build/lib:$MLC_CLI_PATH/mlc-llm/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  *)
    echo "[BOOT] Unsupported TVM_SOURCE=$TVM_SOURCE"
    exit 1
    ;;
esac

echo "[BOOT] TVM_SOURCE=$TVM_SOURCE"
echo "[BOOT] TVM_HOME=$TVM_HOME"
echo "[BOOT] PYTHONPATH=$PYTHONPATH"
echo "[BOOT] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

exec conda run --no-capture-output -n mlc-cli-venv \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
