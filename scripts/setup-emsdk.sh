#!/usr/bin/env bash
# Installs the pinned Emscripten SDK into .tools/emsdk (repo-local, gitignored).
# Usage: scripts/setup-emsdk.sh            (idempotent; re-run after bumping EMSDK_VERSION)
set -euo pipefail

EMSDK_VERSION="${EMSDK_VERSION:-6.0.13}"   # keep in sync with README.md
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EMSDK_DIR="$ROOT/.tools/emsdk"

if [ ! -d "$EMSDK_DIR/.git" ]; then
  mkdir -p "$ROOT/.tools"
  git clone --depth 1 --branch "$EMSDK_VERSION" https://github.com/emscripten-core/emsdk.git "$EMSDK_DIR"
else
  git -C "$EMSDK_DIR" fetch --depth 1 origin tag "$EMSDK_VERSION"
  git -C "$EMSDK_DIR" checkout -q "$EMSDK_VERSION"
fi

"$EMSDK_DIR/emsdk" install "$EMSDK_VERSION"
"$EMSDK_DIR/emsdk" activate "$EMSDK_VERSION"

"$EMSDK_DIR/upstream/emscripten/emcc" --version | head -1
echo "emsdk $EMSDK_VERSION ready. The wasm/Makefile picks it up automatically;"
echo "for an interactive shell: source $EMSDK_DIR/emsdk_env.sh"
