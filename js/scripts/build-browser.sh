#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"
SYNC_ENTRY="${PROJECT_DIR}/src/browser.sync.js"
ASYNC_ENTRY="${PROJECT_DIR}/src/browser.async.js"
REQUIRED_WASM=(
  "${PROJECT_DIR}/wasm/core.sync.js"
  "${PROJECT_DIR}/wasm/core.async.js"
)

for wasm_entry in "${REQUIRED_WASM[@]}"; do
  if [ ! -f "$wasm_entry" ]; then
    echo "ERROR: ${wasm_entry} not found"
    echo "Run 'make -C .. wasm-pkg' first."
    exit 1
  fi
done

PKG_NAME=$(node -e "
  const p = require('${PROJECT_DIR}/package.json')
  const name = p.name.replace(/^@/, '').replace(/[\\/-](\\w)/g, (_, c) => c.toUpperCase())
  console.log(name)
")

if command -v esbuild >/dev/null 2>&1; then
  ESBUILD=(esbuild)
else
  ESBUILD=(npx --no-install esbuild)
fi

COMMON_FLAGS=(
  --bundle
  --platform=browser
  --minify
  --alias:fs=./scripts/empty.js
  --alias:path=./scripts/empty.js
  --alias:crypto=./scripts/empty.js
  --alias:node:fs=./scripts/empty.js
  --alias:node:path=./scripts/empty.js
  --alias:node:crypto=./scripts/empty.js
  --define:__dirname='""'
  --define:__filename='""'
)

build_entry() {
  local entry="$1"
  local suffix="$2"
  local global_name="$3"
  local exports
  exports=$(node -e "const m = require('${entry}'); console.log(Object.keys(m).join(','))")

  echo "  ${suffix}: ${exports}"

  "${ESBUILD[@]}" "${entry}" \
    "${COMMON_FLAGS[@]}" \
    --format=iife \
    --global-name="${global_name}" \
    --outfile="${DIST_DIR}/${PKG_NAME}.${suffix}.js"

  local internal="__${PKG_NAME}_${suffix//./_}"
  "${ESBUILD[@]}" "${entry}" \
    "${COMMON_FLAGS[@]}" \
    --format=iife \
    --global-name="${internal}" \
    --outfile="${DIST_DIR}/${PKG_NAME}.${suffix}.mjs"

  IFS=',' read -ra keys <<< "$exports"
  local destructure export_line
  destructure=$(IFS=','; echo "${keys[*]}")
  export_line=$(IFS=','; echo "${keys[*]}")
  echo "var {${destructure}}=${internal};export{${export_line}};" >> "${DIST_DIR}/${PKG_NAME}.${suffix}.mjs"
}

echo "=== Building browser bundles ==="
echo "  Package: ${PKG_NAME}"

mkdir -p "$DIST_DIR"
rm -f \
  "${DIST_DIR}/${PKG_NAME}.js" \
  "${DIST_DIR}/${PKG_NAME}.mjs" \
  "${DIST_DIR}/${PKG_NAME}.sync.js" \
  "${DIST_DIR}/${PKG_NAME}.sync.mjs" \
  "${DIST_DIR}/${PKG_NAME}.async.js" \
  "${DIST_DIR}/${PKG_NAME}.async.mjs"

build_entry "$SYNC_ENTRY" sync "${PKG_NAME}"
build_entry "$ASYNC_ENTRY" async "${PKG_NAME}"

echo "=== Browser bundles built ==="
ls -lh \
  "${DIST_DIR}/${PKG_NAME}.sync.js" \
  "${DIST_DIR}/${PKG_NAME}.sync.mjs" \
  "${DIST_DIR}/${PKG_NAME}.async.js" \
  "${DIST_DIR}/${PKG_NAME}.async.mjs"
