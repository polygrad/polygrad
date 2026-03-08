#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"
WASM_ENTRY="${PROJECT_DIR}/wasm/polygrad.js"
BROWSER_ENTRY="${PROJECT_DIR}/src/browser.js"

if [ ! -f "$WASM_ENTRY" ]; then
  echo "ERROR: ${WASM_ENTRY} not found"
  echo "Run 'make -C .. wasm-pkg' first."
  exit 1
fi

PKG_NAME=$(node -e "
  const p = require('${PROJECT_DIR}/package.json')
  const name = p.name.replace(/^@/, '').replace(/[/-](\\w)/g, (_, c) => c.toUpperCase())
  console.log(name)
")

EXPORTS=$(node -e "
  const m = require('${BROWSER_ENTRY}')
  console.log(Object.keys(m).join(','))
")

echo "=== Building browser bundles ==="
echo "  Package: ${PKG_NAME}"
echo "  Exports: ${EXPORTS}"

mkdir -p "$DIST_DIR"

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
  --alias:node:fs=./scripts/empty.js
  --alias:node:path=./scripts/empty.js
  --define:__dirname='""'
  --define:__filename='""'
)

"${ESBUILD[@]}" "${BROWSER_ENTRY}" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${PKG_NAME}" \
  --outfile="${DIST_DIR}/${PKG_NAME}.js"

INTERNAL="__${PKG_NAME}"
"${ESBUILD[@]}" "${BROWSER_ENTRY}" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${INTERNAL}" \
  --outfile="${DIST_DIR}/${PKG_NAME}.mjs"

IFS=',' read -ra KEYS <<< "$EXPORTS"
DESTRUCTURE=$(IFS=','; echo "${KEYS[*]}")
EXPORT_LINE=$(IFS=','; echo "${KEYS[*]}")
echo "var {${DESTRUCTURE}}=${INTERNAL};export{${EXPORT_LINE}};" >> "${DIST_DIR}/${PKG_NAME}.mjs"

echo "=== Browser bundles built ==="
ls -lh "${DIST_DIR}/${PKG_NAME}.js" "${DIST_DIR}/${PKG_NAME}.mjs"
