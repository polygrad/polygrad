'use strict'

function validateEnv(env) {
  if (env === undefined) return
  if (!env || typeof env !== 'object' || Array.isArray(env) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(env))) {
    throw new TypeError('polygrad: env must be a string map')
  }
  for (const [key, value] of Object.entries(env)) {
    if (!key || /[=\0]/.test(key) || typeof value !== 'string' || value.includes('\0')) {
      throw new TypeError('polygrad: env requires nonempty names without = or NUL and string values without NUL')
    }
  }
}

function startupEnv(env) {
  const inherited = typeof process !== 'undefined' && process.versions?.node ? process.env : {}
  return {
    values: Object.assign(Object.create(null), inherited, env),
    explicit: Object.assign(Object.create(null), env)
  }
}

// The non-modularized sync artifact executes during require(). Pass its startup
// configuration only for that call; the async factory accepts it directly.
function loadSyncWithEnv(load, env) {
  const key = Symbol.for('polygrad.wasm.env'), previous = globalThis[key]
  globalThis[key] = startupEnv(env)
  try { return load() }
  finally {
    if (previous === undefined) delete globalThis[key]
    else globalThis[key] = previous
  }
}

function checkSyncEnv(Module, env) {
  for (const [key, value] of Object.entries(env || {})) {
    if (Module.__polygradExplicitEnv?.[key] !== value) {
      throw new Error('polygrad: Wasm environment already initialized; use createAsync() for a separate module')
    }
  }
}

module.exports = { validateEnv, startupEnv, loadSyncWithEnv, checkSyncEnv }
