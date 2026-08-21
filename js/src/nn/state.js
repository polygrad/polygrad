'use strict'

function getParameters(obj) {
  // Pinned tinygrad derives parameters from every state-dict value. Preserve
  // diamond alias paths exactly; optimizer code decides which are trainable.
  return Object.values(getStateDict(obj))
}

function getStateDict(obj) {
  const state = {}
  const active = new Set()
  const walk = (x, prefix) => {
    if (!x) return
    if (x._tensor) {
      if (prefix) state[prefix] = x
      return
    }
    if (typeof x !== 'object' || active.has(x)) return
    // Match pinned tinygrad's every-path traversal while terminating only true
    // ancestor cycles. A global visited set incorrectly erased diamond aliases.
    active.add(x)
    try {
      if (Array.isArray(x)) {
        x.forEach((v, i) => walk(v, prefix ? `${prefix}.${i}` : String(i)))
        return
      }
      for (const [k, v] of Object.entries(x)) {
        if (!String(k).startsWith('_')) walk(v, prefix ? `${prefix}.${k}` : k)
      }
    } finally {
      active.delete(x)
    }
  }
  walk(obj, '')
  return state
}

module.exports = { getParameters, getStateDict }
