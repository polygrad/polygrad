'use strict'

function getParameters(obj) {
  const out = []
  const seen = new Set()
  const walk = (x) => {
    if (!x || seen.has(x)) return
    seen.add(x)
    if (x._tensor) {
      if (x.requiresGrad) out.push(x)
      return
    }
    if (Array.isArray(x)) {
      for (const v of x) walk(v)
      return
    }
    if (typeof x === 'object') {
      for (const [k, v] of Object.entries(x)) {
        if (!String(k).startsWith('_')) walk(v)
      }
    }
  }
  walk(obj)
  return out
}

function getStateDict(obj) {
  const state = {}
  const seen = new Set()
  const walk = (x, prefix) => {
    if (!x || seen.has(x)) return
    seen.add(x)
    if (x._tensor) {
      if (prefix) state[prefix] = x
      return
    }
    if (Array.isArray(x)) {
      x.forEach((v, i) => walk(v, prefix ? `${prefix}.${i}` : String(i)))
      return
    }
    if (typeof x === 'object') {
      for (const [k, v] of Object.entries(x)) {
        if (!String(k).startsWith('_')) walk(v, prefix ? `${prefix}.${k}` : k)
      }
    }
  }
  walk(obj, '')
  return state
}

module.exports = { getParameters, getStateDict }
