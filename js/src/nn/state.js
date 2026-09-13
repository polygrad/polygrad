'use strict'

function getParameters(obj) {
  // Pinned tinygrad derives parameters from every state-dict value. Preserve
  // diamond alias paths exactly; optimizer code decides which are trainable.
  return Object.values(getStateDict(obj))
}

function getStateDict(obj, prefix = '') {
  const state = {}
  const active = new Set()
  const walk = (x, prefix) => {
    if (!x) return
    if (x._tensor) {
      state[prefix.replace(/^\.+|\.+$/g, '')] = x
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
        walk(v, prefix ? `${prefix}.${k}` : k)
      }
    } finally {
      active.delete(x)
    }
  }
  walk(obj, prefix.replace(/\.+$/g, ''))
  return state
}

function* stateReplacements(model, state, opts) {
  for (const [name, target] of Object.entries(getStateDict(model))) {
    if (!Object.prototype.hasOwnProperty.call(state, name)) {
      if (opts.strict === false) continue
      throw new Error(`missing state key: ${name}`)
    }
    let source = state[name]
    const a = target.shape, b = source.shape
    if (a.length !== b.length || a.some((dim, i) => dim !== b[i])) {
      if ((a.length === 0 && b.length === 1 && b[0] === 1) ||
          (b.length === 0 && a.length === 1 && a[0] === 1)) {
        source = state[name] = source.reshape(...a)
      } else {
        throw new Error(`Shape mismatch in layer ${name}: expected ${JSON.stringify(a)}, received ${JSON.stringify(b)}`)
      }
    }
    // nn.state.load_state_dict replaces the lazy value, not the target handle
    // or its existing storage bytes. Device/context checks remain in Tensor.
    target.replace(source.to(target.device))
    yield [name, target]
  }
}

function loadStateDict(model, state, opts = {}) {
  const loaded = []
  for (const [name, target] of stateReplacements(model, state, opts)) {
    if (opts.realize !== false) target.realize()
    if (opts.consume) delete state[name]
    loaded.push(target)
  }
  return loaded
}

async function loadStateDictAsync(model, state, opts = {}) {
  const loaded = []
  for (const [name, target] of stateReplacements(model, state, opts)) {
    if (opts.realize !== false) await target.realizeAsync()
    if (opts.consume) delete state[name]
    loaded.push(target)
  }
  return loaded
}

module.exports = { getParameters, getStateDict, loadStateDict, loadStateDictAsync }
