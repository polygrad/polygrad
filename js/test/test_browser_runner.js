'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const { runForDevice, launchOptionsFor } = require('./browser/run')

async function check(device) {
  const results = { passed: 1, failed: 0 }
  let closed = false
  const page = {
    on() {},
    async goto(url, options) {
      assert(url.includes(`device=${device}`))
      assert.equal(options?.waitUntil, 'commit', 'page load must not wait for synchronous tests')
    },
    async waitForFunction(fn, arg, options) {
      assert.equal(arg, null, 'timeout is not the predicate argument')
      assert.equal(options.timeout, device === 'webgpu' ? 120000 : 60000)
      return { async jsonValue() { return results } }
    },
    async evaluate() { return { ok: true, features: [] } },
    async close() { closed = true },
  }
  const browser = { async newPage() { return page }, version() { return 'test-double' } }
  assert.equal(await runForDevice(browser, 1234, device, { label: 'test', launcherName: 'chromium' }), results)
  assert(closed)
}

(async () => {
  for (const device of ['auto', 'interp', 'webgpu']) {
    const spec = { launcherName: 'chromium' }
    const opts = launchOptionsFor(spec, device).opts
    if (fs.existsSync('/usr/bin/google-chrome')) assert.equal(opts.executablePath, '/usr/bin/google-chrome')
    assert.equal(launchOptionsFor({...spec, executablePath:'/chosen/chrome'}, device).opts.executablePath, '/chosen/chrome')
    assert.equal(launchOptionsFor({...spec, channel:'chrome-beta'}, device).opts.channel, 'chrome-beta')
    assert.equal(launchOptionsFor({...spec, channel:'chrome-beta'}, device).opts.executablePath, undefined)
  }
  for (const device of ['auto', 'interp', 'webgpu']) await check(device)
  console.log('Browser runner: 3 passed')
})().catch(error => { console.error(error); process.exitCode = 1 })
