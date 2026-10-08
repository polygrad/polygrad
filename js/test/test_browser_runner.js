'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const { runForDevice, launchOptionsFor, probeDevice, checkTemporarySpace } = require('./browser/run')

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
      assert.equal(options.timeout, 300000)
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
  assert.throws(() => checkTemporarySpace('/small', { bavail: 100000, bsize: 4096 }), /Set BROWSER_TMPDIR/)
  checkTemporarySpace('/large', { bavail: 524288, bsize: 4096 })
  const qwen = fs.readFileSync(require.resolve('./browser/qwen_webgpu'), 'utf8')
  assert.match(qwen, /waitForFunction\(\(\) => window\.__qwenResults, null, \{ timeout: 900000 \}\)/,
    'Qwen timeout must be the third argument, not predicate data')
  for (const device of ['auto', 'interp', 'webgpu']) {
    const spec = { launcherName: 'chromium' }
    const opts = launchOptionsFor(spec, device).opts
    if (fs.existsSync('/usr/bin/google-chrome')) assert.equal(opts.executablePath, '/usr/bin/google-chrome')
    assert.equal(launchOptionsFor({...spec, executablePath:'/chosen/chrome'}, device).opts.executablePath, '/chosen/chrome')
    assert.equal(launchOptionsFor({...spec, channel:'chrome-beta'}, device).opts.channel, 'chrome-beta')
    assert.equal(launchOptionsFor({...spec, channel:'chrome-beta'}, device).opts.executablePath, undefined)
  }
  for (const device of ['auto', 'interp', 'webgpu']) await check(device)
  for (const error of [null, 'navigator.gpu missing', 'requestAdapter returned null']) {
    let closed = false
    const page = {
      async goto(url) { assert.equal(url, 'http://127.0.0.1:1234/preflight') },
      async evaluate(fn, device) { assert.equal(device, 'webgpu'); return error },
      async close() { closed = true },
    }
    const browser = { async newPage() { return page } }
    if (error) await assert.rejects(probeDevice(browser, 1234, 'webgpu'), { message: error })
    else assert.deepEqual(await probeDevice(browser, 1234, 'webgpu'), { passed: 1, failed: 0 })
    assert(closed)
  }
  console.log('Browser runner: 3 passed')
})().catch(error => { console.error(error); process.exitCode = 1 })
