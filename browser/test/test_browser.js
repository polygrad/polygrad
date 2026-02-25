const { chromium, firefox } = require('playwright')

const URL = process.argv[2] || 'http://127.0.0.1:3456/'
const BROWSER = process.argv[3] || 'chromium'

;(async () => {
  const launcher = BROWSER === 'firefox' ? firefox : chromium
  console.log(`Launching ${BROWSER}...`)
  const browser = await launcher.launch({ headless: true })
  const page = await browser.newPage()

  const consoleLogs = []
  const errors = []
  page.on('console', msg => {
    const text = msg.text()
    consoleLogs.push(`[${msg.type()}] ${text}`)
    if (msg.type() === 'error') errors.push(text)
  })
  page.on('pageerror', err => {
    errors.push(`PAGE ERROR: ${err.message}`)
    consoleLogs.push(`[pageerror] ${err.message}`)
  })

  console.log(`Navigating to ${URL} ...`)
  await page.goto(URL, { waitUntil: 'load', timeout: 30000 })

  // Wait for WASM module to initialize
  await page.waitForFunction(() => typeof polygradModule !== 'undefined', { timeout: 15000 })
  console.log('WASM module loaded, triggering tests...')

  // Trigger tests via evaluate (don't await — it runs async in the page)
  await page.evaluate(() => { runAllTests() })

  // Wait for all badges to resolve (pass or fail)
  console.log('Waiting for tests to complete...')
  try {
    await page.waitForFunction(() => {
      const badges = document.querySelectorAll('.badge')
      if (badges.length === 0) return false
      return Array.from(badges).every(b =>
        b.classList.contains('pass') || b.classList.contains('fail')
      )
    }, { timeout: 180000 })
  } catch (e) {
    console.log('Timeout waiting for tests')
  }

  const results = await page.evaluate(() => {
    const blocks = document.querySelectorAll('.test-block')
    return Array.from(blocks).map((b, i) => {
      const title = b.querySelector('h3')?.textContent || `Test ${i + 1}`
      const badge = b.querySelector('.badge')
      let status = 'PENDING'
      let detail = ''
      if (badge) {
        if (badge.classList.contains('pass')) status = 'PASS'
        else if (badge.classList.contains('fail')) {
          status = 'FAIL'
          detail = badge.textContent
        }
      }
      return { title, status, detail }
    })
  })

  console.log('\n=== Test Results ===')
  let nPass = 0, nFail = 0
  for (const r of results) {
    const extra = r.detail && r.status === 'FAIL' ? ` — ${r.detail}` : ''
    console.log(`  ${r.status}: ${r.title}${extra}`)
    if (r.status === 'PASS') nPass++
    if (r.status === 'FAIL') nFail++
  }

  if (errors.length > 0) {
    console.log('\n=== Console Errors ===')
    for (const e of errors) console.log(`  ${e}`)
  }

  if (nFail > 0 || errors.length > 0) {
    console.log('\n=== All Console Output ===')
    for (const l of consoleLogs) console.log(`  ${l}`)
  }

  console.log(`\n${nPass} passed, ${nFail} failed, ${results.length} total (${BROWSER})`)
  await browser.close()
  process.exit(nFail > 0 ? 1 : 0)
})()
