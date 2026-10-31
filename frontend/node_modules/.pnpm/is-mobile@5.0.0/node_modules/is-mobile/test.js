'use strict'

const assert = require('assert')
const test = require('test')
const UserAgent = require('user-agents')
const isMobile = require('./')

const iphone =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A334 Safari/7534.48.3'
const chrome =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.95 Safari/537.36'
const ffos = 'Mozilla/5.0 (Mobile; rv:18.0) Gecko/18.0 Firefox/18.0'
const ipad =
  'Mozilla/5.0 (iPad; CPU OS 9_3_2 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13F69 Safari/601.1'
const ios13ipad =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1'
const ios13ipadpro =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Safari/605.1.15'
const samsung =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/16.0 Chrome/92.0.4515.166 Safari/537.36'
const samsungMobile =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/16.0 Chrome/92.0.4515.166 Mobile Safari/537.36'
const chromeOS =
  'Mozilla/5.0 (X11; CrOS armv7l 12105.100.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.144 Safari/537.36'
const redmiMobile =
  'HashKey/1.20.0 (Redmi K30; Android ) Language/zh Theme/light ScreenWidth/392 ScreenHeight/856'

test('is mobile', function () {
  assert(isMobile({ ua: iphone }))
  assert(isMobile({ ua: ffos }))
  assert(!isMobile({ ua: ipad }))
  assert(isMobile({ ua: ipad, tablet: true }))
  assert(isMobile({ ua: { headers: { 'user-agent': iphone } } }))
  assert(!isMobile({ ua: chrome }))
  assert(!isMobile({ ua: { headers: { 'user-agent': chrome } } }))
  assert(!isMobile())
  assert(!isMobile({ ua: { headers: null } }))
  assert(!isMobile({ ua: { headers: { 'user-agent': null } } }))
  assert(!isMobile({ ua: samsung }))
  assert(isMobile({ ua: samsungMobile }))
  assert(isMobile({ ua: redmiMobile }))
  assert(!isMobile(chromeOS))
  assert(!isMobile(chromeOS, { tablet: true }))

  global.navigator = {}

  global.navigator.userAgent = iphone
  assert(isMobile())
  assert(isMobile({ tablet: true }))

  global.navigator.userAgent = chrome
  assert(!isMobile())
  assert(!isMobile({ tablet: true }))

  global.navigator.userAgent = ipad
  assert(!isMobile())
  assert(isMobile({ tablet: true }))

  global.navigator = { maxTouchPoints: 5 }
  assert(isMobile({ ua: ios13ipad, tablet: true, featureDetect: true }))
  assert(isMobile({ ua: ios13ipadpro, tablet: true, featureDetect: true }))
})

test('ua-bruteforce', function () {
  const limit = 300
  const checks = [
    { deviceCategory: 'mobile', result: true },
    { deviceCategory: 'tablet', result: true, tablet: true },
    { deviceCategory: 'desktop', result: false }
  ]
  const testCases = checks.reduce(
    (cases, { deviceCategory, result, tablet }) => {
      // The same user-agent string belongs to both `desktop` and `mobile` type entries. No chance to detect `deviceType` properly.
      // https://github.com/intoli/user-agents/blob/867e318bc00880ae00437e5e8efaa8e5e7ac0696/src/user-agents.json.gz
      // user-agents v1.0.843
      const exclude =
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
      const ua = new UserAgent([
        ({ userAgent }) => userAgent !== exclude,
        { deviceCategory }
      ])

      return [
        ...cases,
        ...new Array(limit).fill().map(() => ({
          ua: ua.random().toString(),
          result,
          tablet
        }))
      ]
    },
    []
  )

  testCases.forEach(({ ua, result, tablet }) => {
    test(ua, () => {
      assert.strictEqual(isMobile({ ua, tablet }), result)
    })
  })
})
