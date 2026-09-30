#!/usr/bin/env node
/**
 * Contrast gate for src/index.css.
 *
 * index.css opens by claiming every colour in it has been contrast-checked, and
 * tells you to "run the ratio script" before changing any of them. That script
 * did not exist. This is it.
 *
 * It does three jobs:
 *
 *   1. Checks every documented foreground/background pair in both themes.
 *   2. Asserts the values the comments say were *rejected* still fail, so a
 *      future edit cannot quietly turn those comments into lies.
 *   3. Holds the photo grain to a luminance budget in both themes.
 *
 * Zero dependencies; run with `npm run contrast`.
 */

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const CSS_PATH = resolve(here, '../src/index.css')

/* ── colour maths (WCAG 2.1) ─────────────────────────────── */

function parseHex(hex) {
  const h = hex.trim().replace('#', '')
  const full =
    h.length === 3
      ? h
          .split('')
          .map((c) => c + c)
          .join('')
      : h
  if (!/^[0-9a-fA-F]{6}$/.test(full)) throw new Error(`not a hex colour: "${hex}"`)
  return [
    parseInt(full.slice(0, 2), 16),
    parseInt(full.slice(2, 4), 16),
    parseInt(full.slice(4, 6), 16),
  ]
}

/** Relative luminance. The 0.03928 branch is the sRGB transfer curve's linear toe. */
function luminance(rgb) {
  const [r, g, b] = rgb.map((v) => {
    const c = v / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function ratio(fg, bg) {
  const a = luminance(fg)
  const b = luminance(bg)
  const [hi, lo] = a > b ? [a, b] : [b, a]
  return (hi + 0.05) / (lo + 0.05)
}

/* ── parsing index.css ───────────────────────────────────── */

const css = readFileSync(CSS_PATH, 'utf8')

/** Pull `--name: value;` pairs out of a brace-delimited block starting at `from`. */
function tokensAfter(source, from) {
  const open = source.indexOf('{', from)
  if (open === -1) return {}
  let depth = 0
  let end = open
  for (let i = open; i < source.length; i++) {
    if (source[i] === '{') depth++
    else if (source[i] === '}') {
      depth--
      if (depth === 0) {
        end = i
        break
      }
    }
  }
  const body = source.slice(open + 1, end)
  const out = {}
  for (const m of body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) out[m[1]] = m[2].trim()
  return out
}

function blockAt(marker, label) {
  const at = css.indexOf(marker)
  if (at === -1) {
    console.error(`✗ could not find the ${label} block (looked for: ${marker})`)
    process.exit(1)
  }
  return tokensAfter(css, at)
}


// Dark is the default and lives in @theme; light redeclares what changes.
const dark = blockAt('@theme {', '@theme / dark')
const lightAttr = blockAt(":root[data-theme='light'] {", 'explicit-light')
const light = { ...dark, ...lightAttr }

/* ── checks ─────────────────────────────────────────────── */

let failures = 0
let checks = 0

function report(ok, label, detail) {
  checks++
  if (ok) {
    console.log(`  ✓ ${label}${detail ? `  ${detail}` : ''}`)
  } else {
    failures++
    console.log(`  ✗ ${label}${detail ? `  ${detail}` : ''}`)
  }
}

function checkPair(tokens, fgName, bgName, min, theme) {
  const fg = tokens[fgName]
  const bg = tokens[bgName]
  if (!fg || !bg) return report(false, `${theme}: ${fgName} on ${bgName}`, 'token missing')
  const r = ratio(parseHex(fg), parseHex(bg))
  report(r >= min, `${theme}: ${fgName} on ${bgName}`, `${r.toFixed(2)}:1 (need ${min})`)
}

/** Foreground, background, minimum ratio. */
const PAIRS = [
  ['--color-ink', '--color-base', 7.0],
  ['--color-ink', '--color-surface', 7.0],
  ['--color-ink-soft', '--color-base', 4.5],
  ['--color-ink-soft', '--color-surface', 4.5],
  // The binding constraint in the whole palette. Everything else has slack.
  ['--color-ink-faint', '--color-base', 4.5],
  ['--color-ink-faint', '--color-surface', 4.5],
  ['--color-ink-faint', '--color-raised', 4.5],
  ['--color-on-accent', '--color-accent', 4.5],
  ['--color-on-accent', '--color-accent-hover', 4.5],
  ['--color-on-reject', '--color-reject', 4.5],
  ['--color-on-panel', '--color-panel', 7.0],
  ['--color-accent', '--color-base', 4.5],
  ['--color-accent', '--color-surface', 4.5],
  ['--color-sun-ink', '--color-base', 4.5],
  ['--color-high', '--color-base', 4.5],
  ['--color-moderate', '--color-base', 4.5],
  ['--color-low', '--color-base', 4.5],
  // Graphic-only colours: 3:1 is the non-text minimum.
  ['--color-sun', '--color-base', 3.0],
  ['--color-accent-bright', '--color-base', 3.0],
]

console.log('\nDesign token contrast — src/index.css\n')

console.log('Dark theme (default)')
for (const [fg, bg, min] of PAIRS) checkPair(dark, fg, bg, min, 'dark')

console.log('\nLight theme')
for (const [fg, bg, min] of PAIRS) checkPair(light, fg, bg, min, 'light')

/* The header of index.css names values that were measured and rejected. If an
   edit ever makes one of these pass, the note has become wrong and should be
   updated deliberately rather than left to mislead the next reader. */
console.log('\nDocumented rejections still fail')
{
  const r1 = ratio(parseHex('#ffffff'), parseHex(dark['--color-accent']))
  report(r1 < 4.5, 'white on the gold accent still fails — on-accent must stay dark', `${r1.toFixed(2)}:1`)
  const r2 = ratio(parseHex(dark['--color-accent']), parseHex(light['--color-base']))
  report(r2 < 4.5, 'dark-theme gold on light paper still fails — light needs its own accent', `${r2.toFixed(2)}:1`)
}

/* scope-night hand-copies the dark tokens so the rail, hero and footer stay
   dark in the light theme. A value that drifts from @theme would make those
   patches a slightly different night from the rest of the dark page. */
console.log('\nscope-night matches the dark theme')
{
  const night = blockAt('@utility scope-night {', 'scope-night')
  let drift = 0
  for (const [name, value] of Object.entries(night)) {
    if (dark[name] !== value) {
      drift++
      console.log(`  ✗ ${name}: scope-night ${value} vs @theme ${dark[name] ?? '(absent)'}`)
    }
  }
  report(drift === 0, 'every scope-night token equals its @theme value', `${Object.keys(night).length} tokens`)
  checkPair(dark, '--color-paper-ink', '--color-paper', 7.0, 'fixed')
}

/* ── grain budget ───────────────────────────────────────── */

/**
 * Grain sits over photographs, and over the photographic hero that the headline
 * is set on, so it gets an absolute budget: it may not shift the base luminance
 * by more than 6%.
 *
 * Modelled with the real W3C soft-light formula, not source-over. Soft-light
 * against a mid-grey blend layer is very close to identity, which is precisely
 * why it was chosen; the interesting case is a locally dark patch of noise. A
 * glyph spans hundreds of pixels and 4-octave fractalNoise averages out across
 * it, so the honest worst case is about one standard deviation below the mean
 * (cs ≈ 0.35), not the single darkest speck in the tile.
 */
const GRAIN_CS_DARK = 0.35
const GRAIN_MAX_SHIFT = 0.06

function softLight(cb, cs) {
  const d = cb <= 0.25 ? ((16 * cb - 12) * cb + 4) * cb : Math.sqrt(cb)
  return cs <= 0.5 ? cb - (1 - 2 * cs) * cb * (1 - cb) : cb + (2 * cs - 1) * (d - cb)
}

const grainOver = (rgb, cs, alpha) =>
  rgb.map((c) => {
    const cb = c / 255
    return (softLight(cb, cs) * alpha + cb * (1 - alpha)) * 255
  })

const departure = (ground, tokens) => {
  const base = luminance(parseHex(tokens['--color-base']))
  return Math.abs(luminance(ground) - base) / Math.max(base, 1e-4)
}

console.log('\nGrain overlay budget')
for (const [themeName, tokens] of [
  ['dark', dark],
  ['light', light],
]) {
  const alpha = Number.parseFloat(tokens['--grain-opacity'])
  if (Number.isNaN(alpha)) {
    report(false, `${themeName}: --grain-opacity`, 'missing')
    continue
  }
  const base = parseHex(tokens['--color-base'])
  const shift = departure(grainOver(base, GRAIN_CS_DARK, alpha), tokens)
  report(
    shift <= GRAIN_MAX_SHIFT,
    `${themeName}: grain shifts base luminance by ${(shift * 100).toFixed(1)}%`,
    `budget ${(GRAIN_MAX_SHIFT * 100).toFixed(0)}%`,
  )
}

/* ── result ─────────────────────────────────────────────── */

console.log(
  `\n${failures === 0 ? '✓' : '✗'} ${checks - failures}/${checks} checks passed\n`,
)
process.exit(failures === 0 ? 0 : 1)
