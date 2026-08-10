#!/usr/bin/env node
/**
 * Contrast gate for src/index.css.
 *
 * index.css opens by claiming every colour in it has been contrast-checked, and
 * tells you to "run the ratio script" before changing any of them. That script
 * did not exist. This is it.
 *
 * It does four jobs:
 *
 *   1. Checks every documented foreground/background pair in both themes.
 *   2. Asserts the two hand-duplicated dark blocks have not drifted apart.
 *   3. Asserts the values the comments say were *rejected* still fail, so a
 *      future edit cannot quietly turn those comments into lies.
 *   4. Checks text against the *composited* page ground — base plus the forest
 *      bands plus the grain — because the ambient layer sits behind every word
 *      in the app and darkening it eats the margin the tokens were chosen for.
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

/** Source-over compositing of a translucent layer onto an opaque one. */
function composite(over, under, alpha) {
  return over.map((c, i) => c * alpha + under[i] * (1 - alpha))
}

/** Approximates CSS `color-mix(in oklab, A p%, B)` closely enough for a budget check. */
function mix(a, b, percent) {
  const t = percent / 100
  // Mixing in linear light rather than sRGB — oklab is perceptual, but linear is
  // far closer to it than gamma-encoded sRGB, and this only feeds a threshold.
  const lin = (c) => (c / 255 <= 0.03928 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4)
  const enc = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055) * 255
  return a.map((_, i) => enc(lin(a[i]) * t + lin(b[i]) * (1 - t)))
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

const light = blockAt('@theme', '@theme / light')
// The media-query-guarded dark block, and the explicit-attribute one.
const darkMedia = blockAt("@media (prefers-color-scheme: dark)", 'system-dark')
const darkAttr = blockAt(":root[data-theme='dark']", 'explicit-dark')

// Dark only redeclares what changes; everything else still comes from @theme.
const dark = { ...light, ...darkAttr }

/** Collapse whitespace so a multi-line shadow compares equal to a one-line one. */
const normalised = (o) =>
  Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v.replace(/\s+/g, ' ').trim()]))

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
  // The "hover is darker, not lighter" rule this file's comments describe.
  ['--color-on-accent', '--color-accent-hover', 4.5],
  ['--color-on-reject', '--color-reject', 4.5],
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

console.log('Light theme')
for (const [fg, bg, min] of PAIRS) checkPair(light, fg, bg, min, 'light')

console.log('\nDark theme')
for (const [fg, bg, min] of PAIRS) checkPair(dark, fg, bg, min, 'dark')

/* The two dark blocks are maintained by hand. Drift between them means the
   in-app toggle and the OS setting would render different values — a bug that
   is close to invisible in review and obvious to a user with both.
   Every token, not just colours: shadows and grain opacities drift just as
   easily, and an edit that adds a token to one block and not the other is the
   most likely way it happens. */
console.log('\nDuplicate dark blocks are in sync')
{
  const a = normalised(darkMedia)
  const b = normalised(darkAttr)
  const names = new Set([...Object.keys(a), ...Object.keys(b)])
  let drift = 0
  for (const name of names) {
    if (a[name] !== b[name]) {
      drift++
      console.log(`  ✗ ${name}: system-dark ${a[name] ?? '(absent)'} vs explicit-dark ${b[name] ?? '(absent)'}`)
    }
  }
  report(drift === 0, 'all dark tokens identical in both blocks', `${names.size} tokens`)
}

/* The comments name specific values that were measured and rejected. If an edit
   ever makes one of these pass, the comment has become wrong and should be
   updated deliberately rather than left to mislead the next reader. */
console.log('\nDocumented rejections still fail')
{
  const cases = [
    ['#6b7d73', light['--color-base'], 'light tertiary text (comment says 4.13:1)'],
    ['#6b6577', darkAttr['--color-base'], 'old dark ink-faint (comment says 3.39:1)'],
  ]
  for (const [hex, bg, why] of cases) {
    const r = ratio(parseHex(hex), parseHex(bg))
    report(r < 4.5, `${hex} still below 4.5 — ${why}`, `${r.toFixed(2)}:1`)
  }
  // White on dark-mode accent: the comment says 1.74:1, which is why
  // --color-on-accent has to flip with the theme.
  const r = ratio(parseHex('#ffffff'), parseHex(darkAttr['--color-accent']))
  report(r < 4.5, 'white on dark accent still fails — on-accent must flip', `${r.toFixed(2)}:1`)
}

/* ── the ambient budget ─────────────────────────────────── */

/**
 * The forest and the grain sit behind every word in the app, so they spend
 * contrast that the tokens were chosen to have.
 *
 * How this is measured matters, and the obvious approach is wrong. Treating each
 * band as an opaque wash covering the whole viewport scores the canopy that
 * *currently ships* at 3.21:1 in light mode — a flat failure for a design that
 * is in production and perfectly readable. The model overstates the damage
 * because the bands are silhouettes: trees and ridgelines cover a fraction of
 * their band's area, they sit at the top and bottom edges rather than behind
 * body text, and most text in this app is on an opaque card anyway.
 *
 * Rather than invent a coverage fudge factor, the gate is a *regression* test.
 * Same conservative model on both sides, so the unknown coverage cancels:
 * the new ambient stack must depart from --color-base less than today's does,
 * in both themes, with margin. That is objective and needs no magic number.
 *
 * The absolute ratios are still printed, clearly marked as a lower bound that
 * assumes 100% coverage, because the trend is worth watching even when the
 * number is pessimistic.
 */

// DEPTH_MIX comes from the module the renderer uses; the alphas come from the
// stylesheet, where they are declared per theme. Neither is duplicated here, so
// neither can drift out of step with what actually renders.
const { DEPTH_MIX, GRAIN_ALPHA } = await import('../src/components/nature/atmosphere.js')

const BAND_KEYS = ['ridge', 'trees', 'near', 'fringe']

function bandsFor(tokens) {
  return BAND_KEYS.map((k) => {
    const raw = tokens[`--band-${k}`]
    if (raw === undefined) {
      console.error(`✗ --band-${k} is missing from index.css`)
      process.exit(1)
    }
    return { mix: DEPTH_MIX[k], alpha: Number.parseFloat(raw) }
  })
}

/** What ships today: three bands, undiluted foliage colour, opacity for depth. */
const BASELINE = [
  { mix: 100, alpha: 0.07 },
  { mix: 100, alpha: 0.1 },
  { mix: 100, alpha: 0.09 },
]

/** Must improve on the current design by at least this much. */
const REQUIRED_MARGIN = 0.85

function bandGround(bands, tokens) {
  const base = parseHex(tokens['--color-base'])
  const canopy = parseHex(tokens['--color-canopy'])
  let ground = base
  for (const b of bands) ground = composite(mix(canopy, base, b.mix), ground, b.alpha)
  return ground
}

const departure = (ground, tokens) => {
  const base = luminance(parseHex(tokens['--color-base']))
  return Math.abs(luminance(ground) - base) / Math.max(base, 1e-4)
}

console.log('\nAmbient forest — regression against the canopy that ships today')
for (const [themeName, tokens] of [
  ['light', light],
  ['dark', dark],
]) {
  const before = departure(bandGround(BASELINE, tokens), tokens)
  const after = departure(bandGround(bandsFor(tokens), tokens), tokens)
  report(
    after <= before * REQUIRED_MARGIN,
    `${themeName}: departure from base ${(after * 100).toFixed(1)}% vs current ${(before * 100).toFixed(1)}%`,
    `need ≤${(before * REQUIRED_MARGIN * 100).toFixed(1)}%`,
  )
}

/**
 * Grain gets an absolute budget instead, since there is nothing to regress
 * against — today's design has none.
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

console.log('\nGrain overlay budget')
for (const [themeName, tokens] of [
  ['light', light],
  ['dark', dark],
]) {
  const base = parseHex(tokens['--color-base'])
  const grained = grainOver(base, GRAIN_CS_DARK, GRAIN_ALPHA[themeName])
  const shift = departure(grained, tokens)
  report(
    shift <= GRAIN_MAX_SHIFT,
    `${themeName}: grain shifts base luminance by ${(shift * 100).toFixed(1)}%`,
    `budget ${(GRAIN_MAX_SHIFT * 100).toFixed(0)}%`,
  )
}

/* Informational only — assumes every band is a solid full-viewport wash, which
   no silhouette is. Printed to watch the trend, not to gate on. */
console.log('\nFull-coverage upper bound (informational — silhouettes are not solid)')
for (const [themeName, tokens] of [
  ['light', light],
  ['dark', dark],
]) {
  const ground = grainOver(bandGround(bandsFor(tokens), tokens), GRAIN_CS_DARK, GRAIN_ALPHA[themeName])
  const parts = ['--color-ink-faint', '--color-ink-soft', '--color-accent'].map((n) => {
    const r = ratio(parseHex(tokens[n]), ground)
    const now = ratio(
      parseHex(tokens[n]),
      grainOver(bandGround(BASELINE, tokens), GRAIN_CS_DARK, GRAIN_ALPHA[themeName]),
    )
    return `${n.replace('--color-', '')} ${r.toFixed(2)} (today ${now.toFixed(2)})`
  })
  console.log(`  · ${themeName}: ${parts.join(', ')}`)
}

/* ── result ─────────────────────────────────────────────── */

console.log(
  `\n${failures === 0 ? '✓' : '✗'} ${checks - failures}/${checks} checks passed\n`,
)
process.exit(failures === 0 ? 0 : 1)
