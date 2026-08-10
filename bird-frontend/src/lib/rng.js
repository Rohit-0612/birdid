/**
 * Seeded randomness and coherent noise.
 *
 * Everything in the ambient layer is generated rather than drawn, and all of it
 * has to be *deterministic*: the same seed must produce the same forest on every
 * render, in every tab, forever. `Math.random()` cannot do that, and React
 * rejects it in a render body for good reason — see the note in Feathers.jsx.
 *
 * All pure functions, no imports, no React. Lives in a `.js` file on purpose:
 * eslint-plugin-react-refresh rejects exporting non-components beside a
 * component, so helpers must never share a file with JSX.
 */

/**
 * mulberry32 — a 32-bit PRNG. Small, fast, and good enough for visual work:
 * passes gjrand's basic suite and has a 2^32 period, which is several orders of
 * magnitude more numbers than a forest will ever ask for.
 */
export function mulberry32(a) {
  return function next() {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const LATTICE = 256

/**
 * 1-D value noise: a random value at every integer, smoothly interpolated
 * between. Unlike `Math.random()` per sample — which gives static — consecutive
 * samples are correlated, which is what makes it read as terrain rather than
 * noise.
 *
 * Quintic fade (6t⁵-15t⁴+10t³) rather than the cheaper smoothstep because its
 * second derivative is also zero at the endpoints. With smoothstep you can see
 * the lattice: curvature changes abruptly at every integer and the eye picks up
 * the regular spacing, which is exactly the artefact this whole module exists to
 * avoid.
 */
export function makeNoise1D(seed) {
  const rnd = mulberry32(seed)
  const table = Float32Array.from({ length: LATTICE }, () => rnd() * 2 - 1)
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10)

  return function noise(x) {
    const i = Math.floor(x)
    const f = x - i
    // Double-mod so negative x wraps correctly; JS % keeps the sign.
    const a = table[((i % LATTICE) + LATTICE) % LATTICE]
    const b = table[(((i + 1) % LATTICE) + LATTICE) % LATTICE]
    return a + (b - a) * fade(f)
  }
}

/**
 * Fractional Brownian motion: sum octaves of noise at doubling frequency and
 * halving amplitude. This is what turns a smooth wobble into something with both
 * broad structure and fine detail — hills that have bumps that have texture.
 *
 * Lacunarity is 2.03, not 2. With an exact doubling every octave lands on the
 * same lattice points and the octaves re-align periodically, producing a visible
 * repeat — the precise artefact we are removing from the old hand-drawn zigzag.
 * A slightly irrational ratio keeps them permanently out of phase.
 */
export function fbm(noise, x, octaves = 4, lacunarity = 2.03, gain = 0.5) {
  let sum = 0
  let amp = 1
  let norm = 0
  let freq = 1
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise(x * freq)
    norm += amp
    amp *= gain
    freq *= lacunarity
  }
  return sum / norm // normalised back to roughly -1..1
}

/**
 * Ridged fBm. Folding at zero (`1 - |n|`) turns rolling hills into sharp peaks
 * with rounded valleys, which is what a treeline or a ridge actually looks like.
 * Plain fBm gives you dunes.
 */
export function ridged(noise, x, octaves = 4) {
  return 1 - Math.abs(fbm(noise, x, octaves))
}

/**
 * Catmull-Rom through every point, emitted as SVG cubic beziers.
 *
 * The curve passes through the control points (unlike a plain bezier), so the
 * generator can think in terms of "the ridge is this high here" and get a smooth
 * line for free. The 1/6 factor is the standard Catmull-Rom to Bezier conversion:
 * c1 = p1 + (p2-p0)/6, c2 = p2 - (p3-p1)/6.
 */
export function smoothPath(points, close = false) {
  const n = points.length
  if (n < 2) return ''
  const at = (i) => points[close ? ((i % n) + n) % n : Math.min(Math.max(i, 0), n - 1)]

  let d = `M${round(points[0][0])} ${round(points[0][1])}`
  const last = close ? n : n - 1
  for (let i = 0; i < last; i++) {
    const p0 = at(i - 1)
    const p1 = at(i)
    const p2 = at(i + 1)
    const p3 = at(i + 2)
    const c1x = p1[0] + (p2[0] - p0[0]) / 6
    const c1y = p1[1] + (p2[1] - p0[1]) / 6
    const c2x = p2[0] - (p3[0] - p1[0]) / 6
    const c2y = p2[1] - (p3[1] - p1[1]) / 6
    d += `C${round(c1x)} ${round(c1y)} ${round(c2x)} ${round(c2y)} ${round(p2[0])} ${round(p2[1])}`
  }
  return close ? `${d}Z` : d
}

/** Two decimals is well under a device pixel and roughly halves the path string. */
const round = (v) => Math.round(v * 100) / 100

/**
 * Twice the signed area of a polygon (the shoelace sum).
 *
 * Sign is what matters here, not magnitude: it tells you which way the points
 * wind.
 */
export function signedArea(points) {
  let sum = 0
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i]
    const [x2, y2] = points[(i + 1) % points.length]
    sum += x1 * y2 - x2 * y1
  }
  return sum / 2
}

/**
 * Force a polygon to wind consistently.
 *
 * This is not tidiness, it is correctness. When several closed subpaths share
 * one `d`, SVG fills them with the nonzero rule, which cancels wherever two
 * shapes overlap with *opposite* winding — punching holes exactly where a leaf
 * cluster crosses its own branch. The failure looks like a checkerboard and is
 * baffling until you know the cause, so every closed shape goes through here
 * before it is emitted.
 */
export const orient = (points) => (signedArea(points) < 0 ? [...points].reverse() : points)

/** A straight polyline, for the cheap layers where curvature buys nothing. */
export function linePath(points, close = false) {
  if (!points.length) return ''
  let d = `M${round(points[0][0])} ${round(points[0][1])}`
  for (let i = 1; i < points.length; i++) d += `L${round(points[i][0])} ${round(points[i][1])}`
  return close ? `${d}Z` : d
}

export const clamp = (v, min, max) => Math.min(Math.max(v, min), max)

export const lerp = (a, b, t) => a + (b - a) * t

/** Hermite ease between two edges. Returns 0 below `a`, 1 above `b`. */
export function smoothstep(a, b, x) {
  const t = clamp((x - a) / (b - a || 1e-6), 0, 1)
  return t * t * (3 - 2 * t)
}

/** Wrap an angle to (-π, π]. Needed whenever you subtract two headings. */
export const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a))

/**
 * Frame-rate independent exponential approach.
 *
 * The naive `x += (target - x) * k` moves faster at high frame rates, so the
 * bird would bank differently on a 120Hz display than a 60Hz one. Clamping the
 * factor to 1 keeps it stable if a frame takes unusually long.
 */
export const approach = (current, target, rate, dt) =>
  current + (target - current) * clamp(rate * dt, 0, 1)
