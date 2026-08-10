import {
  mulberry32,
  makeNoise1D,
  fbm,
  ridged,
  smoothPath,
  linePath,
  orient,
  smoothstep,
} from '../../lib/rng'

/**
 * A generated forest.
 *
 * The silhouettes this replaces were hand-drawn, and hand-drawn by someone in a
 * hurry: the treeline was a strict zigzag on a ~26px period, and the foliage
 * along the top edge was a repeating scallop. Both are perfectly regular, and
 * regularity at that scale is the thing that makes a background read as
 * generated-by-machine rather than drawn-from-life. Nature has no period.
 *
 * So this generates instead of drawing, and every quantity that could be
 * constant is jittered from a seeded stream: heights, spacing, lean, crown
 * shape, whorl spread, even the ground line under each band. The seed is fixed,
 * so it is the same forest every time — deterministic, but not regular.
 *
 * Output is one path string per depth band. Trees are appended as subpaths of a
 * single `d` rather than as separate elements: the nonzero fill rule merges
 * overlaps for free, and each band costs one node and one draw call.
 */

/**
 * How far the ground fills run past the bottom of the viewBox.
 *
 * The bands parallax downward by up to PARALLAX.near pixels, and a fill that
 * stopped at the viewBox edge would slide up and expose a band of page colour
 * underneath. Comfortably more than the largest travel.
 */
const OVERSCAN = 260

/** Fraction of trees that are conifers rather than broadleaves. */
const CONIFER_SHARE = 0.42

/* ── ground ─────────────────────────────────────────────── */

/**
 * The bottom edge of a band.
 *
 * Even this is noisy. A dead-straight fill under a stand of trees is a
 * surprisingly strong tell — it reads as a strip of colour with trees stuck on
 * top, because ground is never level.
 */
function groundLine(noise, width, baseY, amp, samples = 24) {
  const pts = []
  for (let i = 0; i <= samples; i++) {
    const x = (i / samples) * width
    pts.push([x, baseY + amp * fbm(noise, x / 260, 2)])
  }
  return pts
}

/* ── ridgelines ─────────────────────────────────────────── */

/**
 * A distant ridge.
 *
 * Ridged fBm rather than plain: folding the noise at zero gives sharp summits
 * and rounded valleys, which is what erosion actually produces. Plain fBm gives
 * rolling dunes — recognisably the wrong landform, and close to what the old
 * hand-drawn bezier looked like.
 */
function ridgeline(noise, { width, baseY, amp, scale, samples = 64 }) {
  const pts = []
  for (let i = 0; i <= samples; i++) {
    const x = (i / samples) * width
    pts.push([x, baseY - amp * (0.5 + 0.5 * ridged(noise, x / scale, 4))])
  }
  return pts
}

/* ── trees ──────────────────────────────────────────────── */

/**
 * A conifer.
 *
 * The sawtooth edge comes from whorls — the rings of branches a conifer puts out
 * once a year — not from drawing a triangle. Each whorl steps out to a tip and
 * back in to a notch, so the profile is jagged for a structural reason and the
 * jaggedness varies the way real branch length does.
 *
 * `spread` uses u^0.78 rather than u so the taper is slightly concave, which is
 * how a spruce actually narrows. Linear taper reads as a Christmas-tree icon.
 */
function conifer(rnd, x, baseY, h) {
  const halfW = h * (0.15 + rnd() * 0.07)
  const whorls = 6 + Math.floor(rnd() * 5)
  const lean = (rnd() - 0.5) * h * 0.07 // nothing grows perfectly plumb
  const left = []
  const right = []

  for (let i = 0; i <= whorls; i++) {
    const u = i / whorls // 0 at the tip, 1 at the base
    const y = baseY - h * (1 - u)
    const cx = x + lean * (1 - u)
    const spread = halfW * Math.pow(u, 0.78) * (0.78 + rnd() * 0.44)
    const drop = h * 0.05 * (0.6 + rnd() * 0.8)
    left.push([cx - spread, y], [cx - spread * 0.34, y + drop])
    right.unshift([cx + spread, y], [cx + spread * 0.34, y + drop])
  }

  // A bare trunk below the lowest whorl. This is what lets sky show through
  // under the canopy instead of the band becoming a solid mass at the bottom.
  const trunk = halfW * 0.075
  return linePath(orient([...left, [x - trunk, baseY], [x + trunk, baseY], ...right]), true)
}

/**
 * A broadleaf: a tapered trunk that forks into limbs, under a lumpy crown.
 *
 * The crown is a closed loop of radial fBm — r(θ) rather than a circle — so it
 * is never symmetric and never smooth. `gapBias` pushes it toward whichever
 * neighbour is further away, because a tree in a stand grows into the light it
 * can reach. That one detail is most of what makes a group read as grown rather
 * than as placed.
 */
function broadleaf(rnd, noise, x, baseY, h, gapBias) {
  const parts = []
  const seed = rnd() * 100
  const trunkH = h * (0.42 + rnd() * 0.16)
  const trunkW = h * (0.026 + rnd() * 0.016)
  const lean = (rnd() - 0.5) * h * 0.1

  // Tapered trunk, drawn as a closed sliver rather than a rectangle.
  const topX = x + lean
  parts.push(
    linePath(
      orient([
        [x - trunkW, baseY],
        [topX - trunkW * 0.42, baseY - trunkH],
        [topX + trunkW * 0.42, baseY - trunkH],
        [x + trunkW, baseY],
      ]),
      true,
    ),
  )

  // Two or three limbs off the fork, each with its own angle and length.
  const limbs = 2 + Math.floor(rnd() * 2)
  for (let i = 0; i < limbs; i++) {
    const dir = i === 0 ? -1 : i === 1 ? 1 : rnd() - 0.5
    const angle = dir * (0.35 + rnd() * 0.5)
    const len = h * (0.2 + rnd() * 0.16)
    const ex = topX + Math.sin(angle) * len
    const ey = baseY - trunkH - Math.cos(angle) * len
    const w = trunkW * 0.42
    parts.push(
      linePath(
        orient([
          [topX - w, baseY - trunkH],
          [ex - w * 0.35, ey],
          [ex + w * 0.35, ey],
          [topX + w, baseY - trunkH],
        ]),
        true,
      ),
    )
  }

  // Crown: radial fBm sampled around the circle.
  const cx = topX + gapBias * h * 0.1
  const cy = baseY - h * (0.72 + rnd() * 0.1)
  const R = h * (0.3 + rnd() * 0.12)
  // Enough samples to carry detail finer than a lobe. At 26 the Catmull-Rom
  // smoothed straight through the small stuff and every crown came out as a
  // handful of soft blobs — recognisably broccoli, not a tree.
  const steps = 54
  const crown = []
  for (let i = 0; i < steps; i++) {
    const th = (i / steps) * Math.PI * 2
    // Two scales again: θ*2.4 gives the two or three big masses a crown breaks
    // into, θ*11 gives the ragged clumping along their edges. A real canopy
    // silhouette is jagged at the leaf-cluster scale, and leaving that out is
    // most of why smooth vector foliage looks synthetic.
    const broad = 0.5 + 0.5 * fbm(noise, th * 2.4 + seed, 3)
    const clump = 0.5 + 0.5 * fbm(noise, th * 11 + seed * 3, 2)
    let r = R * (0.66 + 0.42 * broad + 0.16 * clump)
    // Lean into the gap, and sit flatter underneath where the crown shades
    // itself out and the lower branches die back.
    r *= 1 + gapBias * 0.16 * Math.cos(th)
    r *= 1 - 0.16 * Math.max(0, Math.sin(th))
    crown.push([cx + Math.cos(th) * r, cy + Math.sin(th) * r * 0.82])
  }
  parts.push(smoothPath(orient(crown), true))

  return parts.join('')
}

/* ── stands ─────────────────────────────────────────────── */

/**
 * A band of trees.
 *
 * Spacing is proportional to height — a big tree holds more ground than a small
 * one — which is what replaces the old fixed period. Heights are drawn from three
 * strata rather than one uniform range, because a real stand has a canopy, a few
 * emergents pushing through it, and a scrubby understorey, and a single
 * distribution flattens all of that into corduroy.
 */
function stand(rnd, noise, { width, baseY, height, coniferShare = CONIFER_SHARE }) {
  // Two passes: place first so each tree knows its neighbours' gaps, then draw.
  const placed = []
  let x = -0.06 * width
  while (x < width * 1.06) {
    const u = rnd()
    const h =
      u < 0.08
        ? height * (1.32 + rnd() * 0.34) // emergent
        : u < 0.78
          ? height * (0.74 + rnd() * 0.34) // canopy
          : height * (0.3 + rnd() * 0.3) // understorey
    placed.push({ x, h, conifer: rnd() < coniferShare })
    // An occasional clearing. Without these the band is an even wall, and an
    // even wall is the thing the eye reads as wallpaper.
    x += h * (0.22 + rnd() * 0.34) + (rnd() < 0.09 ? height * 0.45 : 0)
  }

  return placed
    .map((tree, i) => {
      const prev = placed[i - 1]
      const next = placed[i + 1]
      const gapL = prev ? tree.x - prev.x : tree.h
      const gapR = next ? next.x - tree.x : tree.h
      // -1 leans left, +1 leans right.
      const bias = (gapR - gapL) / Math.max(gapL + gapR, 1e-3)
      return tree.conifer
        ? conifer(rnd, tree.x, baseY, tree.h)
        : broadleaf(rnd, noise, tree.x, baseY, tree.h, bias)
    })
    .join('')
}

/* ── the top fringe ─────────────────────────────────────── */

/**
 * The canopy you are standing under, seen from below.
 *
 * The old version was a scallop wave across the full width, which read as
 * decorative trim — because a repeating edge treatment is exactly what trim is.
 * The first attempt at replacing it hung separate stems with leaves alternating
 * left and right down each one, and that read as a caterpillar: too regular,
 * too sparse, and structurally nothing like foliage.
 *
 * What this does instead is describe the *lower boundary* of a leaf mass that
 * hangs into frame from above. The boundary is fBm, so it is deeply irregular,
 * and where the noise falls below a threshold it retreats all the way to the top
 * edge — leaving real gaps of open sky rather than a continuous scalloped strip.
 * That asymmetry between dense mass and empty sky is what the eye reads as
 * canopy. A few longer sprigs hang below it to break the silhouette.
 */
function fringe(rnd, noise, { width, height }) {
  const samples = Math.max(48, Math.round(width / 12))
  // Starts above the top edge, not at it: the fringe parallaxes *upward*, and a
  // mass anchored exactly at y=0 would lift off and reveal sky behind itself.
  const overhang = -Math.round(height * 0.6)
  const boundary = [[0, overhang]]

  /** Depth of the leaf mass at x, 0 meaning open sky. */
  const depthAt = (x) => {
    // fBm is a sum of octaves and so clusters hard around its midpoint; used
    // raw, it almost never crossed a threshold and the "gaps" never appeared —
    // the canopy came out as a solid bar with a wavy edge. smoothstep expands
    // the middle of the range back out so both extremes actually occur.
    const broad = smoothstep(0.42, 0.78, 0.5 + 0.5 * fbm(noise, x / 210, 3))
    const detail = 0.5 + 0.5 * fbm(noise, x / 52 + 31.7, 2)
    return broad === 0 ? overhang : height * (0.2 + 0.8 * broad) * (0.72 + 0.28 * detail)
  }

  for (let i = 0; i <= samples; i++) {
    const x = (i / samples) * width
    boundary.push([x, depthAt(x)])
  }
  boundary.push([width, overhang])

  const parts = [smoothPath(orient(boundary), true)]

  // Pendant clumps hanging below the mass.
  //
  // Drawn as a chain of overlapping blobs with no stem between them, and started
  // *inside* the mass so the chain merges with it. An earlier version drew a
  // thin stem with blobs spaced along it and the result was unmistakably a
  // caterpillar: foliage does not hang as beads on a wire, it hangs as clumps
  // touching clumps.
  const sprigs = Math.max(2, Math.round(width / 380))
  for (let i = 0; i < sprigs; i++) {
    const x = ((i + 0.2 + rnd() * 0.6) / sprigs) * width
    const rooted = depthAt(x)
    if (rooted < height * 0.45) continue // never hang a clump out of open sky

    const clumps = 2 + Math.floor(rnd() * 3)
    let cxp = x
    let cyp = rooted - height * 0.14 // start inside the mass, not below it
    for (let k = 0; k < clumps; k++) {
      const lr = height * (0.14 + rnd() * 0.08) * (1 - k * 0.16)
      const steps = 16
      const blob = []
      for (let s = 0; s < steps; s++) {
        const th = (s / steps) * Math.PI * 2
        const r = lr * (0.62 + 0.48 * (0.5 + 0.5 * fbm(noise, th * 2.6 + i * 13 + k * 5, 3)))
        blob.push([cxp + Math.cos(th) * r, cyp + Math.sin(th) * r * 0.86])
      }
      parts.push(smoothPath(orient(blob), true))
      // Step down by less than a radius, so consecutive clumps always overlap.
      cyp += lr * 1.15
      cxp += (rnd() - 0.5) * lr * 0.9
    }
  }
  return parts.join('')
}

/* ── entry point ────────────────────────────────────────── */

/**
 * Build every band for a given viewport.
 *
 * Coordinates are 1:1 CSS pixels, so the caller sets `viewBox="0 0 width height"`
 * and never needs preserveAspectRatio="none". That matters: stretching a fixed
 * viewBox to fit made the trees visibly wider on a desktop than on a laptop.
 * Generating for the actual width means a wide viewport gets *more* trees rather
 * than fatter ones, which is the only correct answer and one the old approach
 * could not express.
 */
export function buildForest({ seed, width, height }) {
  const w = Math.max(320, Math.round(width))
  const h = Math.max(240, Math.round(height))

  // A separate stream per band, so editing one does not reshuffle the others.
  const noise = makeNoise1D(seed)
  const noiseB = makeNoise1D(seed ^ 0x1f3d)

  const ridgePts = ridgeline(noise, {
    width: w,
    baseY: h,
    amp: h * 0.3,
    scale: w / 2.6,
  })
  const far = `${smoothPath(ridgePts)}L${w} ${h + OVERSCAN}L0 ${h + OVERSCAN}Z`

  const midGround = groundLine(noiseB, w, h * 0.995, h * 0.012)
  const mid =
    `${linePath(midGround)}L${w} ${h + OVERSCAN}L0 ${h + OVERSCAN}Z` +
    stand(mulberry32(seed ^ 0x2a11), noise, {
      width: w,
      baseY: h * 0.995,
      height: h * 0.3,
    })

  const nearGround = groundLine(noiseB, w, h + 4, h * 0.01)
  const near =
    `${linePath(nearGround)}L${w} ${h + OVERSCAN}L0 ${h + OVERSCAN}Z` +
    stand(mulberry32(seed ^ 0x77c3), noiseB, {
      width: w,
      baseY: h + 4,
      height: h * 0.46,
      coniferShare: 0.3,
    })

  return {
    far,
    mid,
    near,
    fringe: fringe(mulberry32(seed ^ 0x5b0a), noiseB, { width: w, height: h * 0.22 }),
  }
}

/**
 * Bucket a measured width before generating.
 *
 * Regeneration is cheap but not free, and a resize fires continuously while a
 * window is dragged. Snapping to 160px means at most one rebuild per 160px of
 * drag, and the difference between a forest generated for 1280 and one for 1320
 * is not visible.
 */
export const bucketWidth = (width) => Math.max(320, Math.ceil(width / 160) * 160)
