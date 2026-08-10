/**
 * The one place the ambient layer agrees with itself.
 *
 * Three separate renderers draw this scene — an SVG canopy, a 2D canvas flock,
 * and a WebGL hero bird — and none of them can see the others. Left alone they
 * drift: the sky glows from the left, the bird is lit from the right, the
 * shadows fall straight down, and the wings beat at a rate unrelated to the
 * flock overhead. Individually each layer is fine. Together they read as
 * synthetic, because inconsistent lighting across a single scene is the thing
 * the eye is best at catching and worst at articulating.
 *
 * So every constant that more than one renderer needs lives here, and each
 * renderer derives from it rather than choosing its own.
 */

/**
 * Where the light comes from, in normalised viewport coordinates
 * (0,0 top-left → 1,1 bottom-right).
 *
 * Deliberately off-centre and low. A glow centred at 50% is the single most
 * overused hero background there is, and it also gives the scene no direction:
 * nothing can cast a shadow away from a light that is directly behind it. Low
 * and to the left reads as early morning, which is when people actually go
 * birding, and it gives the canopy, the shafts and the bird a shared azimuth.
 */
export const SUN = { x: 0.26, y: 0.88 }

/** Horizontal sun direction in clip space (-1 left .. +1 right). */
export const SUN_DIR_X = SUN.x * 2 - 1

/** One seed for the whole forest, so the scene is byte-identical run to run. */
export const FOREST_SEED = 0x5eed1a

/**
 * Wingbeat rates, in Hz.
 *
 * BirdFlock stores these as radians/second (its `flap` ranges were 0.9–2.6);
 * these are the same numbers divided by 2π. Expressing them in Hz is what lets
 * the WebGL bird — whose animation clip is measured in seconds — share the
 * numbers with the canvas flock without either knowing the other exists.
 *
 * Small birds beat faster. The far layer is slower not because distance slows
 * anything down but because those are meant to be larger birds further away.
 */
export const FLAP_HZ = {
  far: [0.143, 0.223],
  mid: [0.207, 0.302],
  near: [0.286, 0.414],
}

/**
 * The hero bird's base wingbeat, in Hz.
 *
 * Taken from the model rather than from the table above. The stork clip is 1.3
 * seconds of hand-authored flight — an animator's judgement about what a large
 * bird's wingbeat looks like — which makes its natural rate (1/1.3 ≈ 0.77Hz) the
 * most defensible number available. This sits a little under it: unhurried, but
 * still a bird rather than a slow-motion replay.
 *
 * It is deliberately *not* matched to FLAP_HZ. Those rates are for small birds
 * far away; this is a big bird up close, and large wings beat slower. Making
 * them agree would be the unrealistic choice, not the consistent one.
 */
export const HERO_FLAP_HZ = 0.72

/** The authored clip length, in seconds. Playback is scrubbed against this. */
export const HERO_CLIP_SECONDS = 1.3

/** ±12% wander on that rate, so it is never metronomic. */
export const HERO_FLAP_WANDER = 0.12

/**
 * Parallax depths, in viewport-pixels of travel per 1200px of scroll.
 * Negative moves against the scroll, which is what sells a layer as being in
 * front of the viewport plane rather than behind it.
 */
export const PARALLAX = {
  ridge: 42,
  trees: 96,
  near: 156,
  fringe: -72,
}

/**
 * How much of the foliage colour each depth band keeps, the rest being mixed
 * toward the page background.
 *
 * This is atmospheric perspective, and it is the mechanism that replaces the old
 * "same colour at three opacities" approach. Distant things converge on the
 * colour of the air between you and them; they do not merely get fainter. The
 * bands now sit at nearly the same alpha and get their depth entirely from this
 * mix, which is the inversion that makes the scene read as air rather than as
 * stacked transparencies.
 */
export const DEPTH_MIX = {
  ridge: 20,
  trees: 44,
  near: 78,
  fringe: 84,
}

/**
 * Band alphas live in index.css as --band-ridge/-trees/-near/-fringe, not here.
 *
 * They have to differ per theme — light-on-dark foliage reads far stronger than
 * dark-on-light at the same alpha — and per-theme values belong with the other
 * per-theme values rather than behind a theme check in JS.
 */

/**
 * Grain opacity per theme. Dark needs more: soft-light barely registers against
 * a near-black ground, where the same overlay on mint paper is plainly visible.
 */
export const GRAIN_ALPHA = { light: 0.055, dark: 0.085 }

/**
 * Light shaft periods, in seconds. Pairwise coprime so their sum never repeats
 * inside any plausible session: the pattern would only recur after
 * 61 × 89 × 113 seconds, which is a little over seven days.
 */
export const SHAFT_PERIODS = [61, 89, 113]
