import { useEffect, useMemo, useRef, useState } from 'react'
import { motion, useReducedMotion, useScroll, useTransform } from 'motion/react'

import { buildForest, bucketWidth } from './forest'
import { SUN, FOREST_SEED, PARALLAX, DEPTH_MIX, SHAFT_PERIODS } from './atmosphere'

/**
 * The place the app is standing in.
 *
 * Four depth bands of generated forest, a dawn sky behind them, and light
 * leaking through the canopy. Without this the app is a grid of cards on a tint;
 * with it you are looking out from under a canopy.
 *
 * Two ideas do most of the work here, and both replace something the previous
 * version got wrong.
 *
 * **Depth comes from colour, not from opacity.** The old bands were the same
 * flat green at 7%, 10% and 9%, which is not what distance looks like: distant
 * things converge on the colour of the air in front of them, they do not merely
 * fade. Each band is now mixed toward --color-base in proportion to its
 * distance, and the alphas are nearly equal. Because the mix runs toward the
 * page colour, it inverts correctly in dark mode for free — no conditional.
 *
 * **The geometry is generated, not drawn.** See forest.js.
 */
export function Canopy() {
  const reduced = useReducedMotion()
  const { scrollY } = useScroll()
  const ref = useRef(null)
  const [size, setSize] = useState({ width: 1280, height: 800 })

  // Measure the container rather than the window: this element is fixed and
  // inset-0, but reading it directly means no assumption about that holding.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => {
      const rect = el.getBoundingClientRect()
      setSize((prev) =>
        // Bucketed width, so a drag does not thrash regeneration; raw height,
        // which only changes on rotate or a devtools resize.
        bucketWidth(rect.width) === bucketWidth(prev.width) && Math.abs(rect.height - prev.height) < 24
          ? prev
          : { width: rect.width, height: rect.height },
      )
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const w = bucketWidth(size.width)
  const h = Math.max(320, Math.round(size.height))

  // ~4ms at desktop widths, and only on a bucket boundary.
  const forest = useMemo(() => buildForest({ seed: FOREST_SEED, width: w, height: h }), [w, h])

  // Deeper layers move less; the fringe moves against the scroll, which is what
  // sells it as being in front of the viewport rather than behind it.
  const p = (distance) => (reduced ? 0 : distance)
  const farY = useTransform(scrollY, [0, 1200], [0, p(PARALLAX.ridge)])
  const midY = useTransform(scrollY, [0, 1200], [0, p(PARALLAX.trees)])
  const nearY = useTransform(scrollY, [0, 1200], [0, p(PARALLAX.near)])
  const fringeY = useTransform(scrollY, [0, 1200], [0, p(PARALLAX.fringe)])

  const viewBox = `0 0 ${w} ${h}`

  /**
   * Atmospheric perspective. `percent` is how much foliage colour survives the
   * air; the rest is the page. Far bands also get pushed a little toward the sun
   * colour, because haze scatters warm.
   */
  const band = (percent, warm = 0) => {
    const base = `color-mix(in oklab, var(--color-canopy) ${percent}%, var(--color-base))`
    return warm ? `color-mix(in oklab, ${base} ${100 - warm}%, var(--color-sun))` : base
  }

  return (
    <div ref={ref} aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      {/* Dawn sky. A vertical ramp for the sky itself, then one elliptical glow
          placed off-centre at the shared sun position — a glow centred at 50% is
          both the most overused hero background there is and directionless, and
          every shadow and rim light in the app keys off this same point. */}
      <div
        className="absolute inset-0"
        style={{
          background: `
            linear-gradient(
              to bottom,
              color-mix(in oklab, var(--color-canopy) 5%, var(--color-base)) 0%,
              var(--color-base) 40%,
              color-mix(in oklab, var(--color-sun) var(--sky-warm-mid), var(--color-base)) 82%,
              color-mix(in oklab, var(--color-sun) var(--sky-warm-low), var(--color-base)) 100%
            )`,
        }}
      />
      {/* The sun itself. Tight and weak on purpose: at 22% over a wide ellipse
          this was a peach blob across the bottom third of the page in light and
          olive sludge in dark. Warmth at the horizon should be something you
          notice only if you look for it. */}
      <div
        className="absolute inset-0"
        style={{
          background: `radial-gradient(
            38% 26% at ${SUN.x * 100}% ${SUN.y * 100}%,
            color-mix(in oklab, var(--color-sun) var(--sun-glow), transparent),
            transparent 70%
          )`,
        }}
      />

      {/* Light through the canopy. Three shafts on pairwise-coprime periods, so
          the combination does not repeat inside any session anyone will sit
          through. Blurred hard: a crisp shaft reads as a graphic, a soft one as
          light. */}
      {!reduced &&
        SHAFT_PERIODS.map((period, i) => (
          <motion.div
            key={period}
            className="absolute"
            style={{
              left: `${SUN.x * 100 - 26 + i * 17}%`,
              top: '-30%',
              width: `${13 + i * 5}%`,
              height: '150%',
              transformOrigin: '50% 0%',
              rotate: `${-24 + i * 15}deg`,
              filter: 'blur(28px)',
              // Strength is a per-theme token: dark mode's sun is a saturated
              // yellow and at the light value these three shafts smeared a
              // olive haze across the whole masthead.
              background: `linear-gradient(
                to bottom,
                color-mix(in oklab, var(--color-sun) calc(var(--shaft-strength) - ${i * 3}%), transparent),
                transparent 72%
              )`,
            }}
            animate={{ opacity: [0.3, 0.7, 0.3], scaleX: [1, 1.14, 1] }}
            transition={{ duration: period, repeat: Infinity, ease: 'easeInOut' }}
          />
        ))}

      {/* Far ridge. */}
      <motion.svg
        style={{ y: farY, willChange: 'transform' }}
        viewBox={viewBox}
        className="absolute inset-0 h-full w-full"
      >
        {/* fill via style, not the presentation attribute: color-mix() in an SVG
            fill= has been unreliable in Safari, where the CSSOM path is fine. */}
        <path d={forest.far} style={{ fill: band(DEPTH_MIX.ridge, 8), opacity: 'var(--band-ridge)' }} />
      </motion.svg>

      {/* Mid treeline. */}
      <motion.svg
        style={{ y: midY, willChange: 'transform' }}
        viewBox={viewBox}
        className="absolute inset-0 h-full w-full"
      >
        <path d={forest.mid} style={{ fill: band(DEPTH_MIX.trees, 4), opacity: 'var(--band-trees)' }} />
      </motion.svg>

      {/* Near trunks, thrown slightly out of focus. Depth of field on the
          nearest layer only is the strongest photographic cue available here,
          and it costs one rasterisation — the blur is baked once and then
          composited, because motion only animates transform. */}
      <motion.svg
        style={{ y: nearY, willChange: 'transform', filter: 'blur(1.3px)' }}
        viewBox={viewBox}
        className="absolute inset-0 h-full w-full"
      >
        <path d={forest.near} style={{ fill: band(DEPTH_MIX.near), opacity: 'var(--band-near)' }} />
      </motion.svg>

      {/* Canopy overhead, further out of focus because it is closer still. */}
      <motion.svg
        style={{ y: fringeY, willChange: 'transform', filter: 'blur(2px)' }}
        viewBox={viewBox}
        className="absolute inset-0 h-full w-full"
      >
        <path d={forest.fringe} style={{ fill: band(DEPTH_MIX.fringe), opacity: 'var(--band-fringe)' }} />
      </motion.svg>
    </div>
  )
}

export default Canopy
