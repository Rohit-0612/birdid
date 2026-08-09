import { motion, useReducedMotion, useScroll, useTransform } from 'motion/react'

/**
 * Layered foliage silhouettes that parallax against scroll.
 *
 * This is what gives the page a *place*. Without it the app is a grid of cards on
 * a tint; with it you are looking out from under a canopy. Three depth bands move
 * at different rates, which is the whole trick — parallax is read as distance.
 *
 * Hand-drawn SVG paths rather than images: they scale to any viewport, cost about
 * two kilobytes, and recolour themselves from the theme token.
 */

/** Distant ridgeline — barely there, moves least. */
const RIDGE =
  'M0 120 C 120 96 210 108 320 84 C 430 60 520 96 640 72 C 760 48 850 84 960 66 C 1070 48 1160 78 1280 60 L1280 200 L0 200 Z'

/** Mid-ground treeline. */
const TREES =
  'M0 150 L28 118 L44 138 L70 96 L92 130 L118 104 L140 140 L172 110 L196 144 L224 116 L250 148 ' +
  'L280 108 L306 142 L336 118 L362 150 L394 112 L420 146 L452 122 L478 152 L510 114 L538 148 ' +
  'L568 120 L596 150 L628 110 L656 144 L686 118 L714 150 L746 116 L774 146 L806 122 L834 152 ' +
  'L866 112 L894 146 L926 120 L954 150 L986 116 L1014 144 L1046 120 L1074 150 L1106 114 ' +
  'L1134 146 L1166 120 L1194 150 L1226 116 L1254 144 L1280 124 L1280 200 L0 200 Z'

/** Foreground leaves, hanging from the top edge. */
const LEAVES =
  'M0 0 L1280 0 L1280 26 C 1210 30 1180 66 1120 58 C 1060 50 1046 18 990 30 ' +
  'C 934 42 928 78 866 70 C 804 62 800 26 742 34 C 684 42 680 82 616 72 ' +
  'C 552 62 552 24 492 32 C 432 40 430 80 366 70 C 302 60 306 22 244 30 ' +
  'C 182 38 178 74 118 64 C 58 54 46 24 0 30 Z'

export function Canopy() {
  const reduced = useReducedMotion()
  const { scrollY } = useScroll()

  // Deeper layers move less. With reduced motion the transforms collapse to zero
  // and the silhouettes simply sit still.
  const ridgeY = useTransform(scrollY, [0, 1200], [0, reduced ? 0 : 40])
  const treesY = useTransform(scrollY, [0, 1200], [0, reduced ? 0 : 90])
  const leavesY = useTransform(scrollY, [0, 1200], [0, reduced ? 0 : -60])

  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      {/* Sky wash: a hint of warmth at the horizon, as though it were early. */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(120% 60% at 50% 100%, color-mix(in oklab, var(--color-sun) 10%, transparent), transparent 70%)',
        }}
      />

      <motion.svg
        style={{ y: ridgeY }}
        viewBox="0 0 1280 200"
        preserveAspectRatio="none"
        className="absolute inset-x-0 bottom-0 h-[38vh] w-full opacity-[0.07]"
      >
        <path d={RIDGE} fill="var(--color-canopy)" />
      </motion.svg>

      <motion.svg
        style={{ y: treesY }}
        viewBox="0 0 1280 200"
        preserveAspectRatio="none"
        className="absolute inset-x-0 bottom-0 h-[26vh] w-full opacity-[0.10]"
      >
        <path d={TREES} fill="var(--color-canopy)" />
      </motion.svg>

      <motion.svg
        style={{ y: leavesY }}
        viewBox="0 0 1280 100"
        preserveAspectRatio="none"
        className="absolute inset-x-0 top-0 h-[16vh] w-full opacity-[0.09]"
      >
        <path d={LEAVES} fill="var(--color-canopy)" />
      </motion.svg>
    </div>
  )
}

export default Canopy
