import { motion, useReducedMotion } from 'motion/react'

/**
 * A few feathers drifting down the page.
 *
 * Deliberately sparse — six, not sixty. Past roughly eight this stops reading as
 * "a feather caught the light" and starts reading as snow, which is a different
 * and much cheaper effect.
 *
 * Each feather falls, sways on a sine, and rotates slowly, with its own duration
 * and delay so they never sync up. Pure CSS transforms driven by `motion`, so it
 * runs on the compositor and costs nothing on the main thread.
 */

const FEATHER_PATH =
  'M6 0 C 9.5 3.5 11 8.5 10.5 14 C 10 19.5 8 24.5 6 28 C 4 24.5 2 19.5 1.5 14 ' +
  'C 1 8.5 2.5 3.5 6 0 Z M6 3 L6 26'

const COUNT = 6

/**
 * Seeded once at module scope, not per render.
 *
 * `Math.random()` in a render body — including inside useMemo — is impure and
 * React's lint rules reject it, correctly: under concurrent rendering the same
 * component can render twice and the feathers would jump. Seeding here happens
 * exactly once, and there is exactly one Feathers instance in the app.
 */
const FEATHERS = Array.from({ length: COUNT }, (_, i) => ({
  id: i,
  left: 6 + Math.random() * 88, // vw
  size: 10 + Math.random() * 12,
  duration: 26 + Math.random() * 26,
  delay: -Math.random() * 40, // negative: already mid-flight on first paint
  sway: 30 + Math.random() * 70,
  spin: (Math.random() < 0.5 ? -1 : 1) * (140 + Math.random() * 220),
  opacity: 0.1 + Math.random() * 0.14,
}))

export function Feathers() {
  const reduced = useReducedMotion()
  const feathers = FEATHERS

  if (reduced) return null

  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      {feathers.map((f) => (
        <motion.svg
          key={f.id}
          viewBox="0 0 12 28"
          width={f.size}
          height={f.size * 2.33}
          className="absolute top-0"
          style={{ left: `${f.left}vw`, color: 'var(--color-canopy)' }}
          initial={{ y: '-12vh', x: 0, rotate: 0, opacity: 0 }}
          animate={{
            y: '112vh',
            x: [0, f.sway, -f.sway * 0.6, 0],
            rotate: f.spin,
            opacity: [0, f.opacity, f.opacity, 0],
          }}
          transition={{
            duration: f.duration,
            delay: f.delay,
            repeat: Infinity,
            ease: 'linear',
            x: { duration: f.duration / 2, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' },
            opacity: { duration: f.duration, times: [0, 0.12, 0.85, 1], repeat: Infinity },
          }}
        >
          <path
            d={FEATHER_PATH}
            fill="currentColor"
            stroke="currentColor"
            strokeWidth="0.6"
            strokeLinecap="round"
          />
        </motion.svg>
      ))}
    </div>
  )
}

export default Feathers
