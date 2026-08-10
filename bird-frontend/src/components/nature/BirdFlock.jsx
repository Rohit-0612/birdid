import { useEffect, useRef } from 'react'

import { readToken, subscribeToTheme } from '../../lib/themeTokens'
import { mulberry32 } from '../../lib/rng'
import { FLAP_HZ, FOREST_SEED } from './atmosphere'

/**
 * Birds, on a canvas, crossing the sky behind the interface.
 *
 * The brief for this was "it should feel like glancing at a sky, not like a
 * loading spinner", and the choices follow from that:
 *
 * - **Slow.** A bird takes 30-70 seconds to cross. Fast movement in peripheral
 *   vision reads as an alert; slow movement reads as weather.
 * - **Three depth layers.** Far birds are small, pale and slow; near birds are
 *   larger, darker and faster. Parallax does the rest.
 * - **Wings driven by a sine.** Each bird has its own phase and flap rate, so the
 *   flock never pulses in unison, which is the thing that makes CSS-animated birds
 *   look fake. Distant birds flap slower — they are bigger birds, further away.
 * - **Curved paths.** A quadratic bezier with a gentle vertical drift, so birds
 *   arc rather than sliding along a rail.
 *
 * Drawn on canvas rather than as DOM nodes because a dozen independently-animated
 * elements with transforms is a layout and compositing cost for no benefit; here
 * it is one element and one paint.
 */

/** Hz to radians/second — the flock integrates phase in radians. */
const rads = ([lo, hi]) => [lo * Math.PI * 2, hi * Math.PI * 2]

const LAYERS = [
  { count: 4, scale: [4, 6], speed: [0.10, 0.16], alpha: 0.20, flap: rads(FLAP_HZ.far) },
  { count: 3, scale: [7, 10], speed: [0.18, 0.28], alpha: 0.32, flap: rads(FLAP_HZ.mid) },
  { count: 2, scale: [12, 17], speed: [0.30, 0.45], alpha: 0.42, flap: rads(FLAP_HZ.near) },
]

/**
 * Seeded, like the rest of the ambient layer.
 *
 * This runs in an effect rather than a render body, so it was never the React
 * purity problem Feathers.jsx describes — but leaving it random made the claim
 * that the whole ambient layer is deterministic false, and with it the ability
 * to compare two screenshots of the same scroll position.
 *
 * Re-seeded per resize so a new layout gets a fresh-but-repeatable flock.
 */
let seed = FOREST_SEED ^ 0x4f10c
let random = mulberry32(seed)

const rand = (min, max) => min + random() * (max - min)

function makeBird(layer, width, height, spawnAcross) {
  const scale = rand(...layer.scale)
  const dir = random() < 0.5 ? 1 : -1
  return {
    // spawnAcross seeds the first frame mid-flight, so the sky is never empty
    // on load and then suddenly populated.
    x: spawnAcross ? rand(-0.1, 1.1) * width : dir > 0 ? -0.08 * width : 1.08 * width,
    y: rand(0.06, 0.72) * height,
    dir,
    scale,
    speed: rand(...layer.speed) * dir,
    drift: rand(-0.012, 0.012),
    flap: rand(...layer.flap),
    phase: random() * Math.PI * 2,
    alpha: layer.alpha * rand(0.8, 1.15),
  }
}

/**
 * One bird: two swept wings either side of a short body.
 *
 * The previous version drew a single unbroken stroke from wingtip to wingtip
 * through the middle, which at these sizes is a caret — a "^" floating in the
 * sky, with nothing between the wings. Three changes fix it:
 *
 *   - **A body.** A short filled taper at the centre. A distant bird is mostly
 *     silhouette, and the silhouette of a bird has mass in the middle.
 *   - **Swept wings.** Each wing curves back from the shoulder before rising to
 *     the tip, rather than running straight out. Birds do not fly with their
 *     wings square to the body.
 *   - **Asymmetric fold.** Wings sweep further back and shorten on the upstroke,
 *     matching the same fast-down/slow-up asymmetry the hero bird uses.
 */
function drawBird(ctx, bird, fold) {
  const { x, y, scale, dir } = bird

  // fold runs -1 (wings down) .. 1 (wings up).
  const up = Math.max(0, fold)
  const tipY = -fold * scale * 0.8
  // Wings shorten as they fold, and sweep back hardest at the top of the stroke.
  const tipX = scale * (1 - Math.abs(fold) * 0.3)
  const sweep = scale * (0.3 + up * 0.34)

  ctx.lineWidth = Math.max(1, scale * 0.17)

  for (const side of [-1, 1]) {
    const s = side * dir
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.quadraticCurveTo(
      x + s * sweep,
      y + scale * 0.16, // elbow dips below the shoulder line
      x + s * tipX,
      y + tipY,
    )
    ctx.stroke()
  }

  // Body: a small filled lozenge, longer than it is deep.
  ctx.beginPath()
  ctx.ellipse(x, y + scale * 0.06, scale * 0.3, scale * 0.13, 0, 0, Math.PI * 2)
  ctx.fill()
}

export function BirdFlock({ className = '', density = 1 }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d', { alpha: true })
    if (!ctx) return

    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    let birds = []
    let raf = null
    let last = performance.now()
    let width = 0
    let height = 0

    // Cached, and refreshed only when the theme actually changes.
    //
    // This used to call getComputedStyle() inside the frame loop — once per
    // frame, forever — which forces a style recalculation sixty times a second
    // to re-read a value that changes about twice a day. It showed up in a
    // profile as a Recalculate Style entry on every single frame.
    let ink = readToken('--color-canopy', '#0b3b25')

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const rect = canvas.parentElement.getBoundingClientRect()
      width = rect.width
      height = rect.height
      canvas.width = Math.max(1, width * dpr)
      canvas.height = Math.max(1, height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      // Reset the stream so the same viewport always yields the same flock.
      random = mulberry32(seed)
      birds = LAYERS.flatMap((layer) =>
        Array.from({ length: Math.max(1, Math.round(layer.count * density)) }, () =>
          makeBird(layer, width, height, true),
        ),
      )
    }

    function frame(now) {
      const dt = Math.min(now - last, 50) // clamp: a backgrounded tab must not teleport birds
      last = now
      ctx.clearRect(0, 0, width, height)
      ctx.strokeStyle = ink
      ctx.fillStyle = ink
      ctx.lineCap = 'round'

      for (const bird of birds) {
        bird.x += bird.speed * dt * 0.06
        bird.y += bird.drift * dt * 0.06
        bird.phase += (bird.flap * dt) / 1000

        // Recycle off-screen birds from the opposite edge at a new altitude.
        if ((bird.dir > 0 && bird.x > width * 1.12) || (bird.dir < 0 && bird.x < -width * 0.12)) {
          bird.x = bird.dir > 0 ? -width * 0.1 : width * 1.1
          bird.y = rand(0.06, 0.72) * height
        }

        ctx.globalAlpha = bird.alpha
        drawBird(ctx, bird, Math.sin(bird.phase))
      }
      ctx.globalAlpha = 1
      raf = requestAnimationFrame(frame)
    }

    function start() {
      if (raf == null) {
        last = performance.now()
        raf = requestAnimationFrame(frame)
      }
    }
    function stop() {
      if (raf != null) {
        cancelAnimationFrame(raf)
        raf = null
      }
    }

    resize()

    if (reduced) {
      // One static frame: the birds are there, they simply are not flying.
      ctx.strokeStyle = ink
      ctx.fillStyle = ink
      ctx.lineCap = 'round'
      for (const bird of birds) {
        ctx.globalAlpha = bird.alpha
        drawBird(ctx, bird, 0.35)
      }
    } else {
      start()
    }

    /** Draw one frame without animating — for a paused canvas that must repaint. */
    function repaint() {
      ctx.clearRect(0, 0, width, height)
      ctx.strokeStyle = ink
      ctx.fillStyle = ink
      ctx.lineCap = 'round'
      for (const bird of birds) {
        ctx.globalAlpha = bird.alpha
        drawBird(ctx, bird, reduced ? 0.35 : Math.sin(bird.phase))
      }
      ctx.globalAlpha = 1
    }

    // A background tab should cost nothing. Without this, requestAnimationFrame
    // keeps a throttled loop alive and the laptop fan notices. The same applies
    // once the hero has scrolled out of view, which happens far more often.
    let visible = !document.hidden
    let onScreen = true
    const sync = () => (visible && onScreen && !reduced ? start() : stop())

    const onVisibility = () => {
      visible = !document.hidden
      sync()
    }
    document.addEventListener('visibilitychange', onVisibility)

    const seen = new IntersectionObserver(
      ([entry]) => {
        onScreen = entry.isIntersecting
        sync()
      },
      { rootMargin: '80px' },
    )
    if (canvas.parentElement) seen.observe(canvas.parentElement)

    const observer = new ResizeObserver(resize)
    if (canvas.parentElement) observer.observe(canvas.parentElement)

    // Follow the theme. The flock strokes itself with --color-canopy, which
    // inverts between themes, so without this the birds keep the old palette
    // until the next resize.
    const unsubscribe = subscribeToTheme(() => {
      ink = readToken('--color-canopy', '#0b3b25')
      if (raf == null) repaint()
    })

    return () => {
      stop()
      document.removeEventListener('visibilitychange', onVisibility)
      seen.disconnect()
      observer.disconnect()
      unsubscribe()
    }
  }, [density])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 ${className}`}
    />
  )
}

export default BirdFlock
