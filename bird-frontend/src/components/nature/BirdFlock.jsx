import { useEffect, useRef } from 'react'

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

const LAYERS = [
  { count: 4, scale: [4, 6], speed: [0.10, 0.16], alpha: 0.20, flap: [0.9, 1.4] },
  { count: 3, scale: [7, 10], speed: [0.18, 0.28], alpha: 0.32, flap: [1.3, 1.9] },
  { count: 2, scale: [12, 17], speed: [0.30, 0.45], alpha: 0.42, flap: [1.8, 2.6] },
]

const rand = (min, max) => min + Math.random() * (max - min)

function makeBird(layer, width, height, spawnAcross) {
  const scale = rand(...layer.scale)
  const dir = Math.random() < 0.5 ? 1 : -1
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
    phase: Math.random() * Math.PI * 2,
    alpha: layer.alpha * rand(0.8, 1.15),
  }
}

/** One bird: two tapering wings hinged at a body, folded by `fold`. */
function drawBird(ctx, bird, fold) {
  const { x, y, scale, dir } = bird
  // fold runs -1 (wings down) .. 1 (wings up); the tip rises and draws inward.
  const tipY = -fold * scale * 0.85
  const tipX = scale * (1 - Math.abs(fold) * 0.25)

  ctx.beginPath()
  ctx.moveTo(x - tipX * dir, y + tipY)
  // Shoulder control point gives the wing its curve rather than a straight V.
  ctx.quadraticCurveTo(x - scale * 0.35 * dir, y + scale * 0.14, x, y)
  ctx.quadraticCurveTo(x + scale * 0.35 * dir, y + scale * 0.14, x + tipX * dir, y + tipY)
  ctx.lineWidth = Math.max(1, scale * 0.16)
  ctx.stroke()
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

    // Read the ink colour from the theme so the flock follows light/dark without
    // being told. Falls back to a slate if the variable is somehow missing.
    const inkColour = () =>
      getComputedStyle(document.documentElement).getPropertyValue('--color-canopy').trim() ||
      '#0b3b25'

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
      ctx.strokeStyle = inkColour()
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
      ctx.strokeStyle = inkColour()
      ctx.lineCap = 'round'
      for (const bird of birds) {
        ctx.globalAlpha = bird.alpha
        drawBird(ctx, bird, 0.35)
      }
    } else {
      start()
    }

    // A background tab should cost nothing. Without this, requestAnimationFrame
    // keeps a throttled loop alive and the laptop fan notices.
    const onVisibility = () => (document.hidden || reduced ? stop() : start())
    document.addEventListener('visibilitychange', onVisibility)

    const observer = new ResizeObserver(resize)
    if (canvas.parentElement) observer.observe(canvas.parentElement)

    return () => {
      stop()
      document.removeEventListener('visibilitychange', onVisibility)
      observer.disconnect()
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
