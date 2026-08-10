import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react'

import { BirdFlock } from './BirdFlock'

/**
 * The hero: a canvas flock always, plus one WebGL bird when the machine can.
 *
 * The lazy boundary is the point of this file. Three.js is ~150 KB gzip, and it
 * must not sit in the entry chunk of an app whose first job is to show a photo
 * upload box. `lazy()` puts it in its own chunk that loads after first paint.
 *
 * The flock renders underneath regardless, so there are four graceful outcomes
 * rather than one failure mode:
 *   chunk still loading  -> flock
 *   no WebGL             -> flock
 *   model missing / 404  -> flock
 *   reduced motion       -> flock's static frame, and a single posed 3D bird
 */

const HeroBirdScene = lazy(() => import('./HeroBirdScene'))

function webglAvailable() {
  try {
    const canvas = document.createElement('canvas')
    return Boolean(
      window.WebGLRenderingContext &&
        (canvas.getContext('webgl2') || canvas.getContext('webgl')),
    )
  } catch {
    return false
  }
}

export function HeroBird({ className = '' }) {
  // Deferred to an effect: probing WebGL touches the DOM, and doing it during
  // render would run on the server in any future SSR setup.
  const [enable3D, setEnable3D] = useState(false)
  const [heroFlying, setHeroFlying] = useState(false)

  useEffect(() => {
    if (!webglAvailable()) return
    // Wait for the browser to go idle so the 3D chunk never competes with the
    // first paint or the initial /api/health round trip.
    const schedule = window.requestIdleCallback ?? ((fn) => setTimeout(fn, 400))
    const handle = schedule(() => setEnable3D(true))
    return () => window.cancelIdleCallback?.(handle)
  }, [])

  const onHeroReady = useCallback(() => setHeroFlying(true), [])

  // The render loop writes --bird-x/--bird-y here rather than on its own
  // container. Custom properties inherit down the tree, not sideways, and the
  // shadow below is a *sibling* of the WebGL canvas — writing them on the canvas
  // container would leave the shadow permanently at its fallback position.
  const wrapper = useRef(null)

  return (
    <div ref={wrapper} className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`}>
      {/* Thin the flock once the hero bird is actually in the air. The flock's
          job is to keep the sky from being empty; with a large bird crossing the
          same space, the full count reads as crowded rather than calm. */}
      <BirdFlock density={heroFlying ? 0.65 : 1} />

      {enable3D && (
        <Suspense fallback={null}>
          <HeroBirdScene onReady={onHeroReady} styleTarget={wrapper} />
        </Suspense>
      )}

      {/* A soft shadow tracking the bird across the page beneath it. Driven by
          CSS custom properties that the render loop writes directly, so it costs
          no React work; --bird-shadow fades it as the bird banks and presents
          less of itself to the light. */}
      {heroFlying && (
        <div
          className="absolute size-40 -translate-x-1/2 -translate-y-1/2 opacity-[calc(var(--bird-shadow,1)*var(--bird-shadow-max))]"
          style={{
            left: 'var(--bird-x, 50%)',
            top: 'calc(var(--bird-y, 50%) + 14%)',
            background:
              'radial-gradient(closest-side, color-mix(in oklab, var(--color-canopy) 55%, transparent), transparent)',
            filter: 'blur(14px)',
          }}
        />
      )}
    </div>
  )
}

export default HeroBird
