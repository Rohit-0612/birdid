import { Suspense, lazy, useEffect, useState } from 'react'

import { BirdFlock } from './BirdFlock'

/**
 * The hero: a canvas flock always, plus one WebGL bird when the machine can.
 *
 * The lazy boundary is the point of this file. Three.js is ~150 KB gzip, and it
 * must not sit in the entry chunk of an app whose first job is to show a photo
 * upload box. `lazy()` puts it in its own chunk that loads after first paint.
 *
 * The flock renders underneath regardless, so there are three graceful outcomes
 * rather than one failure mode:
 *   chunk still loading  -> flock
 *   no WebGL             -> flock
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

  useEffect(() => {
    if (!webglAvailable()) return
    // Wait for the browser to go idle so the 3D chunk never competes with the
    // first paint or the initial /api/health round trip.
    const schedule = window.requestIdleCallback ?? ((fn) => setTimeout(fn, 400))
    const handle = schedule(() => setEnable3D(true))
    return () => window.cancelIdleCallback?.(handle)
  }, [])

  return (
    <div className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`}>
      <BirdFlock />
      {enable3D && (
        <Suspense fallback={null}>
          <HeroBirdScene />
        </Suspense>
      )}
    </div>
  )
}

export default HeroBird
