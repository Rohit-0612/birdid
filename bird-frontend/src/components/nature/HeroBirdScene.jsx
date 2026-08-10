import { useEffect, useRef } from 'react'
import * as THREE from 'three'

import { loadBird, refreshBirdMaterial } from './birdModel'
import { makeFlight, makeFlapRate, warpPhase } from './flightPath'
import { readToken, subscribeToTheme, prefersReducedMotion } from '../../lib/themeTokens'
import { SUN_DIR_X, HERO_FLAP_HZ, HERO_FLAP_WANDER, FOREST_SEED } from './atmosphere'

/**
 * The hero bird, in WebGL.
 *
 * Reached only through a dynamic import (see HeroBird.jsx), so three.js lands in
 * its own chunk and never blocks first paint. If anything here fails — no WebGL,
 * no model, an old driver — the caller is already drawing the canvas flock
 * underneath, so the correct failure mode is to leave quietly.
 *
 * The bird's shape and wingbeat come from birdModel.js; where it goes comes from
 * flightPath.js. What is left here is the camera, the light, and the loop.
 */

const MODEL_URL = '/models/stork.glb'

/** Mounts into `container`. Returns a disposer. */
function mountHeroBird(container, { reduced = false, styleTarget = null, onReady = null } = {}) {
  const width = container.clientWidth || 1
  const height = container.clientHeight || 1

  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    // MSAA is a real cost and almost invisible on a 2× display, where the
    // downsample is already doing the work.
    antialias: window.devicePixelRatio < 2,
    powerPreference: 'low-power',
  })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(width, height)
  renderer.setClearAlpha(0)
  // Without tone mapping, bright lights clip to flat white patches — the look
  // usually described as "CGI clay". ACES rolls the highlights off the way film
  // does, and it is the cheapest single improvement to how rendered things sit
  // next to photographs.
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.05
  container.appendChild(renderer.domElement)

  const scene = new THREE.Scene()

  // A long lens, well back, rather than a wide one up close. Wildlife is shot at
  // 400mm and the flattened perspective is part of how those images read; a wide
  // lens exaggerates the near wing and quietly says "rendered".
  const camera = new THREE.PerspectiveCamera(26, width / height, 0.1, 100)
  camera.position.set(0, 0, 9.5)

  /* ── light ──
     One sun, shared with the rest of the page. SUN_DIR_X is the same value the
     canopy's glow and light shafts are positioned from, so the bird is lit from
     where the page's light visibly comes from. The old scene had a white key
     from the upper right and an amber rim at intensity 1.9 from the left, which
     is a studio setup, not an outdoor one. */
  const skyColour = new THREE.Color(readToken('--color-base', '#f4faf5'))
  const groundColour = new THREE.Color(readToken('--color-canopy', '#0b3b25'))

  const hemi = new THREE.HemisphereLight(skyColour, groundColour, 0.55)
  scene.add(hemi)

  const key = new THREE.DirectionalLight(0xfff4e2, 1.1)
  key.position.set(SUN_DIR_X * 3.4, 2.6, 2.0)
  scene.add(key)

  // Light bouncing up off the canopy. Weak, green, and from below — the fill a
  // bird actually gets over a forest.
  const bounce = new THREE.DirectionalLight(groundColour, 0.24)
  bounce.position.set(-SUN_DIR_X * 2.0, -2.2, 1.0)
  scene.add(bounce)

  // Haze. Fog coloured as the page means distance dissolves the bird into the
  // background instead of into grey.
  scene.fog = new THREE.Fog(skyColour, 9, 24)

  let bird = null
  let raf = null
  let running = false
  let disposed = false

  // THREE.Clock is deprecated in this version and logs on construction; Timer is
  // the replacement and has to be advanced explicitly each frame.
  const timer = new THREE.Timer()
  const step = makeFlight(FOREST_SEED)
  const flapRate = makeFlapRate(FOREST_SEED, HERO_FLAP_HZ, HERO_FLAP_WANDER)
  let phase = 0

  function layout() {
    const w = container.clientWidth || 1
    const h = container.clientHeight || 1
    renderer.setSize(w, h)
    camera.aspect = w / h
    camera.updateProjectionMatrix()
  }

  /** One frame of flight. `dt` is clamped so a backgrounded tab cannot teleport it. */
  function advance(dt) {
    const t = timer.getElapsed()
    const rate = flapRate(t)

    // Our own phase, advanced in real time, rather than letting the mixer run.
    // Everything about the wingbeat — the asymmetric duty cycle, the glide hold —
    // is a function of this, so it has to be owned here.
    phase = (phase + rate * dt) % 1

    const state = step(t, dt, phase)

    // Gliding freezes the wings near the top of the upstroke, where a soaring
    // bird actually holds them, instead of wherever the clip happened to be.
    const held = 0.62
    const effective = state.glide > 0.001 ? phase + (held - phase) * state.glide : phase

    if (bird?.action) {
      bird.action.time = warpPhase(effective) * bird.clipDuration
      // Scrub, do not advance: update(0) applies the pose at the time we set.
      // mixer.setTime() would issue an internal delta, and deltas go negative
      // every time the phase wraps.
      bird.mixer.update(0)
    }

    if (bird?.root) {
      // Body bob: the torso rises as the wings drive down. Applied as pure
      // translation, never rotation — this model is a single morph-target mesh
      // with no skeleton, so there is no head bone to counter-rotate, and a bird
      // that nodded in time with its wings would look far worse than one that
      // simply rises and falls with its head held level.
      const bob = 0.03 * Math.sin(2 * Math.PI * (phase - 0.12)) * (1 - state.glide)

      bird.root.position.set(state.x, state.y + bob, state.z)
      bird.root.rotation.set(
        state.pitch,
        bird.yawOffset - state.yaw,
        // Bank. The sign is negative because a left turn (increasing yaw)
        // requires a left roll, and three.js Z-rotation is the other way round.
        -state.roll,
        'YXZ',
      )
    }

    // Hand the bird's screen position back to the page so a soft shadow can
    // track it. Written straight to CSS custom properties rather than React
    // state — this runs at 60Hz and must never touch the render tree.
    const target = styleTarget ?? container
    // The camera's world matrices are refreshed inside renderer.render(), which
    // runs *after* this. Projecting against a stale matrixWorldInverse divides by
    // a perspective w of nearly zero and returns values in the thousands of
    // percent — or -Infinity on the very first frame, when the matrix is still
    // identity and w is exactly zero.
    camera.updateMatrixWorld()
    const projected = new THREE.Vector3(state.x, state.y, state.z).project(camera)
    target.style.setProperty('--bird-x', `${(projected.x * 0.5 + 0.5) * 100}%`)
    target.style.setProperty('--bird-y', `${(-projected.y * 0.5 + 0.5) * 100}%`)
    target.style.setProperty('--bird-shadow', `${0.5 + 0.5 * Math.cos(state.roll)}`)
  }

  function frame() {
    timer.update()
    // Clamped: a tab that was backgrounded for a minute must not teleport the
    // bird a minute's worth of flight on its first frame back.
    const dt = Math.min(timer.getDelta(), 0.05)
    advance(dt)
    renderer.render(scene, camera)
    raf = requestAnimationFrame(frame)
  }

  function start() {
    if (raf == null && running && !reduced && !disposed) {
      timer.update() // discard the time spent paused
      raf = requestAnimationFrame(frame)
    }
  }
  function stop() {
    if (raf != null) {
      cancelAnimationFrame(raf)
      raf = null
    }
  }

  /** Play/pause driven by both tab visibility and whether the hero is on screen. */
  let visible = !document.hidden
  let onScreen = true
  const sync = () => {
    running = visible && onScreen
    if (running) start()
    else stop()
  }

  const onVisibility = () => {
    visible = !document.hidden
    sync()
  }
  document.addEventListener('visibilitychange', onVisibility)

  // The hero sits at the top of a long page. Once it has scrolled away there is
  // no reason to keep a WebGL loop running, and this is a bigger practical win
  // than the visibility check — people scroll far more often than they switch tabs.
  const observer = new IntersectionObserver(
    ([entry]) => {
      onScreen = entry.isIntersecting
      sync()
    },
    { rootMargin: '80px' },
  )
  observer.observe(container)

  const resize = new ResizeObserver(layout)
  resize.observe(container)

  const unsubscribeTheme = subscribeToTheme(() => {
    if (disposed) return
    const sky = new THREE.Color(readToken('--color-base', '#f4faf5'))
    const ground = new THREE.Color(readToken('--color-canopy', '#0b3b25'))
    hemi.color.copy(sky)
    hemi.groundColor.copy(ground)
    bounce.color.copy(ground)
    scene.fog.color.copy(sky)
    if (bird) refreshBirdMaterial(bird.material)
    // A paused scene still has to repaint, or the old palette stays on screen
    // until something else happens to trigger a frame.
    if (raf == null) renderer.render(scene, camera)
  })

  loadBird(MODEL_URL).then((loaded) => {
    if (disposed) {
      loaded?.dispose()
      return
    }
    if (!loaded) return
    bird = loaded
    scene.add(bird.root)

    if (reduced) {
      // One posed frame: mid-downstroke, banking gently, off-centre. Present,
      // not moving.
      phase = 0.18
      advance(0)
      renderer.render(scene, camera)
    } else {
      sync()
    }
    onReady?.()
  })

  return function dispose() {
    disposed = true
    stop()
    document.removeEventListener('visibilitychange', onVisibility)
    observer.disconnect()
    resize.disconnect()
    unsubscribeTheme()
    bird?.dispose()
    scene.traverse((object) => {
      object.geometry?.dispose()
      if (object.material && object.material.dispose) object.material.dispose()
    })
    renderer.dispose()
    // Free the GL context outright. Browsers cap how many can be live at once,
    // and React StrictMode mounts effects twice in development.
    renderer.forceContextLoss()
    if (renderer.domElement.parentNode === container) {
      container.removeChild(renderer.domElement)
    }
  }
}

/** Thin React wrapper. Imported only by the lazy boundary in HeroBird.jsx. */
export default function HeroBirdScene({ onReady, styleTarget }) {
  const ref = useRef(null)

  useEffect(() => {
    const container = ref.current
    if (!container) return

    let live = true
    let dispose = () => {}
    try {
      dispose = mountHeroBird(container, {
        reduced: prefersReducedMotion(),
        styleTarget: styleTarget?.current ?? null,
        // A plain callback rather than a DOM CustomEvent. The event version had
        // to be dispatched on, and listened for on, the same node across an
        // async model load and StrictMode's double mount; a guarded closure says
        // the same thing with nothing to get out of step.
        onReady: () => live && onReady?.(),
      })
    } catch (error) {
      // Context creation still fails on old drivers and under software
      // rendering. The flock is already underneath, so this leaves a working
      // hero rather than a hole.
      console.warn('[HeroBird] WebGL unavailable', error)
    }
    return () => {
      live = false
      dispose()
    }
  }, [onReady, styleTarget])

  return <div ref={ref} aria-hidden="true" className="absolute inset-0" />
}
