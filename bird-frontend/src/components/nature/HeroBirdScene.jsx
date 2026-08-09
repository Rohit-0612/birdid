import { useEffect, useRef } from 'react'
import * as THREE from 'three'

/**
 * A single low-poly bird, in WebGL, banking slowly across the hero.
 *
 * Procedural geometry rather than a downloaded model: there is no asset to
 * license, host or keep in the repo, the whole bird is about forty vertices, and
 * the wing hinge is a real transform rather than baked animation, so the flap can
 * share its sine with the 2D canvas flock and the two read as one flock.
 *
 * This module is only ever reached through a dynamic import (see HeroBird.jsx), so
 * Three.js lands in its own chunk and never blocks first paint.
 */

/** Tapered wing: a hinged quad pair, narrowing to the tip. */
function makeWing(material, side) {
  const shape = new THREE.Shape()
  shape.moveTo(0, -0.10)
  shape.quadraticCurveTo(0.55, -0.30, 1.35, -0.10)
  shape.quadraticCurveTo(0.95, 0.06, 0.55, 0.14)
  shape.quadraticCurveTo(0.28, 0.18, 0, 0.12)
  shape.closePath()

  const wing = new THREE.Mesh(new THREE.ShapeGeometry(shape, 8), material)
  wing.scale.x = side // mirror for the left wing
  // Pivot at the shoulder, so rotation folds the wing rather than sliding it.
  const hinge = new THREE.Group()
  hinge.add(wing)
  return hinge
}

function makeBird() {
  const bird = new THREE.Group()

  const body = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#1d3b2a'),
    roughness: 0.75,
    metalness: 0.05,
    flatShading: true,
  })
  const feather = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#24503a'),
    roughness: 0.85,
    metalness: 0.0,
    flatShading: true,
    side: THREE.DoubleSide,
  })

  // Body: a low-poly spindle. Few segments on purpose — the facets are the look.
  const torso = new THREE.Mesh(new THREE.SphereGeometry(0.30, 7, 5), body)
  torso.scale.set(1.9, 0.72, 0.72)
  bird.add(torso)

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.155, 6, 5), body)
  head.position.set(0.60, 0.10, 0)
  bird.add(head)

  const beak = new THREE.Mesh(new THREE.ConeGeometry(0.055, 0.26, 4), body)
  beak.rotation.z = -Math.PI / 2
  beak.position.set(0.80, 0.08, 0)
  bird.add(beak)

  const tail = new THREE.Mesh(new THREE.ConeGeometry(0.19, 0.62, 4), feather)
  tail.rotation.z = Math.PI / 2
  tail.position.set(-0.68, 0.03, 0)
  tail.scale.y = 0.45
  bird.add(tail)

  const right = makeWing(feather, 1)
  const left = makeWing(feather, -1)
  right.position.set(0.05, 0.10, 0.12)
  left.position.set(0.05, 0.10, -0.12)
  bird.add(right, left)

  return { bird, right, left }
}

/**
 * Mounts the scene into `container`. Returns a disposer.
 *
 * Kept module-private: this file is reached through React.lazy, and exporting a
 * non-component alongside the default breaks Fast Refresh for the whole module.
 */
function mountHeroBird(container, { reduced = false } = {}) {
  const width = container.clientWidth || 1
  const height = container.clientHeight || 1

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(width, height)
  renderer.setClearAlpha(0)
  container.appendChild(renderer.domElement)

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100)
  camera.position.set(0, 0, 6.2)

  // Soft daylight plus a warm rim in the palette's sun colour, so the bird sits in
  // the same light as the rest of the page.
  scene.add(new THREE.HemisphereLight(0xdff3e4, 0x2b3b31, 1.05))
  const key = new THREE.DirectionalLight(0xffffff, 1.5)
  key.position.set(2.5, 3.2, 2.6)
  scene.add(key)
  const rim = new THREE.DirectionalLight(0xd97706, 1.9)
  rim.position.set(-3.2, 1.4, -2.2)
  scene.add(rim)

  const { bird, right, left } = makeBird()
  scene.add(bird)

  let raf = null
  const clock = new THREE.Clock()

  function layout() {
    const w = container.clientWidth || 1
    const h = container.clientHeight || 1
    renderer.setSize(w, h)
    camera.aspect = w / h
    camera.updateProjectionMatrix()
  }

  function pose(t) {
    // A long lazy circuit: across, with a little rise and fall, banking into turns.
    const loop = 26 // seconds for one crossing
    const p = (t % loop) / loop
    const x = -4.6 + p * 9.2
    const y = 0.55 * Math.sin(p * Math.PI * 2) - 0.15
    const z = 0.9 * Math.sin(p * Math.PI * 2 + 1.1)

    bird.position.set(x, y, z)
    // Face the direction of travel, with a bank proportional to turn rate.
    bird.rotation.y = -0.35 + 0.5 * Math.cos(p * Math.PI * 2)
    bird.rotation.z = -0.42 * Math.cos(p * Math.PI * 2)
    bird.rotation.x = 0.10 * Math.sin(p * Math.PI * 2)

    // Wings: the same sine the canvas flock uses, so the rhythms match.
    const flap = Math.sin(t * 2.4)
    right.rotation.z = flap * 0.85
    left.rotation.z = -flap * 0.85
    // A touch of forward sweep on the downstroke reads as thrust.
    right.rotation.y = flap * 0.16
    left.rotation.y = -flap * 0.16
  }

  function frame() {
    pose(clock.getElapsedTime())
    renderer.render(scene, camera)
    raf = requestAnimationFrame(frame)
  }

  function start() {
    if (raf == null && !reduced) raf = requestAnimationFrame(frame)
  }
  function stop() {
    if (raf != null) {
      cancelAnimationFrame(raf)
      raf = null
    }
  }

  if (reduced) {
    // One posed frame: wings mid-beat, mid-arc. Present, not moving.
    pose(6.4)
    renderer.render(scene, camera)
  } else {
    start()
  }

  const onVisibility = () => (document.hidden ? stop() : start())
  document.addEventListener('visibilitychange', onVisibility)
  const observer = new ResizeObserver(layout)
  observer.observe(container)

  return function dispose() {
    stop()
    document.removeEventListener('visibilitychange', onVisibility)
    observer.disconnect()
    scene.traverse((object) => {
      if (object.geometry) object.geometry.dispose()
      if (object.material) object.material.dispose()
    })
    renderer.dispose()
    if (renderer.domElement.parentNode === container) {
      container.removeChild(renderer.domElement)
    }
  }
}

/** Thin React wrapper. Imported only by the lazy boundary in HeroBird.jsx. */
export default function HeroBirdScene() {
  const ref = useRef(null)

  useEffect(() => {
    const container = ref.current
    if (!container) return
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    let dispose = () => {}
    try {
      dispose = mountHeroBird(container, { reduced })
    } catch {
      // WebGL context creation can fail on old drivers or in software rendering.
      // The caller already renders the canvas flock underneath, so failing here
      // silently leaves a working hero rather than a blank hole.
    }
    return () => dispose()
  }, [])

  return <div ref={ref} aria-hidden="true" className="absolute inset-0" />
}
