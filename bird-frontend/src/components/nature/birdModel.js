import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'

import { readToken } from '../../lib/themeTokens'
import { smoothstep, mulberry32 } from '../../lib/rng'

/**
 * Loading and re-dressing the hero bird.
 *
 * The bird is a real animated model rather than the assemblage of spheres and
 * cones it replaces — a stork from the three.js example assets, hand-animated by
 * someone who knew what a stork looks like in the air. What arrives in the file
 * is not what should appear on screen, though: it comes with baked vertex colour
 * from a different palette, no normals at all, and a scale in the hundreds. This
 * module turns it into a bird that belongs in *this* page.
 *
 * Everything here is derived from the file rather than hardcoded, so swapping in
 * Flamingo.glb or Parrot.glb needs no other change.
 */

/**
 * Wingspan after normalisation, in scene units.
 *
 * Sized against the hero band rather than picked: with the camera at z = 9.5 and
 * a 26° vertical field of view, roughly 4.4 units of scene height fill the band,
 * so a 2.6-unit span puts the bird at about 60% of the masthead's height. Big
 * enough to read as a bird rather than a speck, small enough to stay behind the
 * title rather than competing with it.
 */
const TARGET_SPAN = 2.6

/**
 * Extra yaw applied to the model, in radians.
 *
 * Which way a model "faces" is a convention its author chose and the file does
 * not record, so this was measured rather than guessed. The bounding box is
 * symmetric in X (X is the wingspan) and asymmetric in Z (Z runs nose to tail).
 * Which end of Z is the nose was settled by looking at the vertices: the +Z
 * extreme is a single vertex on the centreline held high — a beak tip — while
 * the −Z extreme is ten vertices spread eight units wide at body height, which
 * is a tail fan. So the nose is local +Z, and a quarter turn puts it on +X,
 * where a heading of yaw = 0 points.
 */
const MODEL_YAW_OFFSET = Math.PI / 2

/**
 * Measure a mesh in its rest pose.
 *
 * `Box3.setFromObject` is the obvious way to do this and it is wrong here.
 * It goes through `geometry.computeBoundingBox()`, which expands the box to
 * cover every morph target — and this model's morphs are full wingbeat poses.
 * The box therefore described the union of every position the wings ever reach,
 * came out more than four times the bird's actual extent, and normalising
 * against it shrank the stork to a speck. Measure the base positions only.
 */
function restPoseBounds(geometry) {
  const position = geometry.attributes.position
  const box = new THREE.Box3()
  const point = new THREE.Vector3()
  for (let i = 0; i < position.count; i++) {
    box.expandByPoint(point.fromBufferAttribute(position, i))
  }
  return box
}

/**
 * Paint countershading into the vertex colours.
 *
 * Nearly every bird is dark above and pale below. It is not decoration — it is
 * how a lit body avoids reading as a solid lump, since the sunlit top and shaded
 * underside cancel out into something closer to flat. Getting it right is most
 * of the difference between "a bird" and "a green model of a bird", and it costs
 * one pass over 358 vertices at load.
 *
 * Three ingredients:
 *
 *   - **Dorsal/ventral.** Straight from the surface normal's vertical component.
 *     This *is* countershading, expressed as the one geometric fact defining it.
 *   - **Wingtip darkening.** Primaries are darker than coverts on almost every
 *     species, because the pigment that colours them also stiffens them against
 *     wear. Absent from the old flat-green bird entirely.
 *   - **Jitter.** A few percent of seeded noise. Perfectly uniform shading is
 *     the "plastic" tell, and this is the same argument as the page's grain.
 *
 * The normals used are the *rest pose* ones. That is deliberate: plumage pattern
 * is fixed on the bird, and shading that swam around as the wings beat would be
 * worse than none.
 */
function paintCountershading(geometry) {
  // The file has POSITION, COLOR_0 and TEXCOORD_0 but no NORMAL, so GLTFLoader
  // computes them. Guard anyway — a different model may ship its own.
  if (!geometry.attributes.normal) geometry.computeVertexNormals()

  const normal = geometry.attributes.normal
  const position = geometry.attributes.position
  const count = position.count

  // Rest pose, for the same reason the scale normalisation uses it: the
  // geometry's own bounding box covers every morph target.
  const box = restPoseBounds(geometry)
  const halfSpan = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) / 2 || 1

  const rnd = mulberry32(0xb12d)
  const colors = new Float32Array(count * 3)

  for (let i = 0; i < count; i++) {
    // 1 at the belly, 0 at the back.
    const up = normal.getY(i)
    let tone = 1 - smoothstep(-0.3, 0.55, up)

    // Darker toward the wingtips.
    const spread = Math.abs(position.getX(i)) / halfSpan
    tone *= 1 - 0.45 * smoothstep(0.55, 0.95, spread)

    const jitter = 1 + (rnd() * 0.08 - 0.04)
    const value = (0.34 + 0.66 * tone) * jitter
    colors[i * 3] = value
    colors[i * 3 + 1] = value
    colors[i * 3 + 2] = value
  }

  // Overwrites the model's own COLOR_0, which carries a palette from a different
  // project entirely and is the single loudest "downloaded asset" signal.
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
}

/** Plumage. Exported so the scene can recolour it when the theme flips. */
export function makeBirdMaterial() {
  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(readToken('--color-canopy', '#0b3b25')),
    vertexColors: true, // multiplies the countershading in
    roughness: 0.86,
    metalness: 0,
    // Sheen is the whole reason this is MeshPhysicalMaterial rather than
    // MeshStandardMaterial. It is a grazing-angle scatter term built for cloth,
    // and feathers are the other thing it fits: without it a matte bird reads as
    // moulded plastic no matter how good the shape is.
    sheen: 0.26,
    sheenRoughness: 0.7,
    sheenColor: new THREE.Color(readToken('--color-sun', '#d97706')).multiplyScalar(0.34),
    // The old bird set flatShading: true. Faceting is a deliberate stylisation
    // and it is precisely the one that reads as "generic low-poly asset".
    flatShading: false,
  })
}

/** Re-read palette tokens after a theme change. */
export function refreshBirdMaterial(material) {
  material.color.set(readToken('--color-canopy', '#0b3b25'))
  material.sheenColor.set(readToken('--color-sun', '#d97706'))
  material.sheenColor.multiplyScalar(0.34)
  material.needsUpdate = true
}

/**
 * Load, normalise and re-dress the model.
 *
 * Resolves to null rather than rejecting if anything goes wrong: the caller
 * already has the canvas flock underneath, so a missing or broken model should
 * degrade to "no hero bird" and never to a broken page.
 */
export function loadBird(url) {
  return new Promise((resolve) => {
    new GLTFLoader().load(
      url,
      (gltf) => {
        try {
          resolve(prepare(gltf))
        } catch (error) {
          console.warn('[HeroBird] could not prepare the model', error)
          resolve(null)
        }
      },
      undefined,
      (error) => {
        console.warn('[HeroBird] could not load', url, error)
        resolve(null)
      },
    )
  })
}

function prepare(gltf) {
  const source = gltf.scene

  let mesh = null
  source.traverse((child) => {
    if (child.isMesh && !mesh) mesh = child
  })
  if (!mesh) throw new Error('no mesh in the glTF')

  const material = makeBirdMaterial()
  // The model's own material is unnamed and unused once replaced; dispose it
  // rather than leaking a GPU program for something never drawn.
  const original = mesh.material
  mesh.material = material
  if (original && original.dispose) original.dispose()

  paintCountershading(mesh.geometry)

  // Normalise scale and centre from the rest pose rather than hardcoding. The
  // stork arrives about 197 units across; Flamingo and Parrot differ, and
  // deriving it means the model is a one-line swap.
  const box = restPoseBounds(mesh.geometry)
  const size = box.getSize(new THREE.Vector3())
  // The widest horizontal axis is the wingspan — the model is symmetric across it.
  const span = Math.max(size.x, size.z) || 1
  const scale = TARGET_SPAN / span
  source.scale.setScalar(scale)

  const centre = box.getCenter(new THREE.Vector3()).multiplyScalar(scale)
  source.position.sub(centre)

  // A wrapper so the flight code can set position and rotation on one object
  // without fighting the recentring offset above.
  const root = new THREE.Group()
  root.add(source)

  const mixer = new THREE.AnimationMixer(source)
  const clip = gltf.animations?.[0] ?? null
  let action = null
  if (clip) {
    action = mixer.clipAction(clip)
    action.play()
    // Time is driven by hand, by scrubbing action.time — see the phase warp in
    // flightPath.js. The mixer is only ever advanced by zero.
    action.paused = true
  }

  return {
    root,
    mesh,
    material,
    mixer,
    action,
    clipDuration: clip?.duration ?? 0,
    yawOffset: MODEL_YAW_OFFSET,
    dispose() {
      mixer.stopAllAction()
      mesh.geometry.dispose()
      material.dispose()
    },
  }
}
