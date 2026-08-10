import { makeNoise1D, fbm, wrapPi, approach, smoothstep, clamp } from '../../lib/rng'

/**
 * How the bird decides where to go.
 *
 * The version this replaces was a parametric curve: position, heading and bank
 * were each a function of the same normalised loop position `p`. That is easy to
 * write and it always looks wrong, for two reasons.
 *
 * It repeats. One circuit took 26 seconds and the 27th second was identical to
 * the 1st. Anything watching the header for a minute sees the loop, and a loop
 * is the single clearest signal that something is animated rather than alive.
 *
 * And the bank was decorative. `rotation.z = -0.42 * cos(p·2π)` rolls the bird
 * on a schedule that merely *correlates* with the turn, because both came from
 * `p`. Real banking is not correlated with turning, it is how turning happens:
 * a bird rolls, which points its lift sideways, which is the force that curves
 * its path. Getting the causation backwards is subtle and the eye still catches
 * it, because the roll and the turn drift out of agreement at the extremes.
 *
 * So this integrates instead. A noise field proposes a turn rate, the bird's
 * inertia resists it, the achieved turn rate sets the bank angle through the
 * standard coordinated-turn relation, and the heading advances the position.
 * Nothing is a function of a loop variable, so there is no loop.
 */

/** Metres-ish per second, in the scene's units. */
const SPEED = 1.55

/** Peak turn rate the wander can ask for, radians/second. */
const TURN_GAIN = 0.62

/**
 * Steepest bank, radians. A soaring bird in a relaxed turn sits around 20-30°;
 * without a cap the coordinated-turn relation happily asked for 86°, which is
 * an aerobatic manoeuvre, not a stork crossing a header.
 */
const MAX_ROLL = 0.52

/**
 * Soft bounds, in scene units. Not a wall — the bird is nudged back with a
 * force that grows with how far outside it is, so it banks around rather than
 * bouncing.
 */
const BOUNDS = { x: 4.3, y: 1.35, z: 1.6 }

/**
 * Where containment starts, as a fraction of BOUNDS. Well inside the edge: a
 * bird that only reacts once it is already out has to turn hard to get back,
 * and hard turns near the frame edge are exactly what looks mechanical.
 */
const SOFT_EDGE = 0.62

export function makeFlight(seed) {
  const yawNoise = makeNoise1D(seed + 1)
  const pitchNoise = makeNoise1D(seed + 2)

  const state = {
    x: -2.2,
    y: 0.15,
    z: 0,
    yaw: 0.25, // heading in the XZ plane
    pitch: 0,
    roll: 0,
    yawRate: 0,
    glide: 0,
  }

  /**
   * Advance one frame.
   *
   * `flapPhase` is 0..1 through the wingbeat and comes back in so thrust can be
   * tied to the downstroke; the caller owns the phase because it also drives the
   * animation clip.
   */
  return function step(t, dt, flapPhase) {
    // Two incommensurate periods over a 4-octave lattice. 23.9/7.3 ≈ 3.274 is
    // not a ratio of small integers, so the sum has no practical period — the
    // pattern would need about 3 minutes to come back into phase even
    // approximately, and the noise lattice itself is 256 long on top of that.
    let command =
      TURN_GAIN * (0.75 * fbm(yawNoise, t / 7.3, 4) + 0.45 * fbm(yawNoise, t / 23.9, 3))

    // Containment. Quadratic in the overshoot so it is imperceptible just past
    // the edge and firm well past it, and expressed as a *turn command* rather
    // than a position correction so the bird banks around the way it would if it
    // had decided to come back itself.
    // Horizontal only. Yaw steering cannot change altitude, so including y here
    // meant a climbing bird generated a permanent turn command it could never
    // satisfy: it spiralled at full rate and, because glide keys off turn rate,
    // held its wings out for 98% of the time while ascending out of frame.
    // Height is a pitch problem and is handled below.
    const over = Math.hypot(state.x / BOUNDS.x, state.z / BOUNDS.z)
    if (over > SOFT_EDGE) {
      const homeward = Math.atan2(-state.z, -state.x)
      const delta = wrapPi(homeward - state.yaw)
      // Authority is proportional to how far off the homeward heading it is, up
      // to a half-turn. An earlier version capped this at 1 radian, which meant
      // a bird heading *directly* away had no more corrective urge than one at
      // 57° off — so it sailed out to nearly twice the bound before coming back.
      const urgency = ((over - SOFT_EDGE) / (1 - SOFT_EDGE)) ** 2
      command += (delta / Math.PI) * urgency * 5.5
    }

    // Inertia. A bird cannot change its turn rate instantly; without this the
    // wander reads as twitching.
    state.yawRate = approach(state.yawRate, command, 2.4, dt)
    state.yaw += state.yawRate * dt

    // Coordinated turn: tan(roll) = v·ω / g. The bank is *derived from* the
    // achieved turn rate, so the two can never disagree.
    const targetRoll = clamp(Math.atan2(SPEED * state.yawRate * 6.2, 3.3), -MAX_ROLL, MAX_ROLL)
    state.roll = approach(state.roll, targetRoll, 3.2, dt)

    // Height. A wander, a nose-up bias when banking hard (a turning bird loses
    // vertical lift and must pitch up to hold height), and — the part that
    // actually keeps it on screen — a restoring term that pushes the nose down
    // when it is high and lifts it when low.
    const targetPitch =
      0.2 * fbm(pitchNoise, t / 11.7, 3) +
      0.16 * Math.abs(state.roll) -
      0.62 * clamp(state.y / BOUNDS.y, -1.6, 1.6)
    state.pitch = approach(state.pitch, targetPitch, 1.6, dt)

    /**
     * Glide fraction, 0 flapping .. 1 wings held.
     *
     * Real birds do not flap continuously: they beat through the straights and
     * hold the wings out through a turn, letting the bank do the work. Tying the
     * hold to turn rate gets that for free, and has the side effect of breaking
     * up the wingbeat rhythm so even the flapping never settles into a metronome.
     */
    const wantGlide = smoothstep(0.6, 1.05, Math.abs(state.yawRate) / TURN_GAIN)
    state.glide = approach(state.glide, wantGlide, 1.8, dt)

    // Thrust comes in pulses on the downstroke, so the bird surges slightly
    // rather than travelling at a constant rate. Gliding removes the pulse.
    const surge = 1 + 0.08 * Math.sin(2 * Math.PI * flapPhase) * (1 - state.glide)
    const v = SPEED * surge * (1 - 0.12 * state.glide)

    const cosPitch = Math.cos(state.pitch)
    state.x += Math.cos(state.yaw) * cosPitch * v * dt
    state.z += Math.sin(state.yaw) * cosPitch * v * dt * 0.55
    state.y += Math.sin(state.pitch) * v * dt
    // Gliding sheds height slowly, which is what makes a glide read as a glide
    // rather than as the animation pausing.
    state.y -= state.glide * 0.12 * dt

    return state
  }
}

/**
 * Wingbeat rate wander.
 *
 * Kept separate from the flight state because the caller needs it *before*
 * stepping the flight (the phase feeds into `step`), and threading it through
 * the return value would force an awkward one-frame lag.
 */
export function makeFlapRate(seed, baseHz, wander) {
  const noise = makeNoise1D(seed + 3)
  return (t) => baseHz * (1 + wander * fbm(noise, t / 17.3, 3))
}

/**
 * Fraction of *real time* spent on the downstroke.
 *
 * A real bird drives hard on the downstroke — the stroke that makes lift — and
 * recovers slowly with the wrist folded to cut drag, so the two halves of the
 * beat are nothing like equal in duration. About a third down, two thirds up.
 */
const DOWNSTROKE_TIME = 0.36

/** Fraction of the *clip* that is the downstroke. The stork clip is symmetric. */
const DOWNSTROKE_CLIP = 0.5

/**
 * Warp a normalised wingbeat phase so the downstroke is fast and the upstroke
 * slow.
 *
 * This is the biggest single thing separating a real wingbeat from an animated
 * one, and the old procedural bird got it exactly wrong — `sin(t · 2.4)` is
 * symmetric, so its wings took as long to come up as to go down.
 *
 * The first attempt here replaced that with sine phase distortion,
 * `φ + k·sin(2πφ)/2π`, which was also wrong and in a way worth recording:
 * sin(2πφ) is antisymmetric about φ = 0.5, so τ(0.5) = 0.5 identically, for
 * *any* k. The downstroke therefore still took exactly half the cycle no matter
 * how the constant was tuned. All the distortion did was speed up the ends of
 * each half-stroke and slow down their middles. A five-minute simulation
 * measuring the actual duty cycle is what surfaced it; reading the formula did
 * not.
 *
 * So: map time to clip explicitly, two linear segments meeting at the stroke
 * reversal. Monotone by construction, and the velocity step at the junction is
 * not an artefact — it is the power stroke ending, which is a real and abrupt
 * event in a wingbeat.
 */
export function warpPhase(phase) {
  const p = ((phase % 1) + 1) % 1
  return p < DOWNSTROKE_TIME
    ? (p / DOWNSTROKE_TIME) * DOWNSTROKE_CLIP
    : DOWNSTROKE_CLIP + ((p - DOWNSTROKE_TIME) / (1 - DOWNSTROKE_TIME)) * (1 - DOWNSTROKE_CLIP)
}
