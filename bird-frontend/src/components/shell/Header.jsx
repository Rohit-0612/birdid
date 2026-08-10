import { motion, useReducedMotion } from 'motion/react'
import { Mic, MicOff, Moon, Sun, Volume2, VolumeX } from 'lucide-react'

import { Chip } from '../primitives'
import { HeroBird } from '../nature/HeroBird'

/**
 * The masthead, and the sky the hero bird flies in.
 *
 * The bird used to be confined to the height of a title and a line of subtitle —
 * about 130 pixels — which at a believable wingspan left it the size of a
 * postage stamp. A real bird needs somewhere to be a bird, so the hero is now a
 * band with actual height and the masthead sits inside it. The birds still stop
 * at the bottom of this band: movement in the corner of your eye competes with
 * reading, and nothing should be flying behind a paragraph.
 */

const enter = (index, reduced) =>
  reduced
    ? {}
    : {
        initial: { opacity: 0, y: 8 },
        animate: { opacity: 1, y: 0 },
        transition: { duration: 0.5, delay: index * 0.04, ease: [0.22, 1, 0.36, 1] },
      }

export function Header({ health, speech, voice, theme, onToggleTheme }) {
  const reduced = useReducedMotion()

  return (
    <div className="relative">
      <HeroBird className="-inset-x-6 -bottom-2 [top:-1.5rem]" />

      <header className="relative flex min-h-[13rem] flex-wrap items-start justify-between gap-4 py-8 sm:min-h-[15rem]">
        <motion.div {...enter(0, reduced)}>
          <p className="text-micro font-medium uppercase text-(--color-ink-faint)">
            Field guide · runs on this machine
          </p>
          <h1 className="mt-2 font-display text-display text-(--color-ink)">
            Bird<span className="text-(--color-accent)">ID</span>
          </h1>
          <p className="mt-3 max-w-sm text-lede text-(--color-ink-soft)">
            Identify birds by photo or call, ask about them, collect them
          </p>
        </motion.div>

        <motion.div className="flex flex-wrap items-center gap-2" {...enter(1, reduced)}>
          {/* One capsule rather than five loose chips. The header was presenting
              seven separate objects competing with the title; grouping the
              status into a single surface makes it one. */}
          {health && (
            <div
              className="card-glass flex flex-wrap items-center gap-1.5 px-2.5 py-1.5"
              style={{ borderRadius: 'var(--radius-organic)' }}
            >
              <Chip title="Compute device the model is running on">{health.device_label}</Chip>
              <Chip title="Species the photo model can name">{health.num_species} species</Chip>
              {health.openset?.enabled ? (
                <Chip tone="high" title="Photos that are not one of the known species get rejected">
                  open-set on
                </Chip>
              ) : (
                <Chip tone="moderate" title="Run scripts/fit_openset.py to enable rejection">
                  open-set off
                </Chip>
              )}
              {!health.model_audited && (
                <Chip
                  tone="moderate"
                  title="This checkpoint has no recorded training split — see How it works"
                >
                  unaudited
                </Chip>
              )}
            </div>
          )}

          <div className="card-glass flex items-center gap-1.5 rounded-(--radius-pill) p-1">
            <button
              onClick={onToggleTheme}
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
              title={theme === 'dark' ? 'Dark: forest at night' : 'Light: mint paper'}
              className="cursor-pointer rounded-full p-2 text-(--color-ink-faint) transition-colors duration-200 hover:bg-(--color-sun)/15 hover:text-(--color-sun-ink)"
            >
              {theme === 'dark' ? <Moon size={16} strokeWidth={2} /> : <Sun size={16} strokeWidth={2} />}
            </button>

            <button
              onClick={speech.toggleMute}
              disabled={!speech.supported}
              aria-label={speech.muted ? 'Unmute the narrator' : 'Mute the narrator'}
              title={
                !speech.supported
                  ? 'This browser has no speech synthesis'
                  : speech.muted
                    ? 'Voice output is off'
                    : 'Voice output is on'
              }
              className={`cursor-pointer rounded-full p-2 transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-40 ${
                speech.muted
                  ? 'text-(--color-ink-faint) hover:text-(--color-ink-soft)'
                  : 'text-(--color-accent-hover) hover:bg-(--color-accent)/15'
              }`}
            >
              {speech.muted ? <VolumeX size={16} strokeWidth={2} /> : <Volume2 size={16} strokeWidth={2} />}
            </button>

            <button
              onMouseDown={voice.start}
              onMouseUp={voice.stop}
              onMouseLeave={() => voice.listening && voice.stop()}
              onTouchStart={(e) => {
                e.preventDefault()
                voice.start()
              }}
              onTouchEnd={voice.stop}
              disabled={!voice.supported}
              aria-label="Hold to speak a command"
              title={
                voice.supported
                  ? 'Hold to speak — or hold the space bar. Try “tell me about it”.'
                  : 'This browser has no speech recognition. Try Chrome or Safari.'
              }
              className={`cursor-pointer rounded-full p-2 transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-40 ${
                voice.listening
                  ? 'bg-(--color-accent) text-(--color-on-accent)'
                  : 'text-(--color-ink-faint) hover:bg-(--color-accent)/15 hover:text-(--color-accent-hover)'
              }`}
            >
              {voice.supported ? <Mic size={16} strokeWidth={2} /> : <MicOff size={16} strokeWidth={2} />}
            </button>
          </div>
        </motion.div>
      </header>
    </div>
  )
}

export default Header
