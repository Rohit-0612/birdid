import { Mic, MicOff, Moon, Sun, Volume2, VolumeX } from 'lucide-react'

/**
 * Theme, narrator and push-to-talk — the three global controls.
 *
 * Handlers, labels and titles are exactly those the old masthead used; only the
 * arrangement is new. `vertical` stacks them for the side rail, otherwise they
 * sit in a row for the mobile nav bar.
 */
export function Controls({ speech, voice, theme, onToggleTheme, vertical = false }) {
  const base =
    'grid size-10 cursor-pointer place-items-center rounded-full transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-40'

  return (
    <div className={`flex items-center gap-1.5 ${vertical ? 'flex-col' : ''}`}>
      <button
        onClick={onToggleTheme}
        aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        title={theme === 'dark' ? 'Dark: night mount' : 'Light: warm paper'}
        className={`${base} text-(--color-ink-faint) hover:bg-(--color-accent)/12 hover:text-(--color-accent-hover)`}
      >
        {theme === 'dark' ? <Moon size={17} strokeWidth={1.75} /> : <Sun size={17} strokeWidth={1.75} />}
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
        className={`${base} ${
          speech.muted
            ? 'text-(--color-ink-faint) hover:text-(--color-ink-soft)'
            : 'text-(--color-accent-hover) hover:bg-(--color-accent)/12'
        }`}
      >
        {speech.muted ? <VolumeX size={17} strokeWidth={1.75} /> : <Volume2 size={17} strokeWidth={1.75} />}
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
        className={`${base} ${
          voice.listening
            ? 'bg-(--color-accent) text-(--color-on-accent)'
            : 'border border-(--color-line) text-(--color-ink-soft) hover:border-(--color-accent)/60 hover:text-(--color-accent-hover)'
        }`}
      >
        {voice.supported ? <Mic size={17} strokeWidth={1.75} /> : <MicOff size={17} strokeWidth={1.75} />}
      </button>
    </div>
  )
}
