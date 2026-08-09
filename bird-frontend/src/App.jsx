import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AudioLines, BookMarked, Bird, Mic, MicOff, ScanSearch, ShieldQuestion,
  Volume2, VolumeX,
} from 'lucide-react'

import DotField from './components/DotField'
import { Chip } from './components/primitives'
import { IdentifyView } from './views/IdentifyView'
import { ListenView } from './views/ListenView'
import { GuideView } from './views/GuideView'
import { LifeListView } from './views/LifeListView'
import { ExplainView } from './views/ExplainView'
import { useSpeech } from './hooks/useSpeech'
import { useVoiceCommands } from './hooks/useVoiceCommands'
import * as api from './lib/api'

const VIEWS = [
  { id: 'identify', label: 'Identify', icon: ScanSearch },
  { id: 'listen', label: 'Listen', icon: AudioLines },
  { id: 'guide', label: 'Field guide', icon: Bird },
  { id: 'life', label: 'Life list', icon: BookMarked },
  { id: 'explain', label: 'How it works', icon: ShieldQuestion },
]

export default function App() {
  const [view, setView] = useState('identify')
  const [health, setHealth] = useState(null)
  const [guideFocus, setGuideFocus] = useState(null)
  const [toast, setToast] = useState(null)

  const speech = useSpeech()

  // The active view registers a handler for the commands it owns. Held in a ref
  // so a spoken command dispatches as an event rather than as state a child has
  // to watch — which is both simpler to follow and avoids re-render cascades.
  const viewVoiceHandler = useRef(null)
  const registerVoiceHandler = useCallback((handler) => {
    viewVoiceHandler.current = handler
    return () => {
      if (viewVoiceHandler.current === handler) viewVoiceHandler.current = null
    }
  }, [])

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  const goto = useCallback((next, payload) => {
    setView(next)
    if (next === 'guide' && payload?.folder) setGuideFocus(payload.folder)
  }, [])

  // Global commands are handled here; the rest go to the active view.
  const handleCommand = useCallback(
    (command) => {
      const { intent, target } = command
      if (intent === 'stop') {
        speech.cancel()
        return
      }
      if (intent === 'listen') return setView('listen')
      if (intent === 'lifeList') return setView('life')
      if (intent === 'fieldGuide') return setView('guide')
      if (intent === 'compare' && target) {
        setToast(`Search the field guide for “${target}” and use the ⇆ buttons to compare.`)
        return setView('guide')
      }

      if (viewVoiceHandler.current) {
        viewVoiceHandler.current(command)
      } else {
        setToast(`Heard “${command.transcript}” — no command for that on this screen.`)
      }
    },
    [speech],
  )

  const voice = useVoiceCommands(handleCommand)

  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(null), 3200)
    return () => clearTimeout(timer)
  }, [toast])

  return (
    <div className="relative min-h-screen">
      {/* DotField measures its parent and sizes itself to 100% of it, so it needs
          a container with a resolvable height — inside an auto-height wrapper it
          would compute to zero and render nothing. Fixed and inset-0 gives it the
          viewport, and pointer-events-none keeps it from eating clicks (it tracks
          the cursor on window, so it still reacts). */}
      <div className="pointer-events-none fixed inset-0 z-0" aria-hidden="true">
        <DotField />
      </div>

      <div className="relative z-10 mx-auto max-w-[1500px] px-4 pb-16 sm:px-6">
        <Header health={health} speech={speech} voice={voice} />

        <nav className="mb-6 flex gap-1 overflow-x-auto pb-1" aria-label="Views">
          {VIEWS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              aria-current={view === id ? 'page' : undefined}
              className={`inline-flex shrink-0 cursor-pointer items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors duration-200 ${
                view === id
                  ? 'bg-(--color-accent)/15 text-(--color-accent-bright)'
                  : 'text-(--color-ink-faint) hover:bg-(--color-surface)/60 hover:text-(--color-ink-soft)'
              }`}
            >
              <Icon size={15} strokeWidth={2} />
              {label}
            </button>
          ))}
        </nav>

        <main>
          {view === 'identify' && (
            <IdentifyView
              speech={speech}
              registerVoiceHandler={registerVoiceHandler}
              onGoto={goto}
            />
          )}
          {view === 'listen' && <ListenView speech={speech} />}
          {view === 'guide' && <GuideView focusFolder={guideFocus} speech={speech} />}
          {view === 'life' && <LifeListView />}
          {view === 'explain' && <ExplainView health={health} />}
        </main>

        <footer className="mt-12 border-t border-(--color-line) pt-5 text-xs text-(--color-ink-faint)">
          EfficientNetV2-S on CUB-200-2011 · calls by BirdNET · notes by a local language model,
          not expert-verified · everything runs on this machine
        </footer>
      </div>

      {/* ── Voice status ── */}
      {voice.listening && (
        <div className="fixed inset-x-0 bottom-6 z-30 flex justify-center px-4">
          <div className="flex items-center gap-3 rounded-full border border-(--color-accent)/40 bg-(--color-surface)/95 px-5 py-3 shadow-2xl backdrop-blur">
            <span className="relative flex size-3">
              <span className="absolute inline-flex size-3 animate-ping rounded-full bg-(--color-accent) opacity-75" />
              <span className="relative inline-flex size-3 rounded-full bg-(--color-accent)" />
            </span>
            <span className="text-sm text-(--color-ink)">
              {voice.interim || 'Listening… release to send'}
            </span>
          </div>
        </div>
      )}

      {(toast || voice.error) && (
        <div className="fixed inset-x-0 bottom-6 z-30 flex justify-center px-4">
          <button
            onClick={() => {
              setToast(null)
              voice.clearError()
            }}
            className="max-w-md cursor-pointer rounded-xl border border-(--color-line) bg-(--color-surface)/95 px-4 py-3 text-sm text-(--color-ink-soft) shadow-2xl backdrop-blur"
          >
            {voice.error || toast}
          </button>
        </div>
      )}
    </div>
  )
}

function Header({ health, speech, voice }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-4 py-7">
      <div>
        <h1 className="font-display text-3xl leading-none text-(--color-ink) sm:text-4xl">
          Bird<span className="text-(--color-accent)">ID</span>
        </h1>
        <p className="mt-2 text-sm text-(--color-ink-faint)">
          Identify birds by photo or call, ask about them, keep a life list
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {health && (
          <>
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
              <Chip tone="moderate" title="This checkpoint has no recorded training split — see How it works">
                unaudited
              </Chip>
            )}
          </>
        )}

        {/* ── Voice controls ── */}
        <div className="flex items-center gap-1.5 rounded-full border border-(--color-line) bg-(--color-surface)/60 p-1">
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
                : 'text-(--color-accent-bright) hover:bg-(--color-accent)/15'
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
                ? 'bg-(--color-accent) text-white'
                : 'text-(--color-ink-faint) hover:bg-(--color-accent)/15 hover:text-(--color-accent-bright)'
            }`}
          >
            {voice.supported ? <Mic size={16} strokeWidth={2} /> : <MicOff size={16} strokeWidth={2} />}
          </button>
        </div>
      </div>
    </header>
  )
}
