import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { AudioLines, BookMarked, Bird, ScanSearch, ShieldQuestion } from 'lucide-react'

import { Canopy } from './components/nature/Canopy'
import { Feathers } from './components/nature/Feathers'
import { Header } from './components/shell/Header'
import { NavTabs } from './components/shell/NavTabs'
import { useTheme } from './hooks/useTheme'
import { IdentifyView } from './views/IdentifyView'
import { ListenView } from './views/ListenView'
import { GuideView } from './views/GuideView'
import { DeckView, RecentEncounters } from './views/DeckView'
import { ExplainView } from './views/ExplainView'
import { useSpeech } from './hooks/useSpeech'
import { useVoiceCommands } from './hooks/useVoiceCommands'
import * as api from './lib/api'

const VIEWS = [
  { id: 'identify', label: 'Identify', icon: ScanSearch },
  { id: 'listen', label: 'Listen', icon: AudioLines },
  { id: 'guide', label: 'Field guide', icon: Bird },
  { id: 'deck', label: 'Deck', icon: BookMarked },
  { id: 'explain', label: 'How it works', icon: ShieldQuestion },
]

export default function App() {
  const [view, setView] = useState('identify')
  const [health, setHealth] = useState(null)
  const [guideFocus, setGuideFocus] = useState(null)
  const [toast, setToast] = useState(null)

  const speech = useSpeech()
  const { theme, toggleTheme } = useTheme()

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
      if (intent === 'lifeList') return setView('deck')
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
      {/* The ambient layer: parallax canopy behind, feathers drifting through,
          grain over the lot. Fixed and inset-0 so children have a resolvable
          height, pointer-events-none so it never eats a click.

          `isolation: isolate` is load-bearing, not decoration: it gives this
          subtree its own stacking context so the grain's mix-blend-mode composites
          against the canopy and stops there. Without it the blend would reach
          through to the content layer and tint live text. */}
      <div
        className="pointer-events-none fixed inset-0 z-0 [isolation:isolate]"
        aria-hidden="true"
      >
        <Canopy />
        <Feathers />
        <div className="grain absolute inset-0" />
      </div>

      <div className="relative z-10 mx-auto max-w-[1500px] px-4 pb-16 sm:px-6">
        <Header
          health={health}
          speech={speech}
          voice={voice}
          theme={theme}
          onToggleTheme={toggleTheme}
        />

        <NavTabs views={VIEWS} value={view} onChange={setView} />

        <main>
          {/* mode="wait" so the outgoing view finishes before the next arrives —
              cross-fading two full dashboards at once looks like a glitch. Short
              enough (180ms) that switching never feels like waiting. */}
          <AnimatePresence mode="wait">
            <motion.div
              key={view}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
            >
              {view === 'identify' && (
                <IdentifyView
                  speech={speech}
                  registerVoiceHandler={registerVoiceHandler}
                  onGoto={goto}
                />
              )}
              {view === 'listen' && <ListenView speech={speech} />}
              {view === 'guide' && <GuideView focusFolder={guideFocus} speech={speech} />}
              {view === 'deck' && (
                <div className="space-y-6">
                  <DeckView speech={speech} onGoto={goto} />
                  <RecentEncounters />
                </div>
              )}
              {view === 'explain' && <ExplainView health={health} />}
            </motion.div>
          </AnimatePresence>
        </main>

        <footer className="mt-12 border-t border-(--color-line) pt-5 text-xs text-(--color-ink-faint)">
          EfficientNetV2-S on CUB-200-2011 · calls by BirdNET · notes by a local language model,
          not expert-verified · everything runs on this machine
        </footer>
      </div>

      {/* ── Voice status ── */}
      {voice.listening && (
        <div className="fixed inset-x-0 bottom-6 z-30 flex justify-center px-4">
          <div className="card-glass flex items-center gap-3 rounded-(--radius-pill) px-5 py-3 shadow-[var(--shadow-ambient)]">
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
            className="card-glass max-w-md cursor-pointer rounded-(--radius-pill) px-4 py-3 text-caption text-(--color-ink-soft) shadow-[var(--shadow-ambient)]"
          >
            {voice.error || toast}
          </button>
        </div>
      )}
    </div>
  )
}
