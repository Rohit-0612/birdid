import { useCallback, useEffect, useRef, useState } from 'react'

import { SideRail } from './components/shell/SideRail'
import { TopNav } from './components/shell/TopNav'
import { Hero } from './components/sections/Hero'
import { StatsBand } from './components/sections/StatsBand'
import { SectionFrame } from './components/sections/SectionFrame'
import { Footer } from './components/sections/Footer'
import { useTheme } from './hooks/useTheme'
import { useScrollSpy, scrollToSection } from './hooks/useScrollSpy'
import { IdentifyView } from './views/IdentifyView'
import { ListenView } from './views/ListenView'
import { GuideView } from './views/GuideView'
import { DeckView, RecentEncounters } from './views/DeckView'
import { ExplainView } from './views/ExplainView'
import { useSpeech } from './hooks/useSpeech'
import { useVoiceCommands } from './hooks/useVoiceCommands'
import { PHOTOS } from './lib/photos'
import * as api from './lib/api'

/**
 * One long page. Every tool is a section, all mounted at once; the nav, the side
 * rail and voice commands scroll to them rather than swapping them in.
 *
 * Two things a tab switch used to do implicitly are done explicitly here, so the
 * views behave exactly as before:
 *   · the deck refetched on every visit (it remounted) — now it is handed a new
 *     refreshToken each time its section scrolls into view;
 *   · "Open in field guide" landed on a freshly mounted guide, so the requested
 *     species always won over whatever was open — now the guide is re-keyed on
 *     each such request, which is the same remount.
 */
const SECTIONS = [
  { id: 'identify', label: 'Identify' },
  { id: 'listen', label: 'Listen' },
  { id: 'guide', label: 'Field guide' },
  { id: 'deck', label: 'Deck' },
  { id: 'explain', label: 'How it works' },
]
const SPY_IDS = ['top', ...SECTIONS.map((s) => s.id)]

/** What one photo gets you, beside the Identify tool. */
const IDENTIFY_RETURNS = [
  ['The species', 'With a confidence gauge and the taxonomy trail.'],
  ['Field marks', 'The details that confirm it, and every look-alike with how to tell them apart.'],
  ['Where it looked', 'A heatmap of the pixels that drove the answer.'],
  ['An honest no', 'When the photo isn’t one of the 200, a second model takes a look instead.'],
]
const RAIL_SECTIONS = [{ id: 'top', label: 'Top' }, ...SECTIONS]

export default function App() {
  const [health, setHealth] = useState(null)
  const [guideFocus, setGuideFocus] = useState(null)
  const [guideRequest, setGuideRequest] = useState(0)
  const [deckVisit, setDeckVisit] = useState(0)
  const [toast, setToast] = useState(null)

  const speech = useSpeech()
  const { theme, toggleTheme } = useTheme()
  // Refetch the deck whenever it comes into view, so a bird identified further
  // up the page is already on its card by the time you get there.
  const active = useScrollSpy(SPY_IDS, 'top', (id) => {
    if (id === 'deck') setDeckVisit((n) => n + 1)
  })

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

  // Retried while it fails: a hosted backend that has gone to sleep answers
  // only after it wakes, and the page should fill in by itself when it does.
  useEffect(() => {
    let cancelled = false
    let timer
    const attempt = (n) =>
      api
        .health()
        .then((h) => !cancelled && setHealth(h))
        .catch(() => {
          if (!cancelled && n < 24) timer = setTimeout(() => attempt(n + 1), 5000)
        })
    attempt(0)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  const goto = useCallback((next, payload) => {
    if (next === 'guide' && payload?.folder) {
      setGuideFocus(payload.folder)
      setGuideRequest((n) => n + 1)
    }
    scrollToSection(next)
  }, [])

  // Global commands are handled here; the rest go to the view that registered.
  const handleCommand = useCallback(
    (command) => {
      const { intent, target } = command
      if (intent === 'stop') {
        speech.cancel()
        return
      }
      if (intent === 'listen') return scrollToSection('listen')
      if (intent === 'lifeList') return scrollToSection('deck')
      if (intent === 'fieldGuide') return scrollToSection('guide')
      if (intent === 'compare' && target) {
        setToast(`Search the field guide for “${target}” and use the ⇆ buttons to compare.`)
        return scrollToSection('guide')
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

  const shell = { speech, voice, theme, onToggleTheme: toggleTheme }

  return (
    <div className="relative min-h-screen">
      <SideRail sections={RAIL_SECTIONS} active={active} {...shell} />

      <div className="lg:pl-(--rail)">
        <TopNav sections={SECTIONS} active={active} {...shell} />

        <main>
          <Hero health={health} />
          <StatsBand health={health} />

          <SectionFrame
            id="identify"
            title="Identify by photo"
            lede="Drop in a photo, paste one, or click to browse. Every bird it names files itself into your deck — no save button."
            aside={
              <dl className="divide-y divide-(--color-line) border-y border-(--color-line)">
                {IDENTIFY_RETURNS.map(([term, detail]) => (
                  <div key={term} className="py-4">
                    <dt className="font-display text-[1.05rem] font-bold text-(--color-ink)">{term}</dt>
                    <dd className="mt-1 text-caption text-(--color-ink-soft)">{detail}</dd>
                  </div>
                ))}
              </dl>
            }
          >
            <IdentifyView speech={speech} registerVoiceHandler={registerVoiceHandler} onGoto={goto} />
          </SectionFrame>

          <SectionFrame
            id="listen"
            title="Identify by call"
            lede="Record a few seconds of birdsong, upload a clip, or play one of the bundled recordings. BirdNET listens for about 6,500 species, including many the photo model has never seen."
            photo={PHOTOS.wren}
          >
            <ListenView speech={speech} />
          </SectionFrame>

          <SectionFrame
            id="guide"
            title="Field guide"
            lede="All 200 species the photo model knows. Search them, read their field marks, and put two side by side to see how to tell them apart."
            photo={PHOTOS.brilliant}
          >
            <GuideView key={guideRequest} focusFolder={guideFocus} speech={speech} />
          </SectionFrame>

          <SectionFrame
            id="deck"
            title="Your deck"
            lede="One card for every species you photograph. Complete the 200, and collect the birds beyond them that the second model names."
            photo={PHOTOS.rufous}
          >
            <div className="space-y-6">
              <DeckView speech={speech} onGoto={goto} refreshToken={deckVisit} />
              <RecentEncounters refreshToken={deckVisit} />
            </div>
          </SectionFrame>

          <SectionFrame
            id="explain"
            title="How it works"
            lede="What each model does, where its numbers come from, and where not to trust them."
            photo={PHOTOS.egret}
          >
            <ExplainView health={health} />
          </SectionFrame>
        </main>

        <Footer sections={SECTIONS} />
      </div>

      {/* ── Voice status ── */}
      {voice.listening && (
        <div className="fixed inset-x-0 bottom-6 z-50 flex justify-center px-4">
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
        <div className="fixed inset-x-0 bottom-6 z-50 flex justify-center px-4">
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
