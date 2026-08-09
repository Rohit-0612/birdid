import { useCallback, useEffect, useState } from 'react'
import { Bird } from 'lucide-react'

import { Dropzone } from '../components/Dropzone'
import { ChatPanel } from '../components/ChatPanel'
import { SpeciesCard, SpeciesCardSkeleton } from '../components/SpeciesCard'
import { EmptyState } from '../components/primitives'
import * as api from '../lib/api'

/**
 * The main view: photo in, identification out, voice and chat on top.
 *
 * Grad-CAM is fetched lazily on first toggle rather than with every
 * identification — it costs a second backward pass and most identifications are
 * never inspected that closely.
 */
export function IdentifyView({ speech, registerVoiceHandler, onGoto }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [saved, setSaved] = useState(false)

  const [gradcamUrl, setGradcamUrl] = useState(null)
  const [showGradcam, setShowGradcam] = useState(false)

  const [chatOpen, setChatOpen] = useState(false)
  const [pendingQuestion, setPendingQuestion] = useState(null)

  const identify = useCallback(async (chosen) => {
    setBusy(true)
    setError(null)
    setResult(null)
    setSaved(false)
    setGradcamUrl(null)
    setShowGradcam(false)
    try {
      setResult(await api.identify(chosen))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }, [])

  const handleFile = useCallback(
    (chosen) => {
      setFile(chosen)
      setPreview(URL.createObjectURL(chosen))
      identify(chosen)
    },
    [identify],
  )

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setGradcamUrl(null)
    setShowGradcam(false)
    setSaved(false)
  }

  const save = useCallback(async () => {
    if (!result || saved) return
    try {
      await api.saveSighting(result, { file })
      setSaved(true)
    } catch (err) {
      setError(`Could not save: ${err.message}`)
    }
  }, [result, saved, file])

  const toggleGradcam = useCallback(async () => {
    if (gradcamUrl) {
      setShowGradcam((v) => !v)
      return
    }
    if (!file) return
    setBusy(true)
    try {
      const payload = await api.gradcam(file)
      setGradcamUrl(payload.url)
      setShowGradcam(true)
    } catch (err) {
      setError(`Grad-CAM failed: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }, [file, gradcamUrl])

  // ── Voice intents routed to this view ──────────────────
  // A spoken command is an event, so it is handled in a callback the shell
  // invokes, not by watching a prop from inside an effect. The shell keeps the
  // latest handler in a ref, so this re-registers freely as state changes.
  const handleVoice = useCallback(
    ({ intent, transcript }) => {
      if (intent === 'identify') {
        if (file) identify(file)
        else speech.speak('Drop in a photo first and I will identify it.')
      } else if (intent === 'save') {
        save()
      } else if (intent === 'fieldMarks') {
        const marks = result?.info?.field_marks
        speech.speak(
          marks?.length
            ? `Look for these field marks on the ${result.species.display_name}. ${marks.join(' ')}`
            : 'I have no field marks for this bird.',
        )
      } else if (intent === 'narrate') {
        if (!result) {
          speech.speak('Identify a bird first, then I can tell you about it.')
        } else {
          api
            .narrate(result)
            .then((payload) => speech.speak(payload.text))
            .catch(() => speech.speak(`This looks like a ${result.species.display_name}.`))
        }
      } else if (intent === 'ask') {
        setChatOpen(true)
        setPendingQuestion(transcript)
      }
    },
    [file, identify, save, result, speech],
  )

  useEffect(() => registerVoiceHandler?.(handleVoice), [registerVoiceHandler, handleVoice])

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="min-w-0 space-y-6">
        <Dropzone
          onFile={handleFile}
          preview={preview}
          onClear={clear}
          busy={busy}
          gradcamUrl={gradcamUrl}
          showGradcam={showGradcam}
          onToggleGradcam={toggleGradcam}
        />

        {error && (
          <div className="rounded-xl border border-(--color-low)/40 bg-(--color-low)/10 px-4 py-3 text-sm text-(--color-low)">
            {error}
          </div>
        )}

        {busy && !result && <SpeciesCardSkeleton />}

        {result && (
          <SpeciesCard
            // Remounts on each new identification, resetting narration state.
            key={`${result.species.folder}-${result.timing_ms?.total}`}
            result={result}
            speech={speech}
            onSave={save}
            saved={saved}
            onAsk={() => setChatOpen(true)}
            onPickSpecies={(folder) => onGoto?.('guide', { folder })}
          />
        )}

        {!result && !busy && !error && (
          <div className="card">
            <EmptyState icon={Bird} title="No bird yet">
              Drop in a photo and you will get the species, how confident the model is, how to
              confirm it in the field, and what it is easily confused with. Hold the space bar to
              talk to it.
            </EmptyState>
          </div>
        )}
      </div>

      <aside className="min-w-0">
        {chatOpen || pendingQuestion ? (
          <ChatPanel
            folder={result?.species?.folder}
            speech={speech}
            pendingQuestion={pendingQuestion}
            onConsumePending={() => setPendingQuestion(null)}
            onClose={() => setChatOpen(false)}
          />
        ) : (
          <button
            onClick={() => setChatOpen(true)}
            className="card w-full cursor-pointer px-5 py-4 text-left transition-colors duration-200 hover:border-(--color-accent)/40"
          >
            <p className="text-sm font-medium text-(--color-ink)">Ask the field guide</p>
            <p className="mt-1 text-xs leading-relaxed text-(--color-ink-faint)">
              Follow-up questions answered from the species knowledge base — by typing, or by
              holding the space bar and speaking.
            </p>
          </button>
        )}
      </aside>
    </div>
  )
}
