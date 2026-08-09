import { useCallback, useEffect, useRef, useState } from 'react'
import { AudioLines, Circle, Mic, Play, Square, Upload } from 'lucide-react'

import { Bar, Chip, ConfidenceGauge, EmptyState, SectionTitle, Spinner, Taxonomy } from '../components/primitives'
import { bandOf } from '../lib/confidence'
import * as api from '../lib/api'

/**
 * Identify a bird from its call, via BirdNET.
 *
 * BirdNET covers ~6,500 species against the photo model's 200, so this view can
 * name birds the Identify view cannot — and says which, because "not in the
 * image model" is genuinely useful information rather than a failure.
 *
 * The bundled sample clips exist so the view is never a dead end: an audio
 * feature you cannot try without first finding a recording is a feature nobody
 * tries.
 */
export function ListenView({ speech }) {
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [samples, setSamples] = useState([])
  const [minConf, setMinConf] = useState(0.25)
  const [recording, setRecording] = useState(false)
  const [sourceLabel, setSourceLabel] = useState(null)

  const recorderRef = useRef(null)
  const audioRef = useRef(null)
  const inputRef = useRef(null)
  const [audioUrl, setAudioUrl] = useState(null)

  useEffect(() => {
    api.samples().then((d) => setSamples(d.samples ?? [])).catch(() => setSamples([]))
  }, [])

  const analyse = useCallback(
    async (file, label) => {
      setBusy(true)
      setError(null)
      setResult(null)
      setSourceLabel(label)
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      setAudioUrl(URL.createObjectURL(file))
      try {
        const payload = await api.identifyAudio(file, { minConf })
        if (!payload.ok) setError(payload.error)
        else setResult(payload)
      } catch (err) {
        setError(err.message)
      } finally {
        setBusy(false)
      }
    },
    [minConf, audioUrl],
  )

  const playSample = useCallback(
    async (sample) => {
      setBusy(true)
      setError(null)
      setResult(null)
      setSourceLabel(sample.common_name)
      try {
        const res = await fetch(sample.url)
        const blob = await res.blob()
        const file = new File([blob], sample.file.split('/').pop(), { type: blob.type })
        await analyse(file, sample.common_name)
      } catch (err) {
        setError(`Could not load that sample: ${err.message}`)
        setBusy(false)
      }
    },
    [analyse],
  )

  const toggleRecording = useCallback(async () => {
    if (recording) {
      recorderRef.current?.stop()
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      const chunks = []
      recorder.ondataavailable = (e) => chunks.push(e.data)
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        setRecording(false)
        const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
        // librosa/soundfile decode webm/opus fine; no ffmpeg needed.
        analyse(new File([blob], 'recording.webm', { type: blob.type }), 'your recording')
      }
      recorderRef.current = recorder
      recorder.start()
      setRecording(true)
    } catch (err) {
      setError(`Microphone unavailable: ${err.message}`)
    }
  }, [recording, analyse])

  const speakResult = () => {
    if (!result?.detected) return
    const sp = result.species
    speech.speak(
      `That call is a ${sp.common_name}, ${Math.round(result.confidence * 100)} percent confident. ` +
        (sp.in_image_model
          ? 'It is also one of the species the photo model knows.'
          : 'It is not one of the two hundred species the photo model covers.'),
    )
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="min-w-0 space-y-6">
        {/* ── Input ── */}
        <section className="card p-5">
          <SectionTitle
            right={
              <span className="font-mono text-xs text-(--color-ink-faint)">
                threshold {Math.round(minConf * 100)}%
              </span>
            }
          >
            Record or upload a call
          </SectionTitle>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={toggleRecording}
              className={`inline-flex cursor-pointer items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors duration-200 ${
                recording
                  ? 'bg-(--color-reject) text-white'
                  : 'bg-(--color-accent) text-white hover:bg-(--color-accent-bright)'
              }`}
            >
              {recording ? <Square size={14} strokeWidth={2.5} /> : <Mic size={15} strokeWidth={2} />}
              {recording ? 'Stop recording' : 'Record'}
            </button>
            <button
              onClick={() => inputRef.current?.click()}
              className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-(--color-line) px-3.5 py-2 text-sm text-(--color-ink-soft) transition-colors duration-200 hover:border-(--color-accent)/50 hover:text-(--color-ink)"
            >
              <Upload size={15} strokeWidth={2} />
              Upload a file
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="audio/*,.ogg,.mp3,.wav,.flac,.m4a"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) analyse(file, file.name)
              }}
            />
          </div>

          <label className="mt-4 block">
            <span className="text-[0.68rem] tracking-wider text-(--color-ink-faint) uppercase">
              Confidence threshold
            </span>
            <input
              type="range"
              min={0.05}
              max={0.9}
              step={0.05}
              value={minConf}
              onChange={(e) => setMinConf(Number(e.target.value))}
              className="mt-2 w-full accent-(--color-accent)"
            />
            <span className="text-xs text-(--color-ink-faint)">
              Below about 15% you will mostly hear noise reported as birds.
            </span>
          </label>

          {audioUrl && (
            <audio ref={audioRef} src={audioUrl} controls className="mt-4 w-full" />
          )}
        </section>

        {busy && (
          <div className="card p-8">
            <Spinner label={`Listening to ${sourceLabel ?? 'the clip'}…`} />
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-(--color-low)/40 bg-(--color-low)/10 px-4 py-3 text-sm text-(--color-low)">
            {error}
          </div>
        )}

        {result && !result.detected && (
          <div className="card">
            <EmptyState icon={AudioLines} title="Nothing above the threshold">
              No bird call scored above {Math.round(result.settings.min_conf * 100)}% in this{' '}
              {result.settings.duration_sec}s clip. Try lowering the threshold, or use at least
              three seconds with little background noise.
            </EmptyState>
          </div>
        )}

        {result?.detected && <AudioResult result={result} onSpeak={speakResult} speech={speech} />}
      </div>

      {/* ── Sample clips ── */}
      <aside className="min-w-0">
        <section className="card p-5">
          <SectionTitle>Try a bundled recording</SectionTitle>
          <p className="mb-3 text-xs leading-relaxed text-(--color-ink-faint)">
            Creative-Commons clips from Wikimedia, shipped with the project.
          </p>
          <div className="space-y-1.5">
            {samples.length === 0 && (
              <p className="text-xs text-(--color-ink-faint)">No sample clips on disk.</p>
            )}
            {samples.map((sample) => (
              <button
                key={sample.file}
                onClick={() => playSample(sample)}
                disabled={busy}
                className="group flex w-full cursor-pointer items-center gap-2.5 rounded-lg border border-(--color-line) px-3 py-2 text-left transition-colors duration-200 hover:border-(--color-accent)/50 disabled:opacity-50"
              >
                <Play size={13} strokeWidth={2} className="shrink-0 text-(--color-accent)" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-(--color-ink-soft) group-hover:text-(--color-ink)">
                    {sample.common_name}
                  </span>
                  <span className="block truncate text-[0.68rem] italic text-(--color-ink-faint)">
                    {sample.scientific_name}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </section>
      </aside>
    </div>
  )
}

function AudioResult({ result, onSpeak, speech }) {
  const sp = result.species
  const info = result.info
  const band = bandOf(result.confidence_band)

  return (
    <article className="card overflow-hidden">
      <header className="flex flex-wrap items-start justify-between gap-6 p-6 pb-4">
        <div className="min-w-0 flex-1">
          {info && <Taxonomy order={info.order} family={info.family} name={sp.common_name} />}
          <h2 className="font-display mt-2 text-3xl leading-tight text-(--color-ink)">
            {sp.common_name}
          </h2>
          <p className="mt-1 text-sm italic text-(--color-ink-faint)">{sp.scientific_name}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Chip tone={result.confidence_band}>{band.label}</Chip>
            <Chip>{result.settings.duration_sec}s clip</Chip>
            <Chip>{result.detections.length} detections</Chip>
            {sp.in_image_model ? (
              <Chip tone="accent" title="The photo model has a class for this species">
                also in the photo model
              </Chip>
            ) : (
              <Chip title="BirdNET knows ~6,500 species; the photo model knows 200">
                beyond the photo model
              </Chip>
            )}
          </div>
        </div>
        <ConfidenceGauge value={result.confidence} band={result.confidence_band} />
      </header>

      <div className="border-y border-(--color-line) bg-(--color-void)/40 px-6 py-3">
        <button
          onClick={onSpeak}
          disabled={!speech.supported || speech.muted}
          className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-(--color-accent) px-3.5 py-2 text-sm font-medium text-white transition-colors duration-200 hover:bg-(--color-accent-bright) disabled:cursor-not-allowed disabled:opacity-40"
        >
          <AudioLines size={15} strokeWidth={2} />
          Speak result
        </button>
      </div>

      <div className="space-y-6 p-6">
        <section>
          <SectionTitle>Heard in this recording</SectionTitle>
          <div className="space-y-3">
            {result.grouped.map((row, i) => (
              <Bar
                key={row.common_name}
                index={i}
                label={row.common_name}
                value={row.confidence}
                sublabel={`${Math.round(row.confidence * 100)}% · ${row.count}× from ${row.first_heard}s`}
                highlight={i === 0}
                color={i === 0 ? band.color : 'var(--color-accent-dim)'}
              />
            ))}
          </div>
        </section>

        {result.spectrogram_url && (
          <section>
            <SectionTitle>Spectrogram and detection timeline</SectionTitle>
            <img
              src={result.spectrogram_url}
              alt="Mel spectrogram with a lane per detected species"
              className="w-full rounded-xl border border-(--color-line)"
            />
          </section>
        )}

        {info && (
          <section className="grid gap-3 sm:grid-cols-2">
            {info.habitat && <Detail label="Habitat" value={info.habitat} />}
            {info.migration && <Detail label="Migration" value={info.migration} />}
          </section>
        )}
      </div>

      <footer className="flex items-center gap-2 border-t border-(--color-line) bg-(--color-void)/50 px-6 py-3 text-[0.72rem] text-(--color-ink-faint)">
        <Circle size={7} strokeWidth={3} className="text-(--color-accent)" />
        Identified by BirdNET (Cornell Lab of Ornithology)
        {result.settings.use_location
          ? ` · location prior on (${result.settings.lat}, ${result.settings.lon})`
          : ' · location prior off, scoring against all species'}
      </footer>
    </article>
  )
}

function Detail({ label, value }) {
  return (
    <div>
      <div className="text-[0.68rem] tracking-wider text-(--color-ink-faint) uppercase">{label}</div>
      <p className="mt-0.5 text-sm leading-relaxed text-(--color-ink-soft)">{value}</p>
    </div>
  )
}
