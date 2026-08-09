import { useState } from 'react'
import { motion } from 'motion/react'
import {
  AlertTriangle, Bookmark, Check, Eye, Info, Leaf, MessageCircleQuestion,
  Ruler, Sparkles, Square, Utensils, Volume2, Wind,
} from 'lucide-react'

import { Bar, Chip, ConfidenceGauge, SectionTitle, Spinner, Taxonomy } from './primitives'
import { bandOf } from '../lib/confidence'
import { sentencesWithActive } from '../hooks/useSpeech'
import * as api from '../lib/api'

const IUCN_TONE = {
  'Least Concern': 'high',
  'Near Threatened': 'moderate',
  Vulnerable: 'moderate',
  Endangered: 'low',
  'Critically Endangered': 'low',
}

/**
 * Everything known about one identification.
 *
 * Ordered by what a birder actually needs: is it right (gauge + band), what is
 * it (name + taxonomy), how do I confirm it (field marks), what else could it be
 * (look-alikes), then the extras. The provenance footer is last but never
 * omitted — the checkpoint is unaudited and the prose is machine-written, and
 * the card says so.
 *
 * Callers pass a `key` that changes per identification, so a new result remounts
 * the card and the narration state resets on its own. That is cheaper and less
 * error-prone than an effect that clears state when a prop changes.
 */
export function SpeciesCard({ result, speech, onSave, saved, onAsk, onPickSpecies }) {
  const [narration, setNarration] = useState(null)
  const [narrating, setNarrating] = useState(false)

  const species = result.species
  const info = result.info || {}
  const gate = result.openset || {}
  const rejected = gate.enabled && !gate.is_bird
  const band = bandOf(result.confidence_band)

  async function speakNarration() {
    if (speech.speaking) {
      speech.cancel()
      return
    }
    if (narration) {
      speech.speak(narration.text)
      return
    }
    setNarrating(true)
    try {
      const payload = await api.narrate(result)
      setNarration(payload)
      speech.speak(payload.text)
    } catch (err) {
      const fallback = { text: `This looks like a ${species.display_name}.`, generated: false, reason: err.message }
      setNarration(fallback)
      speech.speak(fallback.text)
    } finally {
      setNarrating(false)
    }
  }

  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="card overflow-hidden"
    >
      {rejected && <RejectionBanner gate={gate} />}

      {/* ── Identity ── */}
      <header className="flex flex-wrap items-start justify-between gap-6 p-6 pb-5">
        <div className="min-w-0 flex-1">
          <Taxonomy order={species.order} family={species.family} name={species.display_name} />
          <h2 className="font-display mt-2 text-3xl leading-tight text-(--color-ink)">
            {species.display_name}
          </h2>
          {species.scientific_name && (
            <p className="mt-1 text-sm italic text-(--color-ink-faint)">{species.scientific_name}</p>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            <Chip tone={rejected ? 'low' : result.confidence_band}>{band.label}</Chip>
            {info.conservation_status && (
              <Chip tone={IUCN_TONE[info.conservation_status] ?? 'neutral'} title="IUCN Red List category">
                <Leaf size={12} strokeWidth={2} />
                {info.conservation_status}
              </Chip>
            )}
            {info.size_cm && (
              <Chip>
                <Ruler size={12} strokeWidth={2} />
                {info.size_cm}
              </Chip>
            )}
            {result.timing_ms?.total != null && (
              <Chip title="Time spent in the model, end to end">{result.timing_ms.total} ms</Chip>
            )}
          </div>
          <p className="mt-3 text-xs text-(--color-ink-faint)">{band.hint}</p>
        </div>
        <ConfidenceGauge value={result.confidence} band={rejected ? 'low' : result.confidence_band} />
      </header>

      {/* ── Actions ── */}
      <div className="flex flex-wrap gap-2 border-y border-(--color-line) bg-(--color-void)/40 px-6 py-3">
        <button
          onClick={speakNarration}
          disabled={!speech.supported || speech.muted || narrating}
          className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-(--color-accent) px-3.5 py-2 text-sm font-medium text-white transition-all duration-200 hover:bg-(--color-accent-bright) disabled:cursor-not-allowed disabled:opacity-40"
          title={
            !speech.supported ? 'This browser has no speech synthesis'
              : speech.muted ? 'Voice is muted — unmute in the header'
                : 'Read this identification aloud'
          }
        >
          {speech.speaking ? <Square size={14} strokeWidth={2.5} /> : <Volume2 size={15} strokeWidth={2} />}
          {speech.speaking ? 'Stop' : narrating ? 'Writing…' : 'Speak'}
        </button>

        <button
          onClick={() => onAsk?.(species.folder)}
          className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-(--color-line) px-3.5 py-2 text-sm text-(--color-ink-soft) transition-colors duration-200 hover:border-(--color-accent)/50 hover:text-(--color-ink)"
        >
          <MessageCircleQuestion size={15} strokeWidth={2} />
          Ask about it
        </button>

        <button
          onClick={onSave}
          disabled={saved}
          className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-(--color-line) px-3.5 py-2 text-sm text-(--color-ink-soft) transition-colors duration-200 hover:border-(--color-accent)/50 hover:text-(--color-ink) disabled:cursor-default disabled:opacity-60"
        >
          {saved ? <Check size={15} strokeWidth={2.5} className="text-(--color-high)" /> : <Bookmark size={15} strokeWidth={2} />}
          {saved ? 'In your life list' : 'Save to life list'}
        </button>
      </div>

      {narration && (
        <NarrationPanel narration={narration} speech={speech} />
      )}

      <div className="grid gap-6 p-6 lg:grid-cols-2">
        {/* ── Field marks ── */}
        {info.field_marks?.length > 0 && (
          <section className="lg:col-span-2">
            <SectionTitle>How to confirm it</SectionTitle>
            <ul className="grid gap-2 sm:grid-cols-2">
              {info.field_marks.map((mark, i) => (
                <motion.li
                  key={mark}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 * i, duration: 0.3 }}
                  className="flex gap-2.5 rounded-lg border border-(--color-line) bg-(--color-void)/30 p-3 text-sm leading-relaxed text-(--color-ink-soft)"
                >
                  <Eye size={15} strokeWidth={2} className="mt-0.5 shrink-0 text-(--color-accent)" />
                  {mark}
                </motion.li>
              ))}
            </ul>
          </section>
        )}

        {/* ── Facts ── */}
        <section className="space-y-3">
          <SectionTitle>Where and how it lives</SectionTitle>
          <Fact icon={Leaf} label="Habitat" value={info.habitat} />
          <Fact icon={Wind} label="Migration" value={info.migration} />
          <Fact icon={Utensils} label="Diet" value={info.diet} />
          {info.range_description && <Fact icon={Info} label="Range" value={info.range_description} />}
        </section>

        {/* ── Ranking ── */}
        <section>
          <SectionTitle right={<span className="text-[0.68rem] text-(--color-ink-faint)">click to open</span>}>
            Other candidates
          </SectionTitle>
          <div className="space-y-3">
            {result.top5?.map((row, i) => (
              <Bar
                key={row.folder}
                index={i}
                label={row.display_name}
                value={row.confidence}
                highlight={i === 0}
                color={i === 0 ? band.color : 'var(--color-accent-dim)'}
                onClick={() => onPickSpecies?.(row.folder)}
              />
            ))}
          </div>
        </section>

        {/* ── Look-alikes ── */}
        {info.similar_species?.length > 0 && (
          <section className="lg:col-span-2">
            <SectionTitle>Easily confused with</SectionTitle>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {info.similar_species.map((similar) => (
                <div
                  key={similar.name}
                  className="rounded-xl border border-(--color-line) bg-(--color-void)/30 p-3.5"
                >
                  <div className="mb-1.5 flex items-center justify-between gap-2">
                    <p className="truncate text-sm font-medium text-(--color-ink)">{similar.name}</p>
                    {similar.source === 'curated' && (
                      <Chip tone="accent" title="Hand-written, not machine-generated">verified</Chip>
                    )}
                  </div>
                  <p className="text-xs leading-relaxed text-(--color-ink-faint)">
                    {similar.how_to_distinguish}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* ── Fun fact ── */}
        {info.fun_fact && (
          <section className="lg:col-span-2">
            <div className="flex gap-3 rounded-xl border border-(--color-accent)/20 bg-(--color-accent)/5 p-4">
              <Sparkles size={16} strokeWidth={2} className="mt-0.5 shrink-0 text-(--color-accent)" />
              <p className="text-sm leading-relaxed text-(--color-ink-soft)">{info.fun_fact}</p>
            </div>
          </section>
        )}
      </div>

      <Provenance provenance={result.provenance} usedLlm={info.used_llm} gate={gate} />
    </motion.article>
  )
}

function Fact({ icon: Icon, label, value }) {
  if (!value) return null
  return (
    <div className="flex gap-3">
      <Icon size={15} strokeWidth={2} className="mt-0.5 shrink-0 text-(--color-ink-faint)" />
      <div className="min-w-0">
        <div className="text-[0.68rem] tracking-wider text-(--color-ink-faint) uppercase">{label}</div>
        <p className="mt-0.5 text-sm leading-relaxed text-(--color-ink-soft)">{value}</p>
      </div>
    </div>
  )
}

/**
 * The narration, with the sentence being spoken highlighted as it is read.
 */
function NarrationPanel({ narration, speech }) {
  const isCurrent = speech.spokenText === narration.text
  const sentences = sentencesWithActive(narration.text, isCurrent ? speech.spokenUpto : -1)

  return (
    <div className="border-b border-(--color-line) bg-(--color-accent)/5 px-6 py-4">
      <SectionTitle
        right={
          <span className="text-[0.68rem] text-(--color-ink-faint)">
            {narration.generated ? `written by ${narration.model ?? 'local model'}` : 'from the knowledge base'}
          </span>
        }
      >
        Spoken summary
      </SectionTitle>
      <p className="text-[0.95rem] leading-relaxed text-(--color-ink-soft)">
        {sentences.map((sentence, i) => (
          <span key={i} className={sentence.active && speech.speaking ? 'speaking' : undefined}>
            {sentence.text}
          </span>
        ))}
      </p>
      {!narration.generated && narration.reason && (
        <p className="mt-2 text-xs text-(--color-ink-faint)">
          Local model unavailable ({narration.reason}) — this is template text.
        </p>
      )}
    </div>
  )
}

function RejectionBanner({ gate }) {
  return (
    <div className="flex gap-3 border-b border-(--color-reject)/30 bg-(--color-reject)/10 px-6 py-4">
      <AlertTriangle size={18} strokeWidth={2} className="mt-0.5 shrink-0 text-(--color-reject)" />
      <div>
        <p className="text-sm font-semibold text-(--color-reject)">
          This does not look like one of the 200 species I know
        </p>
        <p className="mt-1 text-xs leading-relaxed text-(--color-ink-soft)">
          The open-set gate scored it {gate.score?.toFixed(2)} against a threshold of{' '}
          {gate.threshold?.toFixed(2)}. Everything below is what the model would say if forced to
          pick a species — not an identification. Try a clearer photo of a single bird.
        </p>
      </div>
    </div>
  )
}

function Provenance({ provenance = {}, usedLlm, gate }) {
  return (
    <footer className="space-y-1.5 border-t border-(--color-line) bg-(--color-void)/50 px-6 py-4 text-[0.72rem] leading-relaxed text-(--color-ink-faint)">
      {provenance.caveat && (
        <p className="flex gap-2">
          <AlertTriangle size={13} strokeWidth={2} className="mt-0.5 shrink-0 text-(--color-moderate)" />
          <span>
            <strong className="text-(--color-moderate)">Unaudited checkpoint.</strong>{' '}
            {provenance.caveat}
          </span>
        </p>
      )}
      {usedLlm && provenance.kb_source && (
        <p>
          Species notes were generated locally by {provenance.kb_source} and are not
          expert-verified. Look-alike entries marked “verified” are hand-written.
        </p>
      )}
      <p>
        {provenance.checkpoint} · {provenance.num_species} species ·{' '}
        {gate.enabled ? `open-set gate on (${gate.method})` : 'open-set gate off'}
      </p>
    </footer>
  )
}

export function SpeciesCardSkeleton({ label = 'Identifying…' }) {
  return (
    <div className="card flex items-center justify-center p-16">
      <Spinner label={label} />
    </div>
  )
}
