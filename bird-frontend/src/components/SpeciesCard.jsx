import { useState } from 'react'
import { motion } from 'motion/react'
import {
  AlertTriangle, Check, Eye, Info, Leaf, MessageCircleQuestion,
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
 * Both models' answers, side by side, when the verifier was consulted.
 *
 * Shown whenever verification ran, including when the two agree — agreement is the
 * most reassuring thing the app can tell you, and hiding it would waste the signal.
 */
function VerificationPanel({ verification }) {
  if (!verification) return null

  if (!verification.ran) {
    return (
      <div className="border-b border-(--color-line) bg-(--color-raised) px-6 py-3 text-xs text-(--color-ink-faint)">
        A second opinion was wanted here ({verification.reason}) but the
        open-vocabulary verifier is unavailable
        {verification.error ? `: ${verification.error}` : ''}. Install it with{' '}
        <code className="rounded bg-(--color-raised) px-1 py-0.5 font-mono">
          pip3 install -r requirements-verify.txt
        </code>
        .
      </div>
    )
  }

  const best = verification.best
  const agrees = verification.agrees_with_classifier

  return (
    <div className="border-b border-(--color-line) bg-(--color-accent)/5 px-6 py-4">
      <SectionTitle
        right={
          <span className="text-[0.68rem] text-(--color-ink-faint)">
            {verification.species_considered?.toLocaleString()} species considered
          </span>
        }
      >
        Second opinion
      </SectionTitle>

      <p className="mb-3 text-xs text-(--color-ink-faint)">
        Consulted because {verification.reason}. This model is not limited to the 200.
      </p>

      <div className="space-y-2">
        {verification.top.slice(0, 3).map((candidate, i) => (
          <div
            key={candidate.scientific_name}
            className={`flex items-baseline justify-between gap-3 rounded-lg border px-3 py-2 ${
              i === 0
                ? 'border-(--color-accent)/40 bg-(--color-accent)/10'
                : 'border-(--color-line)'
            }`}
          >
            <div className="min-w-0">
              <p className={`truncate text-sm ${i === 0 ? 'text-(--color-ink)' : 'text-(--color-ink-soft)'}`}>
                {candidate.common_name}
              </p>
              <p className="truncate text-[0.68rem] italic text-(--color-ink-faint)">
                {candidate.scientific_name}
              </p>
            </div>
            <span className="shrink-0 font-mono text-xs tabular-nums text-(--color-ink-faint)">
              {candidate.similarity.toFixed(3)}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {verification.confident === false && (
          <Chip tone="low">not confident enough to name it</Chip>
        )}
        {verification.confident && !verification.in_cub_200 && (
          <Chip tone="accent">
            <Sparkles size={11} strokeWidth={2} />
            outside the trained 200
          </Chip>
        )}
        {agrees === true && <Chip tone="high">agrees with the classifier</Chip>}
        {agrees === false && (
          <Chip tone="moderate">
            disagrees — it says {best.common_name}
          </Chip>
        )}
      </div>
    </div>
  )
}

/** What happened to this identification in the deck. */
function DeckOutcome({ deck, onForceAdd, forcing }) {
  if (!deck) return null

  if (!deck.entry) {
    return (
      <div className="flex flex-wrap items-center gap-3 border-b border-(--color-line) bg-(--color-raised) px-6 py-3">
        <p className="min-w-0 flex-1 text-xs text-(--color-ink-faint)">
          <strong className="text-(--color-ink-soft)">Not added to your deck</strong> —{' '}
          {deck.reason}. A wrong card is worse than a missing one, so nothing was
          filed.
        </p>
        <button
          onClick={onForceAdd}
          disabled={forcing}
          className="shrink-0 cursor-pointer rounded-lg border border-(--color-line) px-3 py-1.5 text-xs text-(--color-ink-soft) transition-colors duration-200 hover:border-(--color-accent)/50 hover:text-(--color-ink) disabled:opacity-50"
        >
          {forcing ? 'Adding…' : 'Add anyway'}
        </button>
      </div>
    )
  }

  const entry = deck.entry
  const isNewSpecies = deck.created

  return (
    <motion.div
      // A first capture is the payoff of the whole deck, so it gets a moment: the
      // banner scales in and an amber ring pulses once outward. A repeat encounter
      // just appears — celebrating it every time would cheapen the first.
      initial={isNewSpecies ? { opacity: 0, scale: 0.97 } : false}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 260, damping: 22 }}
      className={`relative flex flex-wrap items-center gap-2 overflow-hidden border-b border-(--color-line) px-6 py-3 ${
        isNewSpecies ? 'bg-(--color-sun)/12' : 'bg-(--color-high)/10'
      }`}
    >
      {isNewSpecies && (
        <motion.span
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 rounded-[inherit] ring-2 ring-(--color-sun)"
          initial={{ opacity: 0.9, scale: 0.98 }}
          animate={{ opacity: 0, scale: 1.02 }}
          transition={{ duration: 1.1, ease: 'easeOut' }}
        />
      )}
      <Check
        size={15}
        strokeWidth={2.5}
        className={`shrink-0 ${isNewSpecies ? 'text-(--color-sun-ink)' : 'text-(--color-high)'}`}
      />
      <p className="min-w-0 flex-1 text-sm text-(--color-ink-soft)">
        {isNewSpecies ? (
          <>
            <strong className="text-(--color-sun-ink)">New species!</strong> {entry.display_name}{' '}
            added to your deck
            {entry.source === 'external' && ' — beyond the trained 200'}.
          </>
        ) : (
          <>
            Already in your deck — <strong className="text-(--color-ink)">{entry.display_name}</strong>{' '}
            seen {entry.encounters} times now.
          </>
        )}
      </p>
    </motion.div>
  )
}

/**
 * Everything known about one identification.
 *
 * Ordered by what a birder actually needs: is it right (gauge + band), what is it
 * (name + taxonomy), what the deck did with it, what the second model thought, how
 * to confirm it (field marks), what else it could be (look-alikes), then the extras.
 * The provenance footer is last but never omitted — the checkpoint is unaudited and
 * the prose is machine-written, and the card says so.
 *
 * Callers pass a `key` that changes per identification, so a new result remounts the
 * card and the narration state resets on its own. That is cheaper and less
 * error-prone than an effect that clears state when a prop changes.
 */
export function SpeciesCard({ result, speech, onAsk, onPickSpecies, onForceAdd, forcing }) {
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
      <div className="flex flex-wrap gap-2 border-y border-(--color-line) bg-(--color-raised) px-6 py-3">
        <button
          onClick={speakNarration}
          disabled={!speech.supported || speech.muted || narrating}
          className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-(--color-accent) px-3.5 py-2 text-sm font-medium text-(--color-on-accent) transition-all duration-200 hover:bg-(--color-accent-hover) disabled:cursor-not-allowed disabled:opacity-40"
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

      </div>

      {/* Registration is automatic, so the card reports what happened rather than
          offering a button. */}
      <DeckOutcome deck={result.deck} onForceAdd={onForceAdd} forcing={forcing} />
      <VerificationPanel verification={result.verification} />

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
                  className="flex gap-2.5 rounded-lg border border-(--color-line) bg-(--color-raised) p-3 text-sm leading-relaxed text-(--color-ink-soft)"
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
                  className="rounded-xl border border-(--color-line) bg-(--color-raised) p-3.5"
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
    <footer className="space-y-1.5 border-t border-(--color-line) bg-(--color-raised) px-6 py-4 text-[0.72rem] leading-relaxed text-(--color-ink-faint)">
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
