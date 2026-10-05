import { useState } from 'react'
import { motion } from 'motion/react'
import {
  BadgeCheck, Check, Eye, Info, Leaf, MessageCircleQuestion, Ruler, SearchX,
  Sparkles, Square, Utensils, Volume2, Wind,
} from 'lucide-react'

import { Chip, ConfidenceGauge, SectionTitle, Spinner, Taxonomy } from './primitives'
import { bandOf } from '../lib/confidence'
import { answerOf } from '../lib/answer'
import { sentencesWithActive } from '../hooks/useSpeech'
import * as api from '../lib/api'

const IUCN_TONE = {
  'Least Concern': 'high',
  'Near Threatened': 'moderate',
  Vulnerable: 'moderate',
  Endangered: 'low',
  'Critically Endangered': 'low',
}

/** What happened to this identification in the deck. */
function DeckOutcome({ deck, onForceAdd, forcing }) {
  if (!deck) return null

  if (!deck.entry) {
    return (
      <div className="flex flex-wrap items-center gap-3 border-b border-(--color-line) bg-(--color-raised) px-6 py-3">
        <p className="min-w-0 flex-1 text-xs text-(--color-ink-faint)">
          <strong className="text-(--color-ink-soft)">Not added to your deck</strong> — only birds
          that can be named confidently are filed. A wrong card is worse than a missing one.
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
 * The answer to one identification, and what a birder needs to confirm it.
 *
 * Only the resolved answer is shown: the classifier when it is trusted, the
 * open-vocabulary verifier when it named the bird instead, or a plain "not
 * sure" when neither would commit. How that was decided stays behind the
 * scenes — the classifier's discarded guess, the verifier's scores and the
 * model details never reach the card.
 *
 * Callers pass a `key` that changes per identification, so a new result remounts the
 * card and the narration state resets on its own. That is cheaper and less
 * error-prone than an effect that clears state when a prop changes.
 */
export function SpeciesCard({ result, speech, onAsk, onForceAdd, forcing }) {
  const [narration, setNarration] = useState(null)
  const [narrating, setNarrating] = useState(false)

  const answer = answerOf(result)
  const info = answer.info || {}
  const unsure = answer.status !== 'identified'
  const byVerifier = answer.identified_by === 'verifier'
  const outside = byVerifier && answer.source === 'external'
  const band = bandOf(answer.band)
  const hasFacts = Boolean(info.habitat || info.migration || info.diet || info.range_description)

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
      const text = answer.display_name
        ? `This is a ${answer.display_name}.`
        : 'I am not sure which bird this is.'
      const fallback = { text, generated: false, reason: err.message }
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
      {/* ── Identity ── */}
      {unsure ? (
        <header className="flex items-start gap-4 p-6 pb-5">
          <span className="grid size-11 shrink-0 place-items-center rounded-full border border-(--color-line) text-(--color-ink-faint)">
            <SearchX size={20} strokeWidth={1.75} />
          </span>
          <div className="min-w-0">
            <h2 className="font-display text-3xl leading-tight text-(--color-ink)">
              Not sure about this one
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-(--color-ink-soft)">
              Neither model could name it confidently — try a closer, sharper photo of a single
              bird.
            </p>
            {answer.best_guess && (
              <p className="mt-3 text-xs text-(--color-ink-faint)">
                Best guess: <span className="text-(--color-ink-soft)">{answer.best_guess}</span> —
                low confidence
              </p>
            )}
          </div>
        </header>
      ) : (
        <header className="flex flex-wrap items-start justify-between gap-6 p-6 pb-5">
          <div className="min-w-0 flex-1">
            <Taxonomy order={answer.order} family={answer.family} name={answer.display_name} />
            <h2 className="font-display mt-2 text-3xl leading-tight text-(--color-ink)">
              {answer.display_name}
            </h2>
            {answer.scientific_name && (
              <p className="mt-1 text-sm italic text-(--color-ink-faint)">{answer.scientific_name}</p>
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              {byVerifier ? (
                <Chip tone="high" title="Named with confidence">
                  <BadgeCheck size={13} strokeWidth={2} />
                  Confirmed match
                </Chip>
              ) : (
                <Chip tone={answer.band}>{band.label}</Chip>
              )}
              {outside && (
                <Chip tone="accent" title="A species outside the 200 in the field guide">
                  <Sparkles size={11} strokeWidth={2} />
                  Beyond the 200
                </Chip>
              )}
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
            </div>
            {!byVerifier && <p className="mt-3 text-xs text-(--color-ink-faint)">{band.hint}</p>}
          </div>
          {!byVerifier && <ConfidenceGauge value={answer.confidence} band={answer.band} />}
        </header>
      )}

      {/* ── Actions ── */}
      {!unsure && (
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
            onClick={() => onAsk?.()}
            className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-(--color-line) px-3.5 py-2 text-sm text-(--color-ink-soft) transition-colors duration-200 hover:border-(--color-accent)/50 hover:text-(--color-ink)"
          >
            <MessageCircleQuestion size={15} strokeWidth={2} />
            Ask about it
          </button>
        </div>
      )}

      {/* Registration is automatic, so the card reports what happened rather than
          offering a button. */}
      <DeckOutcome deck={result.deck} onForceAdd={onForceAdd} forcing={forcing} />

      {narration && (
        <NarrationPanel narration={narration} speech={speech} />
      )}

      {!unsure && (
        <div className="grid gap-6 p-6 lg:grid-cols-2">
          {outside && (
            <p className="text-sm text-(--color-ink-faint) lg:col-span-2">
              Outside the 200-species field guide — no field notes for this bird yet.
            </p>
          )}

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
          {hasFacts && (
            <section className="space-y-3 lg:col-span-2">
              <SectionTitle>Where and how it lives</SectionTitle>
              <div className="grid gap-3 sm:grid-cols-2">
                <Fact icon={Leaf} label="Habitat" value={info.habitat} />
                <Fact icon={Wind} label="Migration" value={info.migration} />
                <Fact icon={Utensils} label="Diet" value={info.diet} />
                {info.range_description && <Fact icon={Info} label="Range" value={info.range_description} />}
              </div>
            </section>
          )}

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
      )}
    </motion.article>
  )
}

function Fact({ icon: Icon, label, value }) {
  if (!value) return null
  return (
    <div className="flex gap-3">
      <Icon size={15} strokeWidth={2} className="mt-0.5 shrink-0 text-(--color-ink-faint)" />
      <div className="min-w-0">
        <div className="text-caption text-(--color-ink-faint)">{label}</div>
        <p className="mt-0.5 text-sm leading-relaxed text-(--color-ink-soft)">{value}</p>
      </div>
    </div>
  )
}

/** Narrations that are template text by design, not because a model was down. */
const DELIBERATE_TEMPLATES = new Set([
  'outside the knowledge base',
  'open-set rejection',
  'no knowledge-base record',
])

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
            {narration.generated ? `written by ${narration.model ?? 'local model'}` : 'from the field guide'}
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
      {!narration.generated && narration.reason && !DELIBERATE_TEMPLATES.has(narration.reason) && (
        <p className="mt-2 text-xs text-(--color-ink-faint)">
          The language model is unavailable right now, so this is template text.
        </p>
      )}
    </div>
  )
}

export function SpeciesCardSkeleton({ label = 'Identifying…' }) {
  return (
    <div className="card flex items-center justify-center p-16">
      <Spinner label={label} />
    </div>
  )
}
