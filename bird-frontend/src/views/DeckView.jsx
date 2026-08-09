import { useCallback, useEffect, useMemo, useState } from 'react'
import { motion } from 'motion/react'
import { Bird, Feather, Lock, Mic, Sparkles, Trash2, X } from 'lucide-react'

import { Bar, Chip, EmptyState, ProgressRing, SectionTitle, Taxonomy } from '../components/primitives'
import * as api from '../lib/api'

/**
 * The deck — one card per species you have found.
 *
 * Two sides, because they are two different achievements. **The 200** is the
 * trained classifier's world, so it has a fixed size and can be shown as a
 * completable grid with locked slots. **New birds** are species the open-vocabulary
 * verifier named from outside those 200, so that side has no ceiling — every card
 * there is a bird the CUB model could never have identified.
 *
 * Rejected uploads never appear here. If neither model was confident, nothing was
 * filed; the deck is meant to be a record, and a wrong card is worse than a gap.
 */
export function DeckView({ speech, onGoto }) {
  const [deck, setDeck] = useState(null)
  const [catalogue, setCatalogue] = useState([])
  const [showAll, setShowAll] = useState(false)
  const [selected, setSelected] = useState(null)
  const [reloadKey, setReloadKey] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    Promise.all([api.deck(), api.speciesList()])
      .then(([d, s]) => {
        if (cancelled) return
        setDeck(d)
        setCatalogue(s.species ?? [])
      })
      .catch(() => !cancelled && setDeck({ cub: [], external: [], stats: {} }))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [reloadKey])

  const remove = useCallback(async (key) => {
    await api.deleteCard(key)
    setSelected(null)
    setReloadKey((k) => k + 1)
  }, [])

  const found = useMemo(
    () => new Map((deck?.cub ?? []).map((entry) => [entry.key, entry])),
    [deck],
  )

  // In all-200 mode, every class gets a slot: a real card if found, a locked one if
  // not. The catalogue endpoint already returns all 200 summaries.
  const cubSlots = useMemo(() => {
    if (!showAll) return deck?.cub ?? []
    return catalogue.map((species) => found.get(species.folder) ?? {
      key: species.folder,
      locked: true,
      display_name: species.display_name,
      scientific_name: species.scientific_name,
      family: species.family,
    })
  }, [showAll, catalogue, found, deck])

  if (loading) {
    return <div className="card p-10 text-sm text-(--color-ink-faint)">Opening your deck…</div>
  }

  const stats = deck?.stats ?? {}
  const isEmpty = !deck?.cub?.length && !deck?.external?.length

  return (
    <div className="space-y-6">
      <DeckHeader stats={stats} showAll={showAll} onToggle={() => setShowAll((v) => !v)} />

      {isEmpty && !showAll ? (
        <div className="card">
          <EmptyState icon={Bird} title="Your deck is empty">
            Identify a bird and it files itself here automatically — no button to
            press. Birds among the trained 200 fill the first section; anything else
            the verifier can name goes to New birds.
          </EmptyState>
        </div>
      ) : (
        <>
          <section>
            <SectionTitle
              right={
                <span className="font-mono text-xs text-(--color-ink-faint)">
                  {stats.found ?? 0}/{stats.total_species ?? 200}
                </span>
              }
            >
              The 200 — species the trained model knows
            </SectionTitle>
            {cubSlots.length === 0 ? (
              <p className="px-1 text-sm text-(--color-ink-faint)">
                None found yet. Turn on “Show all 200” to see what is out there.
              </p>
            ) : (
              <CardGrid cards={cubSlots} onSelect={setSelected} />
            )}
          </section>

          <section>
            <SectionTitle
              right={
                <span className="font-mono text-xs text-(--color-ink-faint)">
                  {deck?.external?.length ?? 0}
                </span>
              }
            >
              New birds — beyond the trained 200
            </SectionTitle>
            {deck?.external?.length ? (
              <CardGrid cards={deck.external} onSelect={setSelected} />
            ) : (
              <p className="px-1 text-sm leading-relaxed text-(--color-ink-faint)">
                Nothing yet. Upload a bird that is not one of the 200 — a macaw, a
                penguin, an owl — and the open-vocabulary verifier will name it and
                start this side of the deck.
              </p>
            )}
          </section>

          {Object.keys(stats.by_family ?? {}).length > 1 && (
            <section className="card p-6">
              <SectionTitle>Families collected</SectionTitle>
              <FamilyBars byFamily={stats.by_family} />
            </section>
          )}
        </>
      )}

      {selected && (
        <CardDetail
          card={selected}
          speech={speech}
          onClose={() => setSelected(null)}
          onDelete={remove}
          onOpenGuide={onGoto}
        />
      )}
    </div>
  )
}

function DeckHeader({ stats, showAll, onToggle }) {
  return (
    <div className="card flex flex-wrap items-center gap-6 p-6">
      <ProgressRing
        value={stats.progress ?? 0}
        label={`${stats.found ?? 0}`}
        sublabel={`of ${stats.total_species ?? 200}`}
      />
      <div className="min-w-40 flex-1">
        <h2 className="font-display text-2xl text-(--color-ink)">
          {stats.found ?? 0} of {stats.total_species ?? 200} collected
        </h2>
        <p className="mt-1 text-sm text-(--color-ink-soft)">
          {stats.total_encounters ?? 0} encounter
          {stats.total_encounters === 1 ? '' : 's'} logged
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {stats.new_birds > 0 && (
            <Chip tone="accent">
              <Sparkles size={11} strokeWidth={2} />
              {stats.new_birds} beyond the 200
            </Chip>
          )}
          {stats.most_seen && (
            <Chip title="The bird you have photographed most">
              most seen: {stats.most_seen.display_name} ×{stats.most_seen.encounters}
            </Chip>
          )}
        </div>
      </div>
      <button
        onClick={onToggle}
        aria-pressed={showAll}
        className={`shrink-0 cursor-pointer rounded-lg border px-3.5 py-2 text-sm transition-colors duration-200 ${
          showAll
            ? 'border-(--color-accent) bg-(--color-accent)/15 text-(--color-accent-bright)'
            : 'border-(--color-line) text-(--color-ink-soft) hover:border-(--color-accent)/50 hover:text-(--color-ink)'
        }`}
      >
        {showAll ? 'Showing all 200' : 'Show all 200'}
      </button>
    </div>
  )
}

function CardGrid({ cards, onSelect }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
      {cards.map((card, i) => (
        <Card key={card.key} card={card} index={i} onSelect={onSelect} />
      ))}
    </div>
  )
}

function Card({ card, index, onSelect }) {
  if (card.locked) {
    return (
      <div
        className="card flex flex-col overflow-hidden opacity-45"
        title={`${card.display_name} — not found yet`}
      >
        <div className="grid aspect-square place-items-center bg-(--color-void)/40 text-(--color-ink-faint)">
          <Lock size={18} strokeWidth={1.5} />
        </div>
        <div className="px-2.5 py-2">
          <p className="truncate text-xs text-(--color-ink-faint)">{card.display_name}</p>
          <p className="mt-0.5 text-[0.62rem] tracking-wider text-(--color-ink-faint) uppercase">
            not found
          </p>
        </div>
      </div>
    )
  }

  const isNew = card.source === 'external'
  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.28, delay: Math.min(index * 0.02, 0.4) }}
      onClick={() => onSelect(card)}
      className="card group flex cursor-pointer flex-col overflow-hidden text-left transition-colors duration-200 hover:border-(--color-accent)/50"
    >
      <div className="relative aspect-square overflow-hidden bg-(--color-void)/40">
        {card.thumb ? (
          <img
            src={api.cardThumbUrl(card.key)}
            alt={card.display_name}
            loading="lazy"
            className="size-full object-cover transition-transform duration-300 group-hover:scale-105"
          />
        ) : (
          <span className="grid size-full place-items-center text-(--color-ink-faint)">
            <Feather size={20} strokeWidth={1.5} />
          </span>
        )}
        {card.encounters > 1 && (
          <span className="absolute top-1.5 right-1.5 rounded-full bg-(--color-void)/85 px-1.5 py-0.5 font-mono text-[0.62rem] text-(--color-ink-soft) backdrop-blur">
            ×{card.encounters}
          </span>
        )}
        {isNew && (
          <span className="absolute bottom-1.5 left-1.5 rounded-full bg-(--color-accent)/85 px-1.5 py-0.5 text-[0.62rem] font-medium text-white backdrop-blur">
            new
          </span>
        )}
      </div>
      <div className="px-2.5 py-2">
        <p className="font-display truncate text-[0.9rem] leading-tight text-(--color-ink)">
          {card.display_name}
        </p>
        {card.scientific_name && (
          <p className="truncate text-[0.66rem] italic text-(--color-ink-faint)">
            {card.scientific_name}
          </p>
        )}
      </div>
    </motion.button>
  )
}

function FamilyBars({ byFamily }) {
  const entries = Object.entries(byFamily).slice(0, 8)
  const max = Math.max(1, ...entries.map(([, n]) => n))
  return (
    <div className="space-y-3">
      {entries.map(([family, count], i) => (
        <Bar key={family} index={i} label={family} value={count / max} sublabel={`${count}`} />
      ))}
    </div>
  )
}

function CardDetail({ card, speech, onClose, onDelete, onOpenGuide }) {
  const isCub = card.source === 'cub'
  return (
    <div
      className="fixed inset-0 z-40 flex items-end justify-center bg-(--color-void)/70 p-4 backdrop-blur-sm sm:items-center"
      onClick={onClose}
      role="presentation"
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        onClick={(e) => e.stopPropagation()}
        className="card w-full max-w-md overflow-hidden"
        role="dialog"
        aria-label={card.display_name}
      >
        {card.thumb && (
          <img
            src={api.cardThumbUrl(card.key)}
            alt={card.display_name}
            className="max-h-64 w-full object-cover"
          />
        )}
        <div className="p-5">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <Taxonomy order={card.order} family={card.family} name={card.display_name} />
              <h3 className="font-display mt-1.5 text-2xl leading-tight text-(--color-ink)">
                {card.display_name}
              </h3>
              {card.scientific_name && (
                <p className="text-sm italic text-(--color-ink-faint)">{card.scientific_name}</p>
              )}
            </div>
            <button
              onClick={onClose}
              aria-label="Close"
              className="cursor-pointer rounded p-1 text-(--color-ink-faint) transition-colors hover:text-(--color-ink)"
            >
              <X size={16} strokeWidth={2} />
            </button>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            <Chip tone={isCub ? 'neutral' : 'accent'}>
              {isCub ? 'one of the 200' : 'beyond the 200'}
            </Chip>
            <Chip>seen ×{card.encounters}</Chip>
            <Chip title={card.identified_by === 'verifier'
              ? 'Named by the open-vocabulary verifier'
              : card.identified_by === 'manual'
                ? 'You added this by hand'
                : 'Named by the trained classifier'}>
              {card.identified_by === 'verifier' ? <Sparkles size={11} strokeWidth={2} /> : null}
              {card.identified_by === 'cub' ? 'classifier' : card.identified_by}
            </Chip>
            {card.agreed === true && <Chip tone="high">both models agreed</Chip>}
            {card.agreed === false && <Chip tone="moderate">models disagreed</Chip>}
          </div>

          <dl className="mt-4 space-y-1.5 text-xs text-(--color-ink-faint)">
            <div className="flex justify-between gap-4">
              <dt>First found</dt>
              <dd className="font-mono">{new Date(card.first_seen).toLocaleString()}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt>Last seen</dt>
              <dd className="font-mono">{new Date(card.last_seen).toLocaleString()}</dd>
            </div>
            {card.best_confidence != null && (
              <div className="flex justify-between gap-4">
                <dt>{card.identified_by === 'verifier' ? 'Best similarity' : 'Best confidence'}</dt>
                <dd className="font-mono">
                  {card.identified_by === 'verifier'
                    ? card.best_confidence.toFixed(3)
                    : `${Math.round(card.best_confidence * 100)}%`}
                </dd>
              </div>
            )}
          </dl>

          <div className="mt-5 flex flex-wrap gap-2">
            {isCub && (
              <button
                onClick={() => {
                  onClose()
                  onOpenGuide?.('guide', { folder: card.key })
                }}
                className="cursor-pointer rounded-lg bg-(--color-accent) px-3.5 py-2 text-sm font-medium text-white transition-colors duration-200 hover:bg-(--color-accent-bright)"
              >
                Open in field guide
              </button>
            )}
            {speech?.supported && (
              <button
                onClick={() =>
                  speech.speak(
                    `${card.display_name}. ${card.scientific_name ?? ''}. ` +
                      `You have seen this bird ${card.encounters} time${card.encounters === 1 ? '' : 's'}.`,
                  )
                }
                className="cursor-pointer rounded-lg border border-(--color-line) px-3.5 py-2 text-sm text-(--color-ink-soft) transition-colors duration-200 hover:border-(--color-accent)/50 hover:text-(--color-ink)"
              >
                Speak
              </button>
            )}
            <button
              onClick={() => onDelete(card.key)}
              className="ml-auto inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-(--color-line) px-3 py-2 text-sm text-(--color-ink-faint) transition-colors duration-200 hover:border-(--color-reject)/50 hover:text-(--color-reject)"
            >
              <Trash2 size={14} strokeWidth={2} />
              Remove
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

/** Encounter history, kept below the deck — the event log the deck is derived from. */
export function RecentEncounters() {
  const [data, setData] = useState(null)

  useEffect(() => {
    let cancelled = false
    api
      .sightings(20)
      .then((d) => !cancelled && setData(d))
      .catch(() => !cancelled && setData({ sightings: [] }))
    return () => {
      cancelled = true
    }
  }, [])

  if (!data?.sightings?.length) return null

  return (
    <section className="card overflow-hidden">
      <header className="flex items-center justify-between border-b border-(--color-line) px-5 py-3.5">
        <h3 className="text-sm font-semibold text-(--color-ink)">Recent encounters</h3>
        <span className="font-mono text-xs text-(--color-ink-faint)">{data.total}</span>
      </header>
      <ul>
        {data.sightings.map((s) => (
          <li
            key={s.id}
            className="flex items-center gap-3 border-b border-(--color-line)/60 px-5 py-2.5 last:border-0"
          >
            <span className="text-(--color-ink-faint)">
              {s.kind === 'audio' ? <Mic size={14} strokeWidth={2} /> : <Feather size={14} strokeWidth={2} />}
            </span>
            <span className="min-w-0 flex-1 truncate text-sm text-(--color-ink-soft)">
              {s.display_name}
            </span>
            {!s.is_bird && <Chip tone="low">rejected</Chip>}
            <span className="shrink-0 font-mono text-[0.7rem] text-(--color-ink-faint)">
              {new Date(s.ts).toLocaleDateString()}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}
