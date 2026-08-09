import { useCallback, useEffect, useState } from 'react'
import { BookMarked, Mic, Trash2 } from 'lucide-react'

import { Bar, Chip, EmptyState, ProgressRing, SectionTitle } from '../components/primitives'
import * as api from '../lib/api'

/**
 * The life list: what you have found, out of the 200 the model knows.
 *
 * Rejected open-set photos appear in the timeline but are excluded from the
 * species count — you did not see a bird, you photographed something else, and
 * quietly counting it would inflate the number this whole view exists to report.
 */
export function LifeListView() {
  const [data, setData] = useState(null)
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  // Bumping this re-runs the fetch. Deleting a sighting changes both the list
  // and the derived stats, so both are always refetched together rather than
  // patched locally and risking the two disagreeing.
  const [reloadKey, setReloadKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    Promise.all([api.sightings(), api.sightingStats()])
      .then(([list, s]) => {
        if (cancelled) return
        setData(list)
        setStats(s)
      })
      .catch(() => {
        if (!cancelled) setData({ sightings: [], total: 0 })
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [reloadKey])

  const remove = useCallback(async (id) => {
    await api.deleteSighting(id)
    setReloadKey((k) => k + 1)
  }, [])

  if (loading && !data) {
    return <div className="card p-10 text-sm text-(--color-ink-faint)">Loading your life list…</div>
  }

  if (!data?.sightings?.length) {
    return (
      <div className="card">
        <EmptyState icon={BookMarked} title="Your life list is empty">
          Identify a bird and choose “Save to life list”, or just say “save this”. Every sighting is
          kept locally in a SQLite file — nothing leaves your machine.
        </EmptyState>
      </div>
    )
  }

  const topFamilies = Object.entries(stats?.by_family ?? {}).slice(0, 6)
  const maxFamily = Math.max(1, ...topFamilies.map(([, n]) => n))

  return (
    <div className="space-y-6">
      {/* ── Summary ── */}
      <div className="grid gap-6 md:grid-cols-[auto_minmax(0,1fr)]">
        <section className="card flex items-center gap-6 p-6">
          <ProgressRing
            value={stats?.progress ?? 0}
            label={`${stats?.species_seen ?? 0}`}
            sublabel={`of ${stats?.total_species ?? 200}`}
          />
          <div>
            <p className="font-display text-2xl text-(--color-ink)">
              {stats?.species_seen ?? 0} species found
            </p>
            <p className="mt-1 text-sm text-(--color-ink-soft)">
              from {stats?.total_sightings ?? 0} sighting
              {stats?.total_sightings === 1 ? '' : 's'}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {stats?.audio_count > 0 && (
                <Chip>
                  <Mic size={11} strokeWidth={2} />
                  {stats.audio_count} by ear
                </Chip>
              )}
              {stats?.rejected_count > 0 && (
                <Chip tone="low" title="Photos the open-set gate rejected — not counted as species">
                  {stats.rejected_count} rejected
                </Chip>
              )}
            </div>
          </div>
        </section>

        <section className="card p-6">
          <SectionTitle>Families</SectionTitle>
          {topFamilies.length === 0 ? (
            <p className="text-sm text-(--color-ink-faint)">No family data yet.</p>
          ) : (
            <div className="space-y-3">
              {topFamilies.map(([family, count], i) => (
                <Bar
                  key={family}
                  index={i}
                  label={family}
                  value={count / maxFamily}
                  sublabel={`${count}`}
                />
              ))}
            </div>
          )}
        </section>
      </div>

      {/* ── Timeline ── */}
      <section className="card overflow-hidden">
        <header className="flex items-center justify-between border-b border-(--color-line) px-5 py-3.5">
          <h3 className="text-sm font-semibold text-(--color-ink)">Every sighting</h3>
          <span className="font-mono text-xs text-(--color-ink-faint)">{data.total}</span>
        </header>
        <ul>
          {data.sightings.map((sighting) => (
            <li
              key={sighting.id}
              className="group flex items-center gap-4 border-b border-(--color-line)/60 px-5 py-3 last:border-0"
            >
              {sighting.thumb ? (
                <img
                  src={api.thumbUrl(sighting.id)}
                  alt=""
                  className="size-12 shrink-0 rounded-lg object-cover"
                />
              ) : (
                <span className="grid size-12 shrink-0 place-items-center rounded-lg border border-(--color-line) text-(--color-ink-faint)">
                  {sighting.kind === 'audio' ? <Mic size={16} strokeWidth={2} /> : <BookMarked size={16} strokeWidth={2} />}
                </span>
              )}

              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-2">
                  <p className="truncate font-medium text-(--color-ink)">{sighting.display_name}</p>
                  {!sighting.is_bird && <Chip tone="low">rejected</Chip>}
                  {sighting.kind === 'audio' && <Chip>by call</Chip>}
                </div>
                <p className="mt-0.5 truncate text-xs text-(--color-ink-faint)">
                  {new Date(sighting.ts).toLocaleString()}
                  {sighting.family && ` · ${sighting.family}`}
                  {sighting.notes && ` · ${sighting.notes}`}
                </p>
              </div>

              {sighting.confidence != null && (
                <span className="shrink-0 font-mono text-xs tabular-nums text-(--color-ink-soft)">
                  {Math.round(sighting.confidence * 100)}%
                </span>
              )}

              <button
                onClick={() => remove(sighting.id)}
                aria-label={`Delete the ${sighting.display_name} sighting`}
                className="shrink-0 cursor-pointer rounded p-1.5 text-(--color-ink-faint) opacity-0 transition-all duration-200 group-hover:opacity-100 hover:text-(--color-reject) focus-visible:opacity-100"
              >
                <Trash2 size={14} strokeWidth={2} />
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
