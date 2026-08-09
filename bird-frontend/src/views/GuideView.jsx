import { useEffect, useMemo, useState } from 'react'
import { ArrowLeftRight, Loader2, Search, X } from 'lucide-react'

import { Chip, EmptyState, SectionTitle, Taxonomy } from '../components/primitives'
import * as api from '../lib/api'

/**
 * Browse all 200 species, and compare any two side by side.
 *
 * The compare tray is the point of this view. Nearly every wrong identification
 * in a fine-grained model is a confusion between two similar species, so being
 * able to pin the two candidates next to each other is the feature that actually
 * resolves a low-confidence result.
 */
export function GuideView({ focusFolder, speech }) {
  const [catalogue, setCatalogue] = useState([])
  const [query, setQuery] = useState('')
  const [familyFilter, setFamilyFilter] = useState('')
  // `undefined` means the user has not chosen yet, so an incoming focusFolder
  // wins; `null` means they explicitly closed the panel, which must stick.
  const [picked, setSelected] = useState(undefined)
  const [detail, setDetail] = useState(null)
  const [tray, setTray] = useState([])
  const [comparison, setComparison] = useState(null)
  const [comparing, setComparing] = useState(false)

  // Derived rather than copied into state by an effect.
  const selected = picked === undefined ? (focusFolder ?? null) : picked

  useEffect(() => {
    api.speciesList().then((d) => setCatalogue(d.species ?? [])).catch(() => setCatalogue([]))
  }, [])

  useEffect(() => {
    if (!selected) return
    // Guard against a slow response for a species the user has already moved on
    // from overwriting the one now on screen.
    let cancelled = false
    api
      .species(selected)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => !cancelled && setDetail(null))
    return () => {
      cancelled = true
    }
  }, [selected])

  const families = useMemo(
    () => [...new Set(catalogue.map((s) => s.family).filter(Boolean))].sort(),
    [catalogue],
  )

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return catalogue.filter((s) => {
      if (familyFilter && s.family !== familyFilter) return false
      if (!q) return true
      return (
        s.display_name.toLowerCase().includes(q) ||
        (s.scientific_name ?? '').toLowerCase().includes(q) ||
        (s.family ?? '').toLowerCase().includes(q) ||
        (s.order ?? '').toLowerCase().includes(q)
      )
    })
  }, [catalogue, query, familyFilter])

  function toggleTray(folder) {
    setComparison(null)
    setTray((prev) => {
      if (prev.includes(folder)) return prev.filter((f) => f !== folder)
      // Two slots: a three-way comparison is a table nobody reads.
      return [...prev, folder].slice(-2)
    })
  }

  async function runComparison() {
    if (tray.length !== 2) return
    setComparing(true)
    try {
      setComparison(await api.compare(tray[0], tray[1]))
    } catch {
      setComparison(null)
    } finally {
      setComparing(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* ── Search and filter ── */}
      <div className="card flex flex-wrap items-center gap-3 p-4">
        <div className="relative min-w-56 flex-1">
          <Search
            size={15}
            strokeWidth={2}
            className="absolute top-1/2 left-3 -translate-y-1/2 text-(--color-ink-faint)"
          />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search 200 species by name, family or order…"
            className="w-full rounded-lg border border-(--color-line) bg-(--color-raised) py-2 pr-3 pl-9 text-sm text-(--color-ink) placeholder:text-(--color-ink-faint) focus:border-(--color-accent)/50 focus:outline-none"
          />
        </div>
        <select
          value={familyFilter}
          onChange={(e) => setFamilyFilter(e.target.value)}
          className="cursor-pointer rounded-lg border border-(--color-line) bg-(--color-raised) px-3 py-2 text-sm text-(--color-ink-soft) focus:border-(--color-accent)/50 focus:outline-none"
        >
          <option value="">All families ({families.length})</option>
          {families.map((family) => (
            <option key={family} value={family}>
              {family}
            </option>
          ))}
        </select>
        <span className="font-mono text-xs text-(--color-ink-faint)">
          {filtered.length}/{catalogue.length}
        </span>
      </div>

      {/* ── Compare tray ── */}
      {tray.length > 0 && (
        <div className="card flex flex-wrap items-center gap-3 border-(--color-accent)/30 bg-(--color-accent)/5 p-4">
          <ArrowLeftRight size={15} strokeWidth={2} className="text-(--color-accent)" />
          <div className="flex flex-wrap gap-2">
            {tray.map((folder) => (
              <button
                key={folder}
                onClick={() => toggleTray(folder)}
                className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-(--color-accent)/40 bg-(--color-accent)/10 px-2.5 py-1 text-xs text-(--color-accent-hover) transition-colors hover:bg-(--color-accent)/20"
              >
                {catalogue.find((s) => s.folder === folder)?.display_name ?? folder}
                <X size={11} strokeWidth={2.5} />
              </button>
            ))}
          </div>
          <button
            onClick={runComparison}
            disabled={tray.length !== 2 || comparing}
            className="ml-auto inline-flex cursor-pointer items-center gap-2 rounded-lg bg-(--color-accent) px-3.5 py-2 text-sm font-medium text-(--color-on-accent) transition-colors duration-200 hover:bg-(--color-accent-hover) disabled:cursor-not-allowed disabled:opacity-40"
          >
            {comparing && <Loader2 size={14} className="animate-spin" />}
            {tray.length === 2 ? 'Compare these two' : 'Pick one more'}
          </button>
        </div>
      )}

      {comparison && <ComparisonTable comparison={comparison} speech={speech} onClose={() => setComparison(null)} />}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        {/* ── Grid ── */}
        <div className="min-w-0">
          {filtered.length === 0 ? (
            <div className="card">
              <EmptyState icon={Search} title="Nothing matches">
                No species matches “{query}”{familyFilter && ` in ${familyFilter}`}.
              </EmptyState>
            </div>
          ) : (
            <div className="grid gap-2.5 sm:grid-cols-2">
              {filtered.map((s) => (
                <div
                  key={s.folder}
                  className={`card cursor-pointer p-3.5 transition-colors duration-200 hover:border-(--color-accent)/40 ${
                    selected === s.folder ? 'border-(--color-accent)/60' : ''
                  }`}
                  onClick={() => setSelected(s.folder)}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="font-display truncate text-[1.05rem] text-(--color-ink)">
                        {s.display_name}
                      </p>
                      {s.scientific_name && (
                        <p className="truncate text-xs italic text-(--color-ink-faint)">
                          {s.scientific_name}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        toggleTray(s.folder)
                      }}
                      title="Add to the comparison tray"
                      className={`shrink-0 cursor-pointer rounded-md border px-1.5 py-1 transition-colors duration-200 ${
                        tray.includes(s.folder)
                          ? 'border-(--color-accent) text-(--color-accent-hover)'
                          : 'border-(--color-line) text-(--color-ink-faint) hover:border-(--color-accent)/50 hover:text-(--color-ink-soft)'
                      }`}
                    >
                      <ArrowLeftRight size={12} strokeWidth={2} />
                    </button>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {s.family && <Chip>{s.family}</Chip>}
                    {s.size_cm && <Chip>{s.size_cm}</Chip>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Detail ── */}
        <aside className="min-w-0">
          {selected && detail ? (
            <SpeciesDetail
              key={detail.folder}
              detail={detail}
              speech={speech}
              onClose={() => setSelected(null)}
            />
          ) : (
            <div className="card">
              <EmptyState title="Pick a species">
                Every entry carries taxonomy, size, diet, habitat, range, field marks and
                look-alikes. Use the ⇆ button on two of them to compare.
              </EmptyState>
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}

function SpeciesDetail({ detail, speech, onClose }) {
  return (
    <article className="card overflow-hidden">
      <header className="border-b border-(--color-line) px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <Taxonomy order={detail.order} family={detail.family} name={detail.display_name} />
            <h2 className="font-display mt-1.5 text-2xl leading-tight text-(--color-ink)">
              {detail.display_name}
            </h2>
            {detail.scientific_name && (
              <p className="text-sm italic text-(--color-ink-faint)">{detail.scientific_name}</p>
            )}
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="cursor-pointer rounded p-1 text-(--color-ink-faint) transition-colors hover:text-(--color-ink)"
          >
            <X size={15} strokeWidth={2} />
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {detail.conservation_status && <Chip>{detail.conservation_status}</Chip>}
          {detail.size_cm && <Chip>{detail.size_cm}</Chip>}
          {detail.has_curated_notes && (
            <Chip tone="accent" title="Has hand-written notes, not only generated ones">
              curated
            </Chip>
          )}
        </div>
      </header>

      <div className="space-y-4 px-5 py-4">
        {detail.field_marks?.length > 0 && (
          <section>
            <SectionTitle>Field marks</SectionTitle>
            <ul className="space-y-1.5">
              {detail.field_marks.map((mark) => (
                <li key={mark} className="flex gap-2 text-sm leading-relaxed text-(--color-ink-soft)">
                  <span className="mt-1.5 size-1 shrink-0 rounded-full bg-(--color-accent)" />
                  {mark}
                </li>
              ))}
            </ul>
          </section>
        )}

        {[
          ['Habitat', detail.habitat],
          ['Range', detail.range_description],
          ['Migration', detail.migration],
          ['Diet', detail.diet],
        ].map(([label, value]) =>
          value ? (
            <section key={label}>
              <SectionTitle>{label}</SectionTitle>
              <p className="text-sm leading-relaxed text-(--color-ink-soft)">{value}</p>
            </section>
          ) : null,
        )}

        {detail.similar_species?.length > 0 && (
          <section>
            <SectionTitle>Look-alikes</SectionTitle>
            <div className="space-y-2">
              {detail.similar_species.map((s) => (
                <div key={s.name} className="rounded-lg border border-(--color-line) p-3">
                  <p className="text-sm font-medium text-(--color-ink)">{s.name}</p>
                  <p className="mt-1 text-xs leading-relaxed text-(--color-ink-faint)">
                    {s.how_to_distinguish}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {detail.fun_fact && (
          <p className="rounded-lg border border-(--color-accent)/20 bg-(--color-accent)/5 p-3 text-sm leading-relaxed text-(--color-ink-soft)">
            {detail.fun_fact}
          </p>
        )}

        {speech.supported && (
          <button
            onClick={() =>
              speech.speak(
                `${detail.display_name}. ${detail.field_marks?.slice(0, 2).join(' ') ?? ''} ${detail.habitat ?? ''}`,
              )
            }
            className="w-full cursor-pointer rounded-lg border border-(--color-line) py-2 text-sm text-(--color-ink-soft) transition-colors duration-200 hover:border-(--color-accent)/50 hover:text-(--color-ink)"
          >
            Read this entry aloud
          </button>
        )}
      </div>
    </article>
  )
}

function ComparisonTable({ comparison, speech, onClose }) {
  return (
    <section className="card overflow-hidden">
      <header className="flex items-center justify-between gap-3 border-b border-(--color-line) px-5 py-3.5">
        <h3 className="font-display text-lg text-(--color-ink)">
          {comparison.a.display_name} <span className="text-(--color-ink-faint)">vs</span>{' '}
          {comparison.b.display_name}
        </h3>
        <button
          onClick={onClose}
          aria-label="Close comparison"
          className="cursor-pointer rounded p-1 text-(--color-ink-faint) transition-colors hover:text-(--color-ink)"
        >
          <X size={15} strokeWidth={2} />
        </button>
      </header>

      <div className="border-b border-(--color-line) bg-(--color-accent)/5 px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <p className="text-sm leading-relaxed text-(--color-ink-soft)">{comparison.summary}</p>
          {speech.supported && (
            <button
              onClick={() => speech.speak(comparison.summary)}
              className="shrink-0 cursor-pointer rounded-lg border border-(--color-line) px-2.5 py-1.5 text-xs text-(--color-ink-soft) transition-colors hover:text-(--color-ink)"
            >
              Speak
            </button>
          )}
        </div>
        <p className="mt-2 text-[0.7rem] text-(--color-ink-faint)">
          {comparison.generated
            ? 'Summary written by the local model from the fields below — the numbers are not generated.'
            : 'Local model unavailable; this summary comes straight from the knowledge base.'}
          {comparison.known_look_alike && ' These two are a recorded look-alike pair.'}
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-sm">
          <thead>
            <tr className="border-b border-(--color-line) text-left">
              <th className="px-5 py-2.5 text-[0.68rem] font-semibold tracking-wider text-(--color-ink-faint) uppercase">
                Field
              </th>
              <th className="px-5 py-2.5 text-[0.68rem] font-semibold tracking-wider text-(--color-ink-faint) uppercase">
                {comparison.a.display_name}
              </th>
              <th className="px-5 py-2.5 text-[0.68rem] font-semibold tracking-wider text-(--color-ink-faint) uppercase">
                {comparison.b.display_name}
              </th>
            </tr>
          </thead>
          <tbody>
            {comparison.rows.map((row) => (
              <tr
                key={row.label}
                className={`border-b border-(--color-line)/60 align-top ${
                  row.differs ? '' : 'opacity-55'
                }`}
              >
                <td className="px-5 py-2.5 text-xs whitespace-nowrap text-(--color-ink-faint)">
                  {row.label}
                </td>
                <td className="px-5 py-2.5 leading-relaxed text-(--color-ink-soft)">{row.a ?? '—'}</td>
                <td className="px-5 py-2.5 leading-relaxed text-(--color-ink-soft)">{row.b ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="px-5 py-3 text-[0.7rem] text-(--color-ink-faint)">
        Rows that match on both species are dimmed, so what is left is what tells them apart.
      </p>
    </section>
  )
}
