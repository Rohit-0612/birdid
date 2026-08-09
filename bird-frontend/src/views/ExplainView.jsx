import { AlertTriangle, Check, Cpu, Database, ShieldQuestion, X } from 'lucide-react'

import { Bar, Chip, SectionTitle } from '../components/primitives'

/**
 * What this model can and cannot do.
 *
 * The unusual thing about this project is that its accuracy numbers are known to
 * be wrong — the checkpoint has no recorded training split and measured evidence
 * says it saw essentially all of CUB. That fact is worth a whole view rather than
 * a footnote, and stating it plainly is more useful than a headline percentage
 * that would be a lie.
 */
export function ExplainView({ health }) {
  if (!health) {
    return <div className="card p-10 text-sm text-(--color-ink-faint)">Loading model details…</div>
  }

  const gate = health.openset ?? {}
  const heldOut = gate.tpr != null

  return (
    <div className="space-y-6">
      {/* ── The honest caveat ── */}
      {health.caveat && (
        <section className="card border-(--color-moderate)/30 bg-(--color-moderate)/5 p-6">
          <div className="flex gap-3">
            <AlertTriangle size={20} strokeWidth={2} className="mt-0.5 shrink-0 text-(--color-moderate)" />
            <div>
              <h2 className="font-display text-xl text-(--color-ink)">
                This checkpoint's accuracy cannot be trusted
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-(--color-ink-soft)">{health.caveat}</p>
              <p className="mt-3 text-sm leading-relaxed text-(--color-ink-faint)">
                Concretely: it scores about 95% on official CUB test images that were never held
                out, which is above the realistic ceiling of roughly 86–88% for this architecture.
                Images it supposedly never saw score higher than images it did. That is the
                signature of a data leak, not of a good model. The species it names are still
                usually right; the confidence number just is not a calibrated probability.
              </p>
              <p className="mt-3 text-sm leading-relaxed text-(--color-ink-faint)">
                The fix already exists in the repository:{' '}
                <code className="rounded bg-(--color-void)/60 px-1.5 py-0.5 font-mono text-xs">train.py</code>{' '}
                trains against the committed split manifests and records their SHA-256 in the
                checkpoint. Any checkpoint carrying that hash is picked up automatically and this
                banner disappears.
              </p>
            </div>
          </div>
        </section>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* ── Model ── */}
        <section className="card p-6">
          <SectionTitle>The photo model</SectionTitle>
          <dl className="space-y-2.5 text-sm">
            <Row label="Architecture" value="EfficientNetV2-S" />
            <Row label="Checkpoint" value={health.checkpoint} mono />
            <Row label="Species" value={health.num_species} mono />
            <Row label="Input" value={`${health.input_size} × ${health.input_size} px`} mono />
            <Row label="Device" value={health.device_label} />
            <Row
              label="Split recorded"
              value={
                health.model_audited ? (
                  <Chip tone="high">
                    <Check size={11} strokeWidth={2.5} />
                    yes
                  </Chip>
                ) : (
                  <Chip tone="low">
                    <X size={11} strokeWidth={2.5} />
                    no
                  </Chip>
                )
              }
            />
          </dl>
        </section>

        {/* ── Knowledge base ── */}
        <section className="card p-6">
          <SectionTitle>The knowledge base</SectionTitle>
          <dl className="space-y-2.5 text-sm">
            <Row label="Records" value={`${health.kb_records} / ${health.num_species}`} mono />
            <Row label="Written by" value={health.kb_source ?? 'n/a'} mono />
            <Row label="Expert-verified" value={<Chip tone="moderate">no</Chip>} />
            <Row label="Hand-written habitat notes" value={`${health.coverage?.habitat ?? 0} species`} mono />
            <Row label="Hand-written migration notes" value={`${health.coverage?.migration ?? 0} species`} mono />
            <Row label="Hand-written look-alikes" value={`${health.coverage?.similar ?? 0} species`} mono />
          </dl>
          <p className="mt-4 text-xs leading-relaxed text-(--color-ink-faint)">
            Curated text always wins where it exists; the generated records only fill genuine gaps.
            Look-alike entries labelled “verified” in the species card are the hand-written ones.
          </p>
        </section>

        {/* ── Open-set gate ── */}
        <section className="card p-6 lg:col-span-2">
          <SectionTitle
            right={
              gate.enabled ? (
                <Chip tone="high">active</Chip>
              ) : (
                <Chip tone="moderate">not fitted</Chip>
              )
            }
          >
            The “is it even a bird?” gate
          </SectionTitle>

          {!gate.enabled ? (
            <p className="text-sm leading-relaxed text-(--color-ink-soft)">
              No threshold has been fitted yet, so every photo is treated as one of the{' '}
              {health.num_species} known species — including photos of dogs. Run{' '}
              <code className="rounded bg-(--color-void)/60 px-1.5 py-0.5 font-mono text-xs">
                python3 scripts/fit_openset.py
              </code>{' '}
              to enable it.
            </p>
          ) : (
            <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <div>
                <p className="text-sm leading-relaxed text-(--color-ink-soft)">
                  A 200-way softmax has no way to say “none of the above” — it renormalises over the
                  classes it knows. The gate reads the raw logits instead and rejects inputs that do
                  not look like any known species.
                </p>
                <dl className="mt-4 space-y-2.5 text-sm">
                  <Row label="Score" value={`${gate.method} (T = ${gate.temperature})`} mono />
                  <Row label="Threshold" value={gate.threshold?.toFixed(3)} mono />
                  <Row label="Fitted" value={gate.fitted ?? 'unknown'} mono />
                </dl>
              </div>

              <div>
                <p className="mb-3 text-xs text-(--color-ink-faint)">
                  Share of each category accepted as a known bird. High is good for the first row,
                  low is good for the rest.
                </p>
                <div className="space-y-3">
                  <Bar
                    label="Real birds it knows"
                    value={gate.tpr ?? 0}
                    color="var(--color-high)"
                    highlight
                    index={0}
                  />
                  <Bar
                    label="Other birds, outside the 200"
                    value={gate.fpr_near_bird ?? 0}
                    color="var(--color-moderate)"
                    index={1}
                  />
                  <Bar
                    label="Other animals"
                    value={gate.fpr_animal ?? 0}
                    color="var(--color-low)"
                    index={2}
                  />
                  <Bar
                    label="Not an animal at all"
                    value={gate.fpr_nonanimal ?? 0}
                    color="var(--color-low)"
                    index={3}
                  />
                </div>
                {heldOut && (
                  <p className="mt-3 text-xs leading-relaxed text-(--color-ink-faint)">
                    The honest limitation: it reliably rejects things that are not birds, and only
                    partially rejects birds that simply are not among the 200. Bird features fire
                    for a macaw by design.
                  </p>
                )}
              </div>
            </div>
          )}
        </section>

        {/* ── Audio + LLM ── */}
        <section className="card p-6">
          <SectionTitle>Call identification</SectionTitle>
          <dl className="space-y-2.5 text-sm">
            <Row label="Model" value="BirdNET (Cornell Lab)" />
            <Row label="Loaded" value={health.birdnet ? <Chip tone="high">yes</Chip> : <Chip tone="low">no</Chip>} />
            <Row label="Species covered" value="~6,500" mono />
          </dl>
          <p className="mt-4 text-xs leading-relaxed text-(--color-ink-faint)">
            BirdNET covers far more species than the photo model, so the Listen view can name birds
            the Identify view cannot. Species are matched back to the 200 by name, which is a
            heuristic — abbreviated CUB folder names like “Cardinal” and misspellings like
            “Artic_Tern” do not line up cleanly with standard names.
          </p>
        </section>

        <section className="card p-6">
          <SectionTitle>Language model</SectionTitle>
          <dl className="space-y-2.5 text-sm">
            <Row label="Runtime" value="Ollama, local" />
            <Row label="Model" value={health.llm?.model ?? 'n/a'} mono />
            <Row
              label="Reachable"
              value={health.llm?.ok ? <Chip tone="high">yes</Chip> : <Chip tone="moderate">no</Chip>}
            />
          </dl>
          <p className="mt-4 text-xs leading-relaxed text-(--color-ink-faint)">
            {health.llm?.ok
              ? 'Used for the spoken summaries, the chat answers and the comparison prose. Facts are passed in from the knowledge base and echoed, never recalled from the model, so sizes and conservation statuses cannot be invented.'
              : `Not reachable (${health.llm?.reason ?? 'unknown'}). Spoken summaries and comparisons fall back to template text built from the knowledge base; nothing else is affected.`}
          </p>
        </section>
      </div>

      <section className="card flex flex-wrap gap-6 p-6 text-xs text-(--color-ink-faint)">
        <span className="inline-flex items-center gap-2">
          <Cpu size={13} strokeWidth={2} /> Everything runs locally
        </span>
        <span className="inline-flex items-center gap-2">
          <Database size={13} strokeWidth={2} /> Life list stored in SQLite on this machine
        </span>
        <span className="inline-flex items-center gap-2">
          <ShieldQuestion size={13} strokeWidth={2} /> No API keys, no telemetry, no network calls
        </span>
      </section>
    </div>
  )
}

function Row({ label, value, mono }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-(--color-line)/50 pb-2 last:border-0">
      <dt className="text-(--color-ink-faint)">{label}</dt>
      <dd className={`text-right text-(--color-ink-soft) ${mono ? 'font-mono text-xs tabular-nums' : ''}`}>
        {value}
      </dd>
    </div>
  )
}
