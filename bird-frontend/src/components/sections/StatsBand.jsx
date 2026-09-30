import { Chip } from '../primitives'

/**
 * What the models on this machine can actually do, read from /api/health.
 *
 * Numbers only appear once the API has answered; until then each cell holds
 * its place so the band does not jump. The status chips the old masthead
 * carried — device, open-set, audit state — live here now, beside the figures
 * they qualify.
 */
export function StatsBand({ health }) {
  const tpr = health?.openset?.enabled ? health.openset.tpr : null
  const cells = [
    {
      value: health?.num_species,
      label: 'species the photo model was trained on',
    },
    {
      value: health?.verifier?.species?.toLocaleString(),
      label: 'more a second model can name when the first is unsure',
    },
    {
      value: '6,500',
      label: 'species BirdNET can pick out of a recording',
    },
    {
      value: tpr != null ? `${Math.round(tpr * 100)}%` : health ? 'off' : undefined,
      label: 'of known species pass the gate that rejects non-birds',
    },
  ]

  return (
    <section aria-label="What it can do" className="border-b border-(--color-line)">
      <div className="grid grid-cols-2 lg:grid-cols-5">
        <div className="col-span-2 flex flex-col justify-center gap-3 border-b border-(--color-line) px-6 py-8 sm:px-10 lg:col-span-1 lg:border-r lg:border-b-0 lg:px-10 xl:px-14">
          <p className="font-display text-heading font-bold text-(--color-ink)">Running on</p>
          {health ? (
            <div className="flex flex-wrap gap-1.5">
              <Chip title="Compute device the model is running on">{health.device_label}</Chip>
              {health.openset?.enabled ? (
                <Chip tone="high" title="Photos that are not one of the known species get rejected">
                  open-set on
                </Chip>
              ) : (
                <Chip tone="moderate" title="Run scripts/fit_openset.py to enable rejection">
                  open-set off
                </Chip>
              )}
              {!health.model_audited && (
                <Chip tone="moderate" title="This checkpoint has no recorded training split — see How it works">
                  unaudited
                </Chip>
              )}
            </div>
          ) : (
            <p className="text-caption text-(--color-ink-faint)">Waking up the server — this can take a minute…</p>
          )}
        </div>

        {cells.map((cell, i) => (
          <div
            key={cell.label}
            className={`flex flex-col gap-3 border-(--color-line) px-6 py-8 sm:px-10 lg:px-8 lg:py-10 xl:px-10 ${
              i % 2 === 0 ? 'border-r' : 'lg:border-r'
            } ${i < 2 ? 'border-b lg:border-b-0' : ''} ${
              i === cells.length - 1 ? 'lg:border-r-0' : ''
            }`}
          >
            <span className="font-display text-[clamp(2.25rem,1.6rem+2vw,3.5rem)] leading-none font-extrabold tracking-[-0.04em] text-(--color-ink) tabular-nums">
              {cell.value ?? <span className="text-(--color-line)">—</span>}
            </span>
            <span className="max-w-[24ch] text-caption text-(--color-ink-soft)">{cell.label}</span>
          </div>
        ))}
      </div>
    </section>
  )
}
