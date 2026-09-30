import { motion } from 'motion/react'

import { bandOf, pct } from '../lib/confidence'

/**
 * Small shared components: SVG meters, chips, taxonomy trail, empty states.
 *
 * The meters are hand-rolled SVG rather than a chart library. There are only
 * three shapes in the whole app — an arc, a bar, a ring — and each is a dozen
 * lines. A charting dependency would be larger than the code it replaced and
 * would look like every other dashboard.
 *
 * Confidence colours and labels live in lib/confidence.js; this file exports
 * components only, so Fast Refresh can patch it in place.
 */

/** Semi-circular gauge. Reads as an instrument, not a pie chart. */
export function ConfidenceGauge({ value = 0, band = 'low', size = 168 }) {
  const { color, label } = bandOf(band)
  const stroke = 11
  const r = (size - stroke) / 2
  const cx = size / 2
  const cy = size / 2
  // Half circle, drawn left to right across the top.
  const arc = (fraction) => {
    const angle = Math.PI * (1 - Math.min(Math.max(fraction, 0), 1))
    return `${cx + r * Math.cos(angle)} ${cy - r * Math.sin(angle)}`
  }
  const track = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`
  const fill = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${arc(value)}`

  return (
    <div className="relative shrink-0" style={{ width: size, height: size / 2 + 26 }}>
      <svg width={size} height={size / 2 + 8} role="img" aria-label={`${label}, ${pct(value)}`}>
        <path d={track} fill="none" stroke="var(--color-line)" strokeWidth={stroke} strokeLinecap="round" />
        <motion.path
          d={fill}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{
            pathLength: { type: 'spring', stiffness: 42, damping: 14, mass: 0.9 },
            opacity: { duration: 0.25 },
          }}
        />
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <div className="font-mono text-3xl leading-none font-medium tabular-nums" style={{ color }}>
          {pct(value)}
        </div>
        <div className="mt-1 text-caption text-(--color-ink-faint)">
          confidence
        </div>
      </div>
    </div>
  )
}

/** Horizontal bar, used for the top-5 ranking and the species-heard list. */
export function Bar({ label, value, sublabel, color = 'var(--color-accent)', highlight = false, index = 0, onClick }) {
  const Wrapper = onClick ? 'button' : 'div'
  return (
    <Wrapper
      onClick={onClick}
      className={`group block w-full text-left ${onClick ? 'cursor-pointer' : ''}`}
    >
      <div className="mb-1 flex items-baseline justify-between gap-3">
        <span
          className={`truncate text-sm transition-colors duration-200 ${
            highlight ? 'font-semibold text-(--color-ink)' : 'text-(--color-ink-soft)'
          } ${onClick ? 'group-hover:text-(--color-accent-hover)' : ''}`}
        >
          {label}
        </span>
        <span className="shrink-0 font-mono text-xs tabular-nums text-(--color-ink-faint)">
          {sublabel ?? pct(value)}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-(--color-line)">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.max((value ?? 0) * 100, 1.5)}%` }}
          transition={{
            type: 'spring',
            stiffness: 90,
            damping: 18,
            delay: index * 0.05,
          }}
        />
      </div>
    </Wrapper>
  )
}

/** Ring for life-list progress: species found out of the full class list. */
export function ProgressRing({ value = 0, size = 148, label, sublabel }) {
  const stroke = 10
  const r = (size - stroke) / 2
  const circumference = 2 * Math.PI * r
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" role="img" aria-label={`${pct(value)} complete`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-line)" strokeWidth={stroke} />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference * (1 - Math.min(value, 1)) }}
          transition={{ type: 'spring', stiffness: 38, damping: 15, mass: 1.1 }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-2xl font-medium tabular-nums text-(--color-ink)">{label}</span>
        {sublabel && <span className="mt-0.5 text-[0.7rem] text-(--color-ink-faint)">{sublabel}</span>}
      </div>
    </div>
  )
}

export function Chip({ children, tone = 'neutral', title }) {
  const tones = {
    neutral: 'border-(--color-line) text-(--color-ink-soft)',
    accent: 'border-(--color-accent)/40 bg-(--color-accent)/10 text-(--color-accent-hover)',
    high: 'border-(--color-high)/40 bg-(--color-high)/10 text-(--color-high)',
    moderate: 'border-(--color-moderate)/40 bg-(--color-moderate)/10 text-(--color-moderate)',
    low: 'border-(--color-low)/40 bg-(--color-low)/10 text-(--color-low)',
  }
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-caption font-medium whitespace-nowrap ${tones[tone] ?? tones.neutral}`}
    >
      {children}
    </span>
  )
}

/** Taxonomy trail: Order → Family → Species. */
export function Taxonomy({ order, family, name }) {
  const steps = [order, family, name].filter(Boolean)
  if (steps.length < 2) return null
  return (
    <nav aria-label="Taxonomy" className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-(--color-ink-faint)">
      {steps.map((step, i) => (
        <span key={step + i} className="flex items-center gap-1.5">
          {i > 0 && <span aria-hidden="true" className="text-(--color-line)">›</span>}
          <span className={i === steps.length - 1 ? 'text-(--color-ink-soft)' : ''}>{step}</span>
        </span>
      ))}
    </nav>
  )
}

export function SectionTitle({ children, right }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-4">
      <h3 className="text-caption font-semibold text-(--color-ink-soft)">
        {children}
      </h3>
      {right}
    </div>
  )
}

export function Spinner({ label }) {
  return (
    <div className="flex items-center gap-3 text-sm text-(--color-ink-soft)">
      <span
        className="size-4 animate-spin rounded-full border-2 border-(--color-line) border-t-(--color-accent)"
        aria-hidden="true"
      />
      {label}
    </div>
  )
}

export function EmptyState({ icon: Icon, title, children }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      {Icon && (
        <span className="grid size-12 place-items-center rounded-full border border-(--color-line) text-(--color-ink-faint)">
          <Icon size={20} strokeWidth={1.5} />
        </span>
      )}
      <p className="font-display text-heading text-(--color-ink)">{title}</p>
      <p className="max-w-sm text-body text-(--color-ink-faint)">{children}</p>
    </div>
  )
}
