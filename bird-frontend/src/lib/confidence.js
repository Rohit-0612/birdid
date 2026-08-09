/**
 * Confidence semantics — one mapping used by every surface.
 *
 * Lives apart from the components that consume it so that primitives.jsx
 * exports components only (mixing the two breaks React Fast Refresh, which then
 * full-reloads the page on every edit).
 *
 * Colour is never the only signal: the band label ships alongside it, so the
 * meaning survives colour blindness, greyscale printing and a screen reader.
 */

export const BAND = {
  high: {
    color: 'var(--color-high)',
    label: 'High confidence',
    hint: 'Very likely correct',
  },
  moderate: {
    color: 'var(--color-moderate)',
    label: 'Moderate confidence',
    hint: 'Probably right — check the field marks',
  },
  low: {
    color: 'var(--color-low)',
    label: 'Low confidence',
    hint: 'Treat as a guess. Try a clearer photo',
  },
}

export const bandOf = (band) => BAND[band] ?? BAND.low

export const pct = (n) => `${Math.round((n ?? 0) * 100)}%`
