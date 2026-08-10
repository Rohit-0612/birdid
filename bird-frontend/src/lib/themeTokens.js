/**
 * Design tokens, for the parts of the app that render outside React.
 *
 * The canvas flock and the WebGL hero both need palette values, and both run in
 * a requestAnimationFrame loop where they cannot read React state. The obvious
 * fix — `getComputedStyle()` when you need a colour — is a trap: it forces a
 * style recalculation, and doing that inside a 60Hz loop means recalculating
 * layout-adjacent state sixty times a second to re-read a value that changes
 * about twice a day. (BirdFlock did exactly this; see the note in its frame loop.)
 *
 * So: read once, cache, and invalidate on an actual theme change.
 *
 * Both routes that can change the theme have to be watched, because useTheme has
 * three states, not two:
 *   - an explicit choice sets/removes `data-theme` on <html>  -> MutationObserver
 *   - no explicit choice falls through to the OS setting      -> matchMedia
 * Watching only the attribute misses the user changing their OS theme with the
 * app open; watching only the media query misses the in-app toggle.
 */

const cache = new Map()
const listeners = new Set()

let observer = null
let media = null

/**
 * Read a CSS custom property off the root element.
 *
 * `fallback` matters more than it looks: this is called during WebGL setup,
 * which can run before the stylesheet has applied in some load orders, and a
 * missing token would otherwise become `new THREE.Color('')` — which throws.
 */
export function readToken(name, fallback = '') {
  if (cache.has(name)) return cache.get(name)
  const value =
    typeof window === 'undefined'
      ? fallback
      : getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
  cache.set(name, value)
  return value
}

/** Same, as a number. For unitless tokens like opacities. */
export function readTokenNumber(name, fallback = 0) {
  const parsed = Number.parseFloat(readToken(name, String(fallback)))
  return Number.isFinite(parsed) ? parsed : fallback
}

function invalidate() {
  cache.clear()
  // Snapshot: a listener may unsubscribe itself while responding.
  for (const listener of [...listeners]) listener()
}

function attach() {
  if (typeof window === 'undefined') return
  observer = new MutationObserver(invalidate)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
  media = window.matchMedia?.('(prefers-color-scheme: dark)')
  media?.addEventListener('change', invalidate)
}

function detach() {
  observer?.disconnect()
  observer = null
  media?.removeEventListener('change', invalidate)
  media = null
}

/**
 * Call `onChange` whenever the resolved theme changes. Returns an unsubscriber.
 *
 * The cache is already cleared by the time `onChange` runs, so subscribers can
 * simply call `readToken` again and get fresh values.
 */
export function subscribeToTheme(onChange) {
  listeners.add(onChange)
  if (listeners.size === 1) attach()
  return () => {
    listeners.delete(onChange)
    if (listeners.size === 0) detach()
  }
}

/** True when the resolved theme is dark, by either route. */
export function isDarkTheme() {
  if (typeof window === 'undefined') return false
  const explicit = document.documentElement.getAttribute('data-theme')
  if (explicit) return explicit === 'dark'
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

/** Shared across every ambient renderer, so nobody re-implements the check. */
export const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false)
