import { useCallback, useEffect, useState } from 'react'

/**
 * Light/dark theme, persisted, with the system setting as the default.
 *
 * Three states, not two — and that distinction is the whole reason this hook
 * exists rather than a boolean:
 *
 *   no stored choice  ->  follow the OS, and keep following it if the user
 *                         changes it while the app is open
 *   'light' / 'dark'  ->  an explicit choice, which must win over the OS
 *
 * The CSS in index.css is written to match: dark values apply under
 * `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme='light'])`,
 * and again under `:root[data-theme='dark']`. Setting the attribute is therefore
 * enough to override the system in either direction.
 */

const KEY = 'bird.theme'

function systemTheme() {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function useTheme() {
  const [stored, setStored] = useState(() => localStorage.getItem(KEY)) // 'light' | 'dark' | null
  const [system, setSystem] = useState(systemTheme)

  const theme = stored ?? system

  // Track the OS while no explicit choice has been made.
  useEffect(() => {
    const query = window.matchMedia?.('(prefers-color-scheme: dark)')
    if (!query) return
    const onChange = (event) => setSystem(event.matches ? 'dark' : 'light')
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])

  // Reflect the resolved theme onto the root element. When there is no explicit
  // choice the attribute is removed entirely, which hands control back to the
  // media query rather than pinning whatever the OS happened to be at load.
  useEffect(() => {
    const root = document.documentElement
    if (stored) root.setAttribute('data-theme', stored)
    else root.removeAttribute('data-theme')
  }, [stored])

  const setTheme = useCallback((next) => {
    if (next == null) {
      localStorage.removeItem(KEY)
      setStored(null)
      return
    }
    localStorage.setItem(KEY, next)
    setStored(next)
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'dark' ? 'light' : 'dark')
  }, [theme, setTheme])

  return { theme, isExplicit: stored != null, setTheme, toggleTheme, followSystem: () => setTheme(null) }
}
