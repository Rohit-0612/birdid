import { useCallback, useEffect, useState } from 'react'

/**
 * Dark/light theme, persisted, dark by default.
 *
 * The design is built dark-first — a wildlife print on a dark mount — so with no
 * stored choice the app is dark regardless of the OS. Light is a deliberate
 * choice made with the toggle and remembered.
 *
 * index.css matches: dark values live in @theme, and light values apply under
 * `:root[data-theme='light']`. The attribute is removed when there is no stored
 * choice, which leaves the dark default in charge.
 */

const KEY = 'bird.theme'
const DEFAULT_THEME = 'dark'

function readStored() {
  try {
    return localStorage.getItem(KEY)
  } catch {
    return null
  }
}

export function useTheme() {
  const [stored, setStored] = useState(readStored) // 'light' | 'dark' | null

  const theme = stored ?? DEFAULT_THEME

  useEffect(() => {
    const root = document.documentElement
    if (stored) root.setAttribute('data-theme', stored)
    else root.removeAttribute('data-theme')
  }, [stored])

  const setTheme = useCallback((next) => {
    try {
      if (next == null) localStorage.removeItem(KEY)
      else localStorage.setItem(KEY, next)
    } catch {
      /* private mode — the choice just won't survive a reload */
    }
    setStored(next ?? null)
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'dark' ? 'light' : 'dark')
  }, [theme, setTheme])

  return { theme, isExplicit: stored != null, setTheme, toggleTheme, resetTheme: () => setTheme(null) }
}
