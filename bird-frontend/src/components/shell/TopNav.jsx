import { useEffect, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { ArrowUpRight, Menu, X } from 'lucide-react'

import { Controls } from './Controls'
import { scrollToSection } from '../../hooks/useScrollSpy'

/**
 * The top bar.
 *
 * Over the hero it is bare type on the photograph; once the page scrolls it
 * gains a glass ground so the links stay legible over whatever passes beneath.
 * On desktop it shares the hero's five-column grid, so the links sit on the same
 * vertical lines the hero draws. On mobile it carries the wordmark, the global
 * controls and a menu.
 *
 * The active marker is the `layoutId` technique the old tab rail used: render
 * the dot only under the active link and the layout engine animates it across.
 */
export function TopNav({ sections, active, speech, voice, theme, onToggleTheme }) {
  const reduced = useReducedMotion()
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const go = (id) => {
    setOpen(false)
    scrollToSection(id)
  }

  const solid = scrolled || open

  return (
    <header
      className={`fixed inset-x-0 top-0 z-30 transition-[background-color,border-color,backdrop-filter] duration-300 lg:left-(--rail) ${
        solid
          ? 'border-b border-(--color-line) bg-(--color-base)/80 backdrop-blur-xl backdrop-saturate-150'
          : 'scope-night border-b border-transparent bg-transparent'
      }`}
    >
      <div className="flex h-(--nav-h) items-center justify-between px-5 sm:px-8 lg:grid lg:grid-cols-5 lg:px-0">
        {/* Mobile wordmark; on desktop the rail carries it. */}
        <button
          onClick={() => go('top')}
          className="cursor-pointer font-display text-xl font-extrabold tracking-[-0.04em] text-(--color-ink) lg:hidden"
        >
          BirdID<span className="text-(--color-accent)">.</span>
        </button>
        <span className="hidden lg:block" aria-hidden="true" />

        <nav aria-label="Sections" className="hidden items-center gap-6 lg:col-span-3 lg:flex lg:px-10 xl:gap-9">
          {sections.map(({ id, label }) => {
            const on = active === id
            return (
              <button
                key={id}
                onClick={() => go(id)}
                aria-current={on ? 'true' : undefined}
                className={`relative cursor-pointer py-2 text-[0.95rem] font-semibold tracking-[-0.005em] whitespace-nowrap transition-colors duration-200 ${
                  on ? 'text-(--color-ink)' : 'text-(--color-ink-soft) hover:text-(--color-ink)'
                }`}
              >
                {label}
                {on && (
                  <motion.span
                    layoutId="nav-dot"
                    aria-hidden="true"
                    className="absolute -bottom-1.5 left-1/2 size-1.5 -translate-x-1/2 rounded-full bg-(--color-accent)"
                    transition={reduced ? { duration: 0 } : { type: 'spring', stiffness: 420, damping: 34 }}
                  />
                )}
              </button>
            )
          })}
        </nav>

        <div className="flex items-center justify-end gap-2 lg:px-8">
          <button
            onClick={() => go('identify')}
            className="hidden cursor-pointer items-center gap-2 rounded-full bg-(--color-accent) py-2.5 pr-3 pl-5 text-sm font-semibold text-(--color-on-accent) transition-colors duration-200 hover:bg-(--color-accent-hover) lg:inline-flex"
          >
            Identify a bird
            <ArrowUpRight size={16} strokeWidth={2.25} />
          </button>

          <div className="lg:hidden">
            <Controls speech={speech} voice={voice} theme={theme} onToggleTheme={onToggleTheme} />
          </div>
          <button
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-controls="mobile-menu"
            aria-label={open ? 'Close menu' : 'Open menu'}
            className="grid size-10 cursor-pointer place-items-center rounded-full text-(--color-ink) lg:hidden"
          >
            {open ? <X size={20} strokeWidth={1.75} /> : <Menu size={20} strokeWidth={1.75} />}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.nav
            id="mobile-menu"
            aria-label="Sections"
            initial={reduced ? false : { opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, y: -8 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="border-t border-(--color-line) px-5 pt-2 pb-6 sm:px-8 lg:hidden"
          >
            {sections.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => go(id)}
                aria-current={active === id ? 'true' : undefined}
                className={`flex w-full cursor-pointer items-center justify-between border-b border-(--color-line) py-4 text-left font-display text-2xl font-bold tracking-[-0.02em] ${
                  active === id ? 'text-(--color-accent)' : 'text-(--color-ink)'
                }`}
              >
                {label}
                <ArrowUpRight size={20} strokeWidth={1.75} className="text-(--color-ink-faint)" />
              </button>
            ))}
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  )
}
