import { Controls } from './Controls'
import { scrollToSection } from '../../hooks/useScrollSpy'

/**
 * The spine of the page, desktop only.
 *
 * Wordmark at the top, one tick per section in the middle — the active one runs
 * long, so the rail doubles as a scroll position indicator — and the global
 * controls at the foot. Always night, like the hero it borders.
 */
export function SideRail({ sections, active, speech, voice, theme, onToggleTheme }) {
  return (
    <aside className="scope-night fixed inset-y-0 left-0 z-40 hidden w-(--rail) flex-col items-center justify-between border-r border-(--color-line) bg-(--color-base) py-8 lg:flex">
      <button
        onClick={() => scrollToSection('top')}
        className="cursor-pointer font-display text-[1.3rem] leading-none font-extrabold tracking-[-0.04em] text-(--color-ink)"
        aria-label="BirdID, back to top"
      >
        BirdID<span className="text-(--color-accent)">.</span>
      </button>

      <nav aria-label="Sections" className="flex flex-col items-center gap-4">
        {sections.map(({ id, label }) => {
          const on = active === id
          return (
            <button
              key={id}
              onClick={() => scrollToSection(id)}
              aria-label={label}
              aria-current={on ? 'true' : undefined}
              title={label}
              className="group flex h-4 w-12 cursor-pointer items-center justify-start"
            >
              <span
                className={`block h-px transition-all duration-500 ease-(--ease-out-soft) ${
                  on
                    ? 'w-12 bg-(--color-ink)'
                    : 'w-7 bg-(--color-ink-faint)/60 group-hover:w-9 group-hover:bg-(--color-ink-soft)'
                }`}
              />
            </button>
          )
        })}
      </nav>

      <Controls speech={speech} voice={voice} theme={theme} onToggleTheme={onToggleTheme} vertical />
    </aside>
  )
}
