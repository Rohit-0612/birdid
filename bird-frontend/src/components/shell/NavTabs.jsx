import { AnimatePresence, motion, useReducedMotion } from 'motion/react'

/**
 * The view switcher.
 *
 * The moving highlight is the `layoutId` technique from ibelick's Animated Tabs
 * on 21st.dev (https://21st.dev/@ibelick/components/animated-tabs, MIT): render
 * the indicator only under the active item and give it a shared `layoutId`, and
 * the layout engine animates it between positions for you. No measuring, no
 * refs, no resize handling — and it stays correct when the labels reflow.
 *
 * The original is a generic `AnimatedBackground` that wraps arbitrary children
 * with Children.map and cloneElement. That indirection buys nothing here, where
 * there is one fixed list of views, so what is kept is the technique rather than
 * the abstraction.
 *
 * These are destinations, not tab panels, so the markup stays <nav> + <button> +
 * aria-current="page". Adding role="tablist" would promise arrow-key navigation
 * between panels that this app does not implement, and would take plain Tab
 * away from people who expect it.
 */
export function NavTabs({ views, value, onChange }) {
  const reduced = useReducedMotion()

  return (
    <nav aria-label="Views" className="mb-6">
      {/* p-1 and scroll-p-1 are not padding taste: the rail scrolls sideways on
          narrow screens, and without room inside it the 2px focus ring on the
          first and last buttons is clipped by the scroll container. */}
      <div className="card-glass inline-flex max-w-full gap-1 overflow-x-auto scroll-p-1 rounded-(--radius-pill) p-1">
        {views.map(({ id, label, icon: Icon }) => {
          const active = value === id
          return (
            <button
              key={id}
              type="button"
              onClick={() => onChange(id)}
              aria-current={active ? 'page' : undefined}
              className={`relative isolate inline-flex shrink-0 cursor-pointer items-center gap-2 rounded-(--radius-pill) px-3.5 py-2 text-caption font-medium transition-colors duration-200 ${
                active
                  ? 'text-(--color-accent-hover)'
                  : 'text-(--color-ink-faint) hover:text-(--color-ink-soft)'
              }`}
            >
              <AnimatePresence initial={false}>
                {active && (
                  <motion.span
                    layoutId="nav-indicator"
                    aria-hidden="true"
                    className="absolute inset-0 -z-10 rounded-(--radius-pill) bg-(--color-accent)/14 shadow-[inset_0_0_0_1px_color-mix(in_oklab,var(--color-accent)_28%,transparent)]"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={
                      reduced
                        ? { duration: 0 }
                        : { type: 'spring', stiffness: 420, damping: 34, mass: 0.7 }
                    }
                  />
                )}
              </AnimatePresence>

              <Icon size={15} strokeWidth={2} />
              {label}
            </button>
          )
        })}
      </div>
    </nav>
  )
}

export default NavTabs
