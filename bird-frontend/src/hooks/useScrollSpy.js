import { useEffect, useRef, useState } from 'react'

/**
 * Which section is under the reading line.
 *
 * The reading line is a thin band just above the middle of the viewport: a
 * section is "active" while any part of it crosses that band. One observer, no
 * scroll listener, so it costs nothing while the page is still.
 *
 * `onEnter(id)` fires each time a section becomes active — an event, for work
 * that should happen on arrival rather than be derived from state.
 */
export function useScrollSpy(ids, initial = ids[0], onEnter) {
  const [active, setActive] = useState(initial)
  const onEnterRef = useRef(onEnter)

  useEffect(() => {
    onEnterRef.current = onEnter
  })

  useEffect(() => {
    const elements = ids.map((id) => document.getElementById(id)).filter(Boolean)
    if (!elements.length || !('IntersectionObserver' in window)) return

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue
          setActive(entry.target.id)
          onEnterRef.current?.(entry.target.id)
        }
      },
      { rootMargin: '-42% 0px -52% 0px' },
    )
    elements.forEach((el) => observer.observe(el))
    return () => observer.disconnect()
    // ids is a static list from the caller; joining keeps the dependency stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids.join('|')])

  return active
}

/** Smooth-scroll a section into view, or jump when motion is reduced. */
export function scrollToSection(id) {
  const el = document.getElementById(id)
  if (!el) return
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' })
}
