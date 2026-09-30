import { useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion, useScroll, useTransform } from 'motion/react'
import { ArrowLeft, ArrowRight, ArrowUpRight } from 'lucide-react'

import { HERO_SLIDES, PHOTOS } from '../../lib/photos'
import { scrollToSection } from '../../hooks/useScrollSpy'

/**
 * The opening frame.
 *
 * Five columns drawn as hairlines, a photograph filling the right four, the
 * headline in the dark first column, and a row of panels along the foot — the
 * two ways to identify a bird, and the birds beyond the trained 200. The two
 * rounded tiles top right are the next slides; clicking one brings it forward,
 * as do the arrows. Nothing here advances on its own.
 *
 * Always night: the photographs were chosen for dark or warm grounds, and the
 * headline needs that ground under it in either theme.
 */

const EASE = [0.22, 1, 0.36, 1]

export function Hero({ health }) {
  const reduced = useReducedMotion()
  const [index, setIndex] = useState(0)
  const ref = useRef(null)

  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] })
  const photoY = useTransform(scrollYProgress, [0, 1], ['0%', '14%'])

  const count = HERO_SLIDES.length
  const slide = HERO_SLIDES[index]
  const upcoming = [1, 2].map((step) => (index + step) % count)
  const go = (next) => setIndex((next + count) % count)

  const species = health?.num_species ?? 200
  const beyond = health?.verifier?.species

  // One orchestrated entrance: the headline rises line by line, then the rest
  // settles in behind it. Everything after this moment on the page is still.
  const rise = (delay) =>
    reduced
      ? {}
      : {
          initial: { opacity: 0, y: 24 },
          animate: { opacity: 1, y: 0 },
          transition: { duration: 0.9, delay, ease: EASE },
        }
  const line = (delay) =>
    reduced
      ? {}
      : {
          initial: { y: '105%' },
          animate: { y: '0%' },
          transition: { duration: 1.05, delay, ease: EASE },
        }

  return (
    <section
      id="top"
      ref={ref}
      aria-label="BirdID"
      className="scope-night relative isolate overflow-hidden bg-(--color-base)"
    >
      {/* ── The photograph ── */}
      <div className="absolute inset-x-0 top-0 h-[78svh] min-h-[32rem] overflow-hidden lg:inset-y-0 lg:left-[20%] lg:h-auto lg:[mask-image:linear-gradient(to_right,transparent,black_24%)]">
        <motion.div className="absolute inset-0" style={reduced ? undefined : { y: photoY }}>
          <AnimatePresence initial={false}>
            <motion.img
              key={slide.src}
              src={slide.src}
              alt={slide.alt}
              fetchPriority={index === 0 ? 'high' : undefined}
              className="absolute inset-0 size-full scale-[1.08] object-cover"
              style={{ objectPosition: slide.position }}
              initial={reduced ? { opacity: 0 } : { opacity: 0, scale: 1.14 }}
              animate={{ opacity: 1, scale: 1.08 }}
              exit={{ opacity: 0 }}
              transition={{ duration: reduced ? 0.2 : 1.2, ease: EASE }}
            />
          </AnimatePresence>
        </motion.div>
        <div className="grain pointer-events-none absolute inset-0" aria-hidden="true" />
        {/* Legibility: fade the foot of the photo into the page on small screens,
            and darken the left edge where the headline crosses it on large ones. */}
        <div
          aria-hidden="true"
          className="absolute inset-0 bg-[linear-gradient(to_bottom,color-mix(in_oklab,var(--color-base)_60%,transparent),transparent_22%),linear-gradient(to_top,var(--color-base)_4%,transparent_55%)] lg:bg-[linear-gradient(to_top,color-mix(in_oklab,var(--color-base)_70%,transparent),transparent_30%)]"
        />
      </div>

      {/* ── The column lines ── */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 hidden grid-cols-5 lg:grid">
        {Array.from({ length: 5 }, (_, i) => (
          <span key={i} className={i === 0 ? '' : 'border-l border-(--color-ink)/12'} />
        ))}
      </div>
      {/* The band the nav sits in, as in a print masthead. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 hidden h-(--nav-h) bg-(--color-ink)/[0.045] lg:left-[20%] lg:block"
      />

      {/* ── Content ── */}
      <div className="relative grid min-h-[100svh] grid-cols-1 lg:grid-cols-5 lg:grid-rows-[var(--nav-h)_1fr_auto]">
        {/* Next-slide tiles */}
        <div className="hidden justify-end gap-4 px-8 pt-5 lg:col-span-2 lg:col-start-4 lg:row-start-2 lg:flex lg:items-start">
          {upcoming.map((i, n) => {
            const s = HERO_SLIDES[i]
            return (
              <motion.button
                key={s.src}
                onClick={() => go(i)}
                aria-label={`Show the ${s.name}`}
                className="group relative h-44 w-36 cursor-pointer overflow-hidden rounded-(--radius-photo) shadow-[var(--shadow-lift)] xl:h-52 xl:w-44"
                {...rise(0.55 + n * 0.08)}
              >
                <img
                  src={s.thumb}
                  alt=""
                  className="size-full object-cover transition-transform duration-700 ease-(--ease-out-soft) group-hover:scale-105"
                  style={{ objectPosition: s.position }}
                />
                <span className="absolute inset-x-0 bottom-0 bg-[linear-gradient(to_top,rgba(10,8,6,0.75),transparent)] px-3.5 pt-8 pb-3 text-left text-xs font-semibold text-[#f7f1e6]">
                  {s.name}
                </span>
              </motion.button>
            )
          })}
        </div>

        {/* Headline */}
        <div className="flex flex-col justify-end px-6 pt-[46svh] pb-10 sm:px-10 lg:col-span-3 lg:col-start-1 lg:row-start-2 lg:justify-center lg:px-10 lg:py-0 xl:px-14">
          <h1 className="font-display text-hero font-extrabold text-(--color-ink)">
            <span className="block overflow-hidden pb-[0.06em]">
              <motion.span className="block" {...line(0.1)}>
                Name that
              </motion.span>
            </span>
            <span className="block overflow-hidden pb-[0.08em]">
              <motion.span className="block" {...line(0.2)}>
                bird.
              </motion.span>
            </span>
          </h1>

          <motion.button
            onClick={() => scrollToSection('identify')}
            className="group mt-6 inline-flex w-fit cursor-pointer items-center gap-4 text-left text-lg font-semibold text-(--color-ink) sm:text-xl"
            {...rise(0.45)}
          >
            Show it a photo or play it a call
            <span className="grid size-11 shrink-0 place-items-center rounded-full bg-(--color-ink) text-(--color-base) transition-transform duration-300 ease-(--ease-out-soft) group-hover:translate-x-1">
              <ArrowRight size={20} strokeWidth={2.25} />
            </span>
          </motion.button>
          <motion.p className="mt-4 max-w-md text-body text-(--color-ink-soft)" {...rise(0.55)}>
            It names the species, says how sure it is, and shows you how to confirm it — then files
            the bird in your deck.
          </motion.p>
        </div>

        {/* ── Foot row ── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:col-span-5 lg:row-start-3 lg:grid-cols-5">
          {/* Caption for the current slide */}
          <motion.div
            className="order-last flex flex-col justify-end gap-1 px-6 py-6 sm:col-span-2 sm:px-10 lg:order-none lg:col-span-1 lg:px-10 xl:px-14"
            {...rise(0.7)}
          >
            <p className="font-mono text-xs text-(--color-ink-faint) tabular-nums" aria-live="polite">
              {String(index + 1).padStart(2, '0')} / {String(count).padStart(2, '0')}
            </p>
            <p className="font-display text-[1.05rem] font-bold text-(--color-ink)">{slide.name}</p>
            <p className="text-caption text-(--color-ink-soft) italic">{slide.sci}</p>
            <a
              href={slide.source}
              target="_blank"
              rel="noreferrer"
              className="mt-1 w-fit text-micro text-(--color-ink-faint) underline-offset-2 hover:text-(--color-ink-soft) hover:underline"
            >
              Photo: {slide.author}, {slide.license}
            </a>
          </motion.div>

          <FootCard
            number="01."
            title="Identify by photo"
            body={`${species} species, each with the field marks that confirm it.`}
            onClick={() => scrollToSection('identify')}
            className="bg-(--color-paper) text-(--color-paper-ink)"
            motionProps={rise(0.62)}
          />
          <FootCard
            number="02."
            title="Identify by call"
            body="BirdNET listens for about 6,500 species, many the photo model has never seen."
            onClick={() => scrollToSection('listen')}
            className="bg-(--color-panel) text-(--color-on-panel)"
            motionProps={rise(0.7)}
          />

          <motion.button
            onClick={() => scrollToSection('deck')}
            className="group relative isolate min-h-60 cursor-pointer overflow-hidden text-left sm:col-span-2 lg:col-span-1"
            {...rise(0.78)}
          >
            <img
              src={PHOTOS.scarlet.thumb}
              alt={PHOTOS.scarlet.alt}
              className="absolute inset-0 -z-10 size-full object-cover transition-transform duration-700 ease-(--ease-out-soft) group-hover:scale-105"
              style={{ objectPosition: PHOTOS.scarlet.position }}
            />
            <span className="absolute inset-0 -z-10 bg-[linear-gradient(to_top,rgba(10,8,6,0.92),rgba(10,8,6,0.55)_55%,rgba(10,8,6,0.15))]" />
            <span className="flex h-full flex-col justify-end p-7 text-[#f7f1e6] xl:p-9">
              <span className="font-display text-lg leading-snug font-bold">Beyond the 200</span>
              <span className="mt-2 text-sm leading-relaxed text-[#e6dccb]">
                A second model can name{' '}
                {beyond ? `${beyond.toLocaleString()} more birds` : 'thousands more birds'}, and they
                go in your deck too.
              </span>
            </span>
            <ArrowUpRight
              size={20}
              strokeWidth={2}
              className="absolute top-6 right-6 text-[#f7f1e6] opacity-0 transition-opacity duration-300 group-hover:opacity-100"
            />
          </motion.button>

          {/* Slide controls */}
          <motion.div
            className="flex items-center justify-center gap-4 px-6 py-6 sm:col-span-2 lg:col-span-1 lg:flex-col lg:gap-5"
            {...rise(0.85)}
          >
            <ArrowButton label="Previous photo" onClick={() => go(index - 1)}>
              <ArrowLeft size={20} strokeWidth={1.75} />
            </ArrowButton>
            <ArrowButton label="Next photo" onClick={() => go(index + 1)}>
              <ArrowRight size={20} strokeWidth={1.75} />
            </ArrowButton>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

function FootCard({ number, title, body, onClick, className, motionProps }) {
  return (
    <motion.button
      onClick={onClick}
      className={`group relative flex min-h-60 cursor-pointer flex-col p-7 text-left xl:p-9 ${className}`}
      {...motionProps}
    >
      <span className="font-display text-lg font-bold">{number}</span>
      <span className="font-display text-lg leading-snug font-bold">{title}</span>
      <span className="mt-4 max-w-[26ch] text-[0.95rem] leading-relaxed opacity-85">{body}</span>
      <ArrowUpRight
        size={20}
        strokeWidth={2}
        className="absolute top-7 right-7 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
      />
    </motion.button>
  )
}

function ArrowButton({ label, onClick, children }) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="grid size-14 cursor-pointer place-items-center rounded-full border-2 border-(--color-ink)/85 text-(--color-ink) transition-colors duration-200 hover:bg-(--color-ink) hover:text-(--color-base)"
    >
      {children}
    </button>
  )
}
