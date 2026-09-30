import { PHOTOS } from '../../lib/photos'
import { scrollToSection } from '../../hooks/useScrollSpy'

/**
 * Colophon: what built the answers, and who took the photographs.
 * Every photo is credited with its licence, as those licences require.
 */
export function Footer({ sections }) {
  const photos = Object.values(PHOTOS)
  return (
    <footer className="scope-night bg-(--color-base)">
      <div className="grid gap-12 px-6 pt-20 pb-10 sm:px-10 lg:grid-cols-5 lg:gap-0 lg:px-0 lg:pt-28">
        <div className="lg:col-span-3 lg:px-10 xl:px-14">
          <p className="font-display text-hero font-extrabold tracking-[-0.03em] text-(--color-ink)">
            BirdID<span className="text-(--color-accent)">.</span>
          </p>
          <p className="mt-6 max-w-[56ch] text-body text-(--color-ink-soft)">
            EfficientNetV2-S trained on CUB-200-2011, BioCLIP for the birds beyond it, calls by
            BirdNET, and notes written by a language model — a local one when it is running,
            otherwise Groq — that are not expert-verified.
          </p>
          <nav aria-label="Sections" className="mt-10 flex flex-wrap gap-x-7 gap-y-3">
            {sections.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => scrollToSection(id)}
                className="cursor-pointer text-sm font-semibold text-(--color-ink-soft) transition-colors duration-200 hover:text-(--color-accent-hover)"
              >
                {label}
              </button>
            ))}
          </nav>
        </div>

        <div className="lg:col-span-2 lg:px-8 xl:px-10">
          <p className="text-sm font-semibold text-(--color-ink)">Photographs</p>
          <ul className="mt-4 space-y-2.5 text-caption text-(--color-ink-faint)">
            {photos.map((p) => (
              <li key={p.src} className="flex flex-wrap justify-between gap-x-4">
                <span className="text-(--color-ink-soft)">{p.name}</span>
                <a
                  href={p.source}
                  target="_blank"
                  rel="noreferrer"
                  className="underline-offset-2 hover:text-(--color-ink-soft) hover:underline"
                >
                  {p.author} · {p.license}
                </a>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-micro text-(--color-ink-faint)">
            From Wikimedia Commons, resized for the web. Licence terms at each source page.
          </p>
        </div>
      </div>
    </footer>
  )
}
