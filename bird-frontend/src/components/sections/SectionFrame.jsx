/**
 * One chapter of the page: a big title and a plain-language lede on the left,
 * a photograph on the right, and the working tool full-width underneath.
 *
 * The frame is layout only. Whatever view renders inside it is untouched — it
 * receives exactly the props it did when it lived behind a tab.
 */
export function SectionFrame({ id, title, lede, photo, aside, children }) {
  const headingId = `${id}-title`
  return (
    <section id={id} aria-labelledby={headingId} className="border-b border-(--color-line)">
      <div className="grid gap-10 px-6 pt-20 pb-12 sm:px-10 lg:grid-cols-5 lg:gap-0 lg:px-0 lg:pt-28 lg:pb-16">
        <header className="lg:col-span-3 lg:px-10 xl:px-14">
          <h2
            id={headingId}
            className="font-display text-display font-extrabold text-balance text-(--color-ink)"
          >
            {title}
          </h2>
          <p className="mt-6 max-w-[52ch] text-lede text-(--color-ink-soft)">{lede}</p>
        </header>

        {aside && <div className="lg:col-span-2 lg:px-8 xl:px-10">{aside}</div>}

        {photo && (
          <figure className="lg:col-span-2 lg:px-8 xl:px-10">
            <div className="overflow-hidden rounded-(--radius-photo) bg-(--color-mat) shadow-[var(--shadow-lift)]">
              <img
                src={photo.thumb}
                alt={photo.alt}
                loading="lazy"
                decoding="async"
                className="aspect-[16/10] w-full object-cover"
                style={{ objectPosition: photo.position }}
              />
            </div>
            <figcaption className="mt-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-caption text-(--color-ink-faint)">
              <span>
                <span className="font-semibold text-(--color-ink-soft)">{photo.name}</span>{' '}
                <span className="italic">{photo.sci}</span>
              </span>
              <a
                href={photo.source}
                target="_blank"
                rel="noreferrer"
                className="text-micro underline-offset-2 hover:text-(--color-ink-soft) hover:underline"
              >
                {photo.author}, {photo.license}
              </a>
            </figcaption>
          </figure>
        )}
      </div>

      <div className="mx-auto max-w-[1680px] px-4 pb-24 sm:px-10 lg:px-10 lg:pb-32 xl:px-14">{children}</div>
    </section>
  )
}
