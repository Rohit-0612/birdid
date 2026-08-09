import { useCallback, useRef, useState } from 'react'
import { ImagePlus, RotateCcw, Sparkle } from 'lucide-react'

/**
 * Image dropzone with preview.
 *
 * Drag-and-drop plus click plus paste. Paste matters more than it sounds: the
 * fastest way to test a bird photo is to copy it from a browser tab, and
 * without a paste handler you have to save it to disk first.
 */
export function Dropzone({ onFile, preview, onClear, busy, gradcamUrl, showGradcam, onToggleGradcam }) {
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const accept = useCallback(
    (file) => {
      if (!file) return
      if (!file.type.startsWith('image/')) {
        setError('That is not an image file.')
        return
      }
      if (file.size > 25 * 1024 * 1024) {
        setError('That image is larger than 25 MB.')
        return
      }
      setError(null)
      onFile(file)
    },
    [onFile],
  )

  return (
    <div
      onPaste={(e) => {
        const file = Array.from(e.clipboardData?.files ?? [])[0]
        if (file) accept(file)
      }}
      tabIndex={-1}
    >
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          accept(e.dataTransfer.files?.[0])
        }}
        className={`card relative overflow-hidden transition-all duration-200 ${
          dragging
            ? 'scale-[1.006] border-dashed border-(--color-accent) bg-(--color-accent)/10 shadow-[var(--shadow-lift)]'
            : ''
        }`}
      >
        {preview ? (
          // A photograph is mounted on a dark mat — print field guides do this so a
          // bright page cannot wash out the plate, and it is what keeps a light
          // interface viable for an app that is mostly photographs. Grad-CAM is the
          // exception: it is a matplotlib figure on white, so a dark mount would
          // frame it as a white sheet on black. It gets a light mount instead.
          <div
            className={`relative m-2 overflow-hidden rounded-[calc(var(--radius-card)-0.35rem)] ${
              showGradcam && gradcamUrl ? 'bg-(--color-raised)' : 'bg-(--color-mat)'
            }`}
          >
            <img
              src={showGradcam && gradcamUrl ? gradcamUrl : preview}
              alt={showGradcam ? 'Grad-CAM attention overlay' : 'Uploaded bird photo'}
              className="max-h-[380px] w-full object-contain"
            />
            {busy && (
              <div className="absolute inset-0 grid place-items-center bg-(--color-surface)/65 backdrop-blur-sm">
                <span className="size-8 animate-spin rounded-full border-2 border-(--color-line) border-t-(--color-accent)" />
              </div>
            )}
            <div className="absolute right-3 bottom-3 flex gap-2">
              {gradcamUrl && (
                <button
                  onClick={onToggleGradcam}
                  className={`inline-flex cursor-pointer items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs backdrop-blur transition-colors duration-200 ${
                    showGradcam
                      ? 'border-(--color-accent) bg-(--color-accent)/25 text-(--color-accent-hover)'
                      : 'border-(--color-line) bg-(--color-surface)/85 text-(--color-ink-soft) hover:text-(--color-ink)'
                  }`}
                  title="Show which pixels drove the prediction"
                >
                  <Sparkle size={13} strokeWidth={2} />
                  {showGradcam ? 'Heatmap on' : 'Heatmap'}
                </button>
              )}
              <button
                onClick={onClear}
                className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-(--color-line) bg-(--color-surface)/85 px-2.5 py-1.5 text-xs text-(--color-ink-soft) backdrop-blur transition-colors duration-200 hover:text-(--color-ink)"
              >
                <RotateCcw size={13} strokeWidth={2} />
                New photo
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => inputRef.current?.click()}
            className="group flex w-full cursor-pointer flex-col items-center gap-3 px-6 py-14 text-center transition-colors duration-200 hover:bg-(--color-accent)/5"
          >
            <span className="grid size-14 place-items-center rounded-full border border-(--color-line) text-(--color-accent) transition-all duration-300 group-hover:-translate-y-0.5 group-hover:border-(--color-accent)/50 group-hover:bg-(--color-accent)/8">
              <ImagePlus size={22} strokeWidth={1.5} />
            </span>
            <span className="font-display text-lg text-(--color-ink)">Drop a bird photo</span>
            <span className="text-sm text-(--color-ink-faint)">
              or click to browse, or paste from your clipboard
            </span>
          </button>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => accept(e.target.files?.[0])}
        />
      </div>
      {error && <p className="mt-2 text-xs text-(--color-low)">{error}</p>}
    </div>
  )
}
