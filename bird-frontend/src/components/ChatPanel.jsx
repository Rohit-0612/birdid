import { useEffect, useRef, useState } from 'react'
import { Send, Sparkles, Volume2, X } from 'lucide-react'

import { Chip } from './primitives'
import * as api from '../lib/api'

/**
 * Ask-the-field-guide chat, grounded in the species knowledge base.
 *
 * Streams tokens so the first words appear in about a second rather than after
 * the whole paragraph. Answers are spoken automatically when they arrived by
 * voice, and only then: a typed question gets a typed answer, because having the
 * app start talking because you used the keyboard is startling.
 */
export function ChatPanel({ folder, subject, speech, pendingQuestion, onConsumePending, onClose }) {
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [grounding, setGrounding] = useState([])
  const scrollRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [turns, streaming])

  // A question arriving from the voice layer is asked immediately and spoken back.
  useEffect(() => {
    if (pendingQuestion) {
      onConsumePending?.()
      ask(pendingQuestion, { speakAnswer: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingQuestion])

  useEffect(() => () => abortRef.current?.abort(), [])

  async function ask(question, { speakAnswer = false } = {}) {
    const text = question.trim()
    if (!text || streaming) return

    setDraft('')
    setTurns((prev) => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    setStreaming(true)
    setGrounding([])

    const history = turns.slice(-4)
    const controller = new AbortController()
    abortRef.current = controller
    let answer = ''

    const append = (chunk) => {
      answer += chunk
      setTurns((prev) => {
        const next = [...prev]
        next[next.length - 1] = { role: 'assistant', content: answer }
        return next
      })
    }

    try {
      await api.chatStream(text, { folder, subject, history, signal: controller.signal }, (event) => {
        if (event.type === 'grounding') setGrounding(event.folders ?? [])
        else if (event.type === 'token') append(event.text)
        else if (event.type === 'error') append(event.text)
      })
    } catch (err) {
      if (err.name !== 'AbortError') {
        // Streaming failed outright — fall back to the non-streaming endpoint so
        // the question still gets an answer.
        try {
          const payload = await api.chat(text, { folder, subject, history })
          append(payload.text)
        } catch (fallbackErr) {
          append(`I could not reach the assistant. ${fallbackErr.message}`)
        }
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
      if (speakAnswer && answer.trim()) speech?.speak(answer)
    }
  }

  return (
    <section className="card flex max-h-[560px] flex-col overflow-hidden">
      <header className="flex items-center justify-between gap-3 border-b border-(--color-line) px-5 py-3.5">
        <div className="flex items-center gap-2">
          <Sparkles size={15} strokeWidth={2} className="text-(--color-accent)" />
          <h3 className="text-sm font-semibold text-(--color-ink)">Ask the field guide</h3>
        </div>
        <div className="flex items-center gap-2">
          {grounding.length > 0 && (
            <Chip tone="accent" title={`Answered using: ${grounding.join(', ')}`}>
              {grounding.length} record{grounding.length > 1 ? 's' : ''}
            </Chip>
          )}
          {onClose && (
            <button
              onClick={onClose}
              aria-label="Close chat"
              className="cursor-pointer rounded p-1 text-(--color-ink-faint) transition-colors duration-200 hover:text-(--color-ink)"
            >
              <X size={15} strokeWidth={2} />
            </button>
          )}
        </div>
      </header>

      <div ref={scrollRef} className="min-h-32 flex-1 space-y-3 overflow-y-auto px-5 py-4">
        {turns.length === 0 && (
          <div className="space-y-3 py-2">
            <p className="text-sm text-(--color-ink-faint)">
              Grounded in the 200-species knowledge base. Ask about the bird on screen, or any
              species in the guide.
            </p>
            <div className="flex flex-wrap gap-2">
              {[
                'How do I tell it from similar species?',
                'Is it migratory?',
                'What does it eat?',
                'Where would I find one?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => ask(suggestion)}
                  className="cursor-pointer rounded-full border border-(--color-line) px-3 py-1.5 text-xs text-(--color-ink-soft) transition-colors duration-200 hover:border-(--color-accent)/50 hover:text-(--color-ink)"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, i) => (
          <div
            key={i}
            className={turn.role === 'user' ? 'flex justify-end' : 'flex justify-start'}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                turn.role === 'user'
                  ? 'bg-(--color-accent)/20 text-(--color-ink)'
                  : 'border border-(--color-line) bg-(--color-raised) text-(--color-ink-soft)'
              }`}
            >
              {turn.content || (
                <span className="inline-flex gap-1" aria-label="Thinking">
                  {[0, 1, 2].map((d) => (
                    <span
                      key={d}
                      className="size-1.5 animate-bounce rounded-full bg-(--color-accent)"
                      style={{ animationDelay: `${d * 0.15}s` }}
                    />
                  ))}
                </span>
              )}
              {turn.role === 'assistant' && turn.content && !streaming && speech?.supported && (
                <button
                  onClick={() => speech.speak(turn.content)}
                  className="mt-2 flex cursor-pointer items-center gap-1.5 text-xs text-(--color-ink-faint) transition-colors duration-200 hover:text-(--color-accent-hover)"
                >
                  <Volume2 size={12} strokeWidth={2} />
                  Read aloud
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          ask(draft)
        }}
        className="flex gap-2 border-t border-(--color-line) px-5 py-3.5"
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask about this bird…"
          className="min-w-0 flex-1 rounded-lg border border-(--color-line) bg-(--color-raised) px-3 py-2 text-sm text-(--color-ink) placeholder:text-(--color-ink-faint) focus:border-(--color-accent)/50 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!draft.trim() || streaming}
          aria-label="Send question"
          className="inline-flex cursor-pointer items-center rounded-lg bg-(--color-accent) px-3 py-2 text-(--color-on-accent) transition-colors duration-200 hover:bg-(--color-accent-hover) disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Send size={15} strokeWidth={2} />
        </button>
      </form>
    </section>
  )
}
