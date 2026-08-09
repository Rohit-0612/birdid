import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Push-to-talk voice commands via the Web Speech Recognition API.
 *
 * Push-to-talk, not always-listening, and that is a deliberate product choice:
 * a permanently open microphone is a privacy cost the user did not agree to, it
 * fires on television dialogue, and Chrome's continuous mode restarts itself in
 * ways that are hard to reason about. Hold the button (or the spacebar), speak,
 * release.
 *
 * Recognised phrasings map to intents. Anything unmatched is not an error — it
 * becomes a question for the grounded chat endpoint, which is the behaviour you
 * want: "how big is it" should be answered, not rejected.
 */

const INTENTS = [
  { intent: 'identify', patterns: [/\bidentify\b/, /what (bird|species) is (this|it)/, /who is this/, /^what is this/] },
  { intent: 'narrate', patterns: [/tell me (more|about)/, /read (it|this) (out|aloud)/, /^describe/, /read it/] },
  { intent: 'fieldMarks', patterns: [/field marks?/, /how do i (spot|recognise|recognize) it/, /what does it look like/] },
  { intent: 'save', patterns: [/save (this|it)/, /add (this|it)? ?to (my )?(life )?list/, /log (this|it)/] },
  { intent: 'stop', patterns: [/^(stop|quiet|silence|shut up|be quiet)\b/, /stop (talking|reading|speaking)/] },
  { intent: 'compare', patterns: [/compare (it |this )?(with|to) (?<target>.+)/, /difference between .+ and (?<target>.+)/] },
  { intent: 'listen', patterns: [/^(listen|audio|identify by sound)/, /identify (the )?(call|song|sound)/] },
  { intent: 'lifeList', patterns: [/(open|show|go to) (my )?life list/, /what have i seen/] },
  { intent: 'fieldGuide', patterns: [/(open|show|go to) (the )?field guide/, /browse species/] },
]

export function parseIntent(transcript) {
  const said = transcript.toLowerCase().trim()
  for (const { intent, patterns } of INTENTS) {
    for (const pattern of patterns) {
      const match = said.match(pattern)
      if (match) {
        return { intent, target: match.groups?.target?.trim() || null, transcript: said }
      }
    }
  }
  // Not a command — treat it as a question.
  return { intent: 'ask', target: null, transcript: said }
}

export function useVoiceCommands(onCommand) {
  const SpeechRecognition =
    typeof window !== 'undefined' &&
    (window.SpeechRecognition || window.webkitSpeechRecognition)
  const supported = Boolean(SpeechRecognition)

  const [listening, setListening] = useState(false)
  const [interim, setInterim] = useState('')
  const [error, setError] = useState(null)

  const recognitionRef = useRef(null)
  // Held in a ref so starting/stopping never depends on a fresh callback and
  // the recognition object does not need rebuilding on every render.
  const handlerRef = useRef(onCommand)
  useEffect(() => {
    handlerRef.current = onCommand
  }, [onCommand])

  useEffect(() => {
    if (!supported) return

    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.continuous = false
    recognition.interimResults = true
    recognition.maxAlternatives = 1

    recognition.onresult = (event) => {
      let finalText = ''
      let partial = ''
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i]
        if (result.isFinal) finalText += result[0].transcript
        else partial += result[0].transcript
      }
      setInterim(partial || finalText)
      if (finalText.trim()) {
        handlerRef.current?.(parseIntent(finalText))
      }
    }

    recognition.onerror = (event) => {
      // 'aborted' and 'no-speech' are normal outcomes of releasing the button
      // without saying anything; they are not worth showing to the user.
      if (event.error !== 'aborted' && event.error !== 'no-speech') {
        setError(
          event.error === 'not-allowed'
            ? 'Microphone permission denied. Allow it in your browser settings.'
            : `Speech recognition error: ${event.error}`,
        )
      }
      setListening(false)
    }

    recognition.onend = () => {
      setListening(false)
      setInterim('')
    }

    recognitionRef.current = recognition
    return () => {
      recognition.onresult = null
      recognition.onerror = null
      recognition.onend = null
      try {
        recognition.abort()
      } catch {
        /* already stopped */
      }
    }
  }, [supported, SpeechRecognition])

  const start = useCallback(() => {
    if (!supported || listening) return
    setError(null)
    setInterim('')
    try {
      recognitionRef.current?.start()
      setListening(true)
    } catch {
      /* start() throws if it is already running — harmless */
    }
  }, [supported, listening])

  const stop = useCallback(() => {
    try {
      recognitionRef.current?.stop()
    } catch {
      /* not running */
    }
  }, [])

  // Hold space to talk. Ignored while typing, so the chat box still works.
  useEffect(() => {
    if (!supported) return
    const isTyping = (el) =>
      el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)

    const down = (e) => {
      if (e.code !== 'Space' || e.repeat || isTyping(document.activeElement)) return
      e.preventDefault()
      start()
    }
    const up = (e) => {
      if (e.code !== 'Space' || isTyping(document.activeElement)) return
      e.preventDefault()
      stop()
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [supported, start, stop])

  return { supported, listening, interim, error, start, stop, clearError: () => setError(null) }
}
