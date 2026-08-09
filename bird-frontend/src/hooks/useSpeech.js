import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Text-to-speech via the browser's Web Speech API.
 *
 * No dependency, no model download, no network call, and macOS ships genuinely
 * good voices. The one real constraint is that speech must begin inside a user
 * gesture, so every caller is triggered by a click or a key — nothing here
 * speaks on its own.
 *
 * The interesting part is `spokenUpto`: the utterance emits `boundary` events as
 * it moves through the text, which lets the UI underline the sentence being read
 * right now. That is what makes the narration feel like part of the page rather
 * than a detached audio track.
 */

const PREFERRED = [
  // macOS voices, best first. Samantha and Daniel are the two most natural
  // en-US/en-GB system voices; the rest are reasonable fallbacks elsewhere.
  'Samantha',
  'Daniel',
  'Karen',
  'Moira',
  'Google US English',
  'Microsoft Aria Online (Natural) - English (United States)',
]

function pickVoice(voices) {
  for (const name of PREFERRED) {
    const hit = voices.find((v) => v.name === name)
    if (hit) return hit
  }
  return voices.find((v) => v.lang?.startsWith('en') && v.localService) ||
    voices.find((v) => v.lang?.startsWith('en')) ||
    voices[0] || null
}

export function useSpeech() {
  const supported = typeof window !== 'undefined' && 'speechSynthesis' in window

  const [voices, setVoices] = useState([])
  const [voiceName, setVoiceName] = useState(() => localStorage.getItem('bird.voice') || '')
  const [rate, setRate] = useState(() => Number(localStorage.getItem('bird.rate')) || 1)
  const [muted, setMuted] = useState(() => localStorage.getItem('bird.muted') === '1')
  const [speaking, setSpeaking] = useState(false)
  const [spokenText, setSpokenText] = useState('')
  const [spokenUpto, setSpokenUpto] = useState(0)

  const utteranceRef = useRef(null)

  // Chrome populates the voice list asynchronously and fires voiceschanged.
  useEffect(() => {
    if (!supported) return
    const load = () => setVoices(window.speechSynthesis.getVoices())
    load()
    window.speechSynthesis.addEventListener('voiceschanged', load)
    return () => window.speechSynthesis.removeEventListener('voiceschanged', load)
  }, [supported])

  useEffect(() => {
    if (voiceName) localStorage.setItem('bird.voice', voiceName)
  }, [voiceName])
  useEffect(() => localStorage.setItem('bird.rate', String(rate)), [rate])
  useEffect(() => localStorage.setItem('bird.muted', muted ? '1' : '0'), [muted])

  // A page unload mid-utterance otherwise leaves the voice talking to itself.
  useEffect(() => {
    if (!supported) return
    const stop = () => window.speechSynthesis.cancel()
    window.addEventListener('beforeunload', stop)
    return () => {
      window.removeEventListener('beforeunload', stop)
      stop()
    }
  }, [supported])

  const cancel = useCallback(() => {
    if (!supported) return
    window.speechSynthesis.cancel()
    utteranceRef.current = null
    setSpeaking(false)
    setSpokenUpto(0)
  }, [supported])

  const speak = useCallback(
    (text) => {
      if (!supported || muted || !text?.trim()) return false

      // Always cancel first. Queueing utterances means a second click makes the
      // app talk over itself, and 'speak' here means 'say this now'.
      window.speechSynthesis.cancel()

      const utterance = new SpeechSynthesisUtterance(text)
      const chosen = voices.find((v) => v.name === voiceName) || pickVoice(voices)
      if (chosen) {
        utterance.voice = chosen
        utterance.lang = chosen.lang
      }
      utterance.rate = rate
      utterance.pitch = 1

      setSpokenText(text)
      setSpokenUpto(0)
      setSpeaking(true)

      utterance.onboundary = (event) => {
        if (typeof event.charIndex === 'number') setSpokenUpto(event.charIndex)
      }
      utterance.onend = () => {
        setSpeaking(false)
        setSpokenUpto(text.length)
        utteranceRef.current = null
      }
      utterance.onerror = () => {
        setSpeaking(false)
        utteranceRef.current = null
      }

      utteranceRef.current = utterance
      window.speechSynthesis.speak(utterance)
      return true
    },
    [supported, muted, voices, voiceName, rate],
  )

  const toggleMute = useCallback(() => {
    setMuted((m) => {
      if (!m) window.speechSynthesis?.cancel()
      return !m
    })
  }, [])

  return {
    supported,
    speak,
    cancel,
    speaking,
    muted,
    toggleMute,
    setMuted,
    rate,
    setRate,
    voices: voices.filter((v) => v.lang?.startsWith('en')),
    voiceName: voiceName || pickVoice(voices)?.name || '',
    setVoiceName,
    spokenText,
    spokenUpto,
  }
}

/**
 * Split text into sentences and mark which one the voice is currently in.
 *
 * Used by the narration panel to highlight along with the audio. Kept here
 * beside the hook because it depends on the same charIndex contract.
 */
export function sentencesWithActive(text, charIndex) {
  if (!text) return []
  const parts = text.match(/[^.!?]+[.!?]*\s*/g) || [text]
  let cursor = 0
  return parts.map((sentence) => {
    const start = cursor
    cursor += sentence.length
    return { text: sentence, active: charIndex >= start && charIndex < cursor }
  })
}
