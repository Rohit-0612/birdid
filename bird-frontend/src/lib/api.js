/**
 * The whole HTTP surface, in one place.
 *
 * Every path is relative: Vite proxies /api to 127.0.0.1:8000 in dev, and on
 * Vercel a rewrite in vercel.json forwards /api to the Hugging Face Space. No
 * base URL to configure, no CORS.
 */

async function json(res) {
  if (!res.ok) {
    // FastAPI puts the useful message in `detail`; surface it rather than a
    // bare status code, which is what the user would otherwise see in a toast.
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const health = () => fetch('/api/health').then(json)

/**
 * Identify a photo, and by default file it straight into the deck.
 *
 * One request does identification, open-vocabulary verification (when the
 * confidence or the open-set gate warrants it) and deck registration, because the
 * server already has the image — splitting it up would mean uploading twice.
 */
export function identify(file, { register = true, verify = true } = {}) {
  const body = new FormData()
  body.append('image', file)
  body.append('add_to_deck', String(register))
  body.append('verify', String(verify))
  return fetch('/api/identify', { method: 'POST', body }).then(json)
}

export function identifyAudio(file, { minConf = 0.25, useLocation = false, lat = 39.83, lon = -98.58, obsDate = '' } = {}) {
  const body = new FormData()
  body.append('audio', file)
  body.append('min_conf', String(minConf))
  body.append('use_location', String(useLocation))
  body.append('lat', String(lat))
  body.append('lon', String(lon))
  body.append('obs_date', obsDate)
  return fetch('/api/identify/audio', { method: 'POST', body }).then(json)
}

export function gradcam(file) {
  const body = new FormData()
  body.append('image', file)
  return fetch('/api/gradcam', { method: 'POST', body }).then(json)
}

export const speciesList = () => fetch('/api/species').then(json)
export const species = (folder) => fetch(`/api/species/${encodeURIComponent(folder)}`).then(json)
export const compare = (a, b) =>
  fetch(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`).then(json)
export const samples = () => fetch('/api/samples').then(json)

export const narrate = (result) =>
  fetch('/api/narrate', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ result }),
  }).then(json)

export const chat = (question, { folder = null, subject = null, history = [] } = {}) =>
  fetch('/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question, folder, subject, history }),
  }).then(json)

/**
 * Streaming chat over SSE.
 *
 * Uses fetch + a ReadableStream rather than EventSource because EventSource
 * cannot POST, and the question, the on-screen species and the history all need
 * to go in a body. `onEvent` receives each parsed event as it arrives.
 */
export async function chatStream(question, { folder = null, subject = null, history = [], signal } = {}, onEvent) {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question, folder, subject, history }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`stream failed: ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by a blank line. Anything after the last
    // separator is a partial frame and waits for the next chunk.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue
      try {
        onEvent(JSON.parse(line.slice(6)))
      } catch {
        /* ignore a malformed frame rather than killing the stream */
      }
    }
  }
}

export const sightings = (limit = 200) => fetch(`/api/sightings?limit=${limit}`).then(json)
export const sightingStats = () => fetch('/api/sightings/stats').then(json)
export const deleteSighting = (id) =>
  fetch(`/api/sightings/${encodeURIComponent(id)}`, { method: 'DELETE' }).then(json)

export function saveSighting(result, { notes = '', file = null } = {}) {
  const body = new FormData()
  body.append('result', JSON.stringify(result))
  body.append('notes', notes)
  if (file) body.append('image', file)
  return fetch('/api/sightings', { method: 'POST', body }).then(json)
}

export const thumbUrl = (id) => `/api/sightings/${encodeURIComponent(id)}/thumb`

// ── The deck ──────────────────────────────────────────────
export const deck = () => fetch('/api/deck').then(json)
export const deckStats = () => fetch('/api/deck/stats').then(json)
export const deleteCard = (key) =>
  fetch(`/api/deck/${encodeURIComponent(key)}`, { method: 'DELETE' }).then(json)

/** Card photo. Keys contain a colon ('ext:Ara macao'), so they must be encoded. */
export const cardThumbUrl = (key) => `/api/deck/${encodeURIComponent(key)}/thumb`

/** Force a card in when neither model was confident enough to file it. */
export function registerCard(result, file = null) {
  const body = new FormData()
  body.append('result', JSON.stringify(result))
  body.append('force', 'true')
  if (file) body.append('image', file)
  return fetch('/api/deck/register', { method: 'POST', body }).then(json)
}
