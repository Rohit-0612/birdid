/**
 * The one answer to show for an identification.
 *
 * The API resolves it (`result.answer`) with the same rule the deck uses to file
 * a card: the 200-species classifier when it is trusted, otherwise the
 * open-vocabulary verifier when it is confident, otherwise "unsure". The rest of
 * the pipeline stays behind the scenes.
 *
 * A backend deployed before `answer` existed does not send it, so the same
 * decision is derived here from the classifier and verifier fields. That lets
 * the site and the API be redeployed in either order.
 */
const TRUST_CONFIDENCE = 0.6

export function answerOf(result) {
  if (!result) return null
  if (result.answer) return result.answer

  const species = result.species || {}
  const gateOk = result.openset?.is_bird ?? true
  if (gateOk && (result.confidence ?? 0) >= TRUST_CONFIDENCE && species.folder) {
    return {
      status: 'identified',
      identified_by: 'cub',
      display_name: species.display_name,
      scientific_name: species.scientific_name,
      family: species.family,
      order: species.order,
      folder: species.folder,
      source: 'cub',
      confidence: result.confidence,
      band: result.confidence_band,
      info: result.info || {},
    }
  }

  const v = result.verification
  if (v?.ran && v.confident && v.best) {
    return {
      status: 'identified',
      identified_by: 'verifier',
      display_name: v.best.common_name,
      scientific_name: v.best.scientific_name,
      family: null,
      order: null,
      folder: v.cub_folder ?? null,
      source: v.cub_folder ? 'cub' : 'external',
      confidence: null,
      band: 'confirmed',
      info: {},
    }
  }

  return {
    status: 'unsure',
    identified_by: null,
    display_name: null,
    best_guess: species.display_name ?? null,
    folder: null,
    info: {},
  }
}

/** "Barn Owl (Tyto alba)" — how the chat is told which bird is on screen. */
export function answerSubject(answer) {
  if (!answer?.display_name) return null
  return answer.scientific_name
    ? `${answer.display_name} (${answer.scientific_name})`
    : answer.display_name
}
