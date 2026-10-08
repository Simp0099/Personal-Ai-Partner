import axios from 'axios'

/**
 * JARVIS 2.0 — API Utility
 *
 * Axios client configured for the local backend.
 * In dev mode, Vite proxies /api -> localhost:8000.
 * In production, set VITE_API_BASE_URL or use relative /api.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

/**
 * Stable per-tab conversation id.
 *
 * The backend keeps one conversation per id. Without this every browser tab
 * shared a single Gemini history, so one conversation's context leaked into
 * another. Stored in sessionStorage so it survives reloads within the tab but
 * not across tabs.
 */
export function getConversationId() {
  const KEY = 'jarvis_conversation_id'
  let id = sessionStorage.getItem(KEY)
  if (!id) {
    id =
      globalThis.crypto?.randomUUID?.() ??
      `conv-${Date.now()}-${Math.random().toString(36).slice(2)}`
    sessionStorage.setItem(KEY, id)
  }
  return id
}

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    config.metadata = { startTime: performance.now() }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor for latency tracking
apiClient.interceptors.response.use(
  (response) => {
    const latency = Math.round(performance.now() - response.config.metadata.startTime)
    response.data.latency = latency
    return response
  },
  (error) => {
    if (error.config?.metadata) {
      error.latency = Math.round(performance.now() - error.config.metadata.startTime)
    }
    return Promise.reject(error)
  }
)

/**
 * Send a message to the JARVIS backend.
 *
 * The message is sent exactly as given, tagged with this tab's conversation id.
 *
 * @param {string} message
 * @param {string} [conversationId]
 * @returns {Promise<{response: string, latency: number, error: boolean}>}
 */
export async function sendJarvisMessage(message, conversationId = getConversationId()) {
  try {
    const { data } = await apiClient.post('/chat', {
      message,
      conversation_id: conversationId,
    })
    return {
      response: data.response || data.text || 'Standing by, Boss.',
      latency: data.latency || 0,
      error: Boolean(data.error),
    }
  } catch (err) {
    const isTimeout = err.code === 'ECONNABORTED'
    const isNetworkError = !err.response

    // Surface the server's real reason instead of inventing an in-persona
    // excuse, so transport failures are not mistaken for model replies.
    const serverDetail = err.response?.data?.detail
    const fallbackMessage = serverDetail
      ? String(serverDetail)
      : isTimeout
      ? 'The backend took too long to respond.'
      : isNetworkError
      ? "Cannot reach the JARVIS backend. Is the server running?"
      : err.message || 'Request failed.'

    console.error('[API Error]', err.message)

    return {
      response: fallbackMessage,
      latency: err.latency || 0,
      error: true,
    }
  }
}

/**
 * Clear a conversation on the backend.
 *
 * Without this the UI could show a "clean slate" while the backend kept the
 * full conversation and continued answering from it.
 *
 * @param {string} [conversationId]
 */
export async function clearJarvisConversation(conversationId = getConversationId()) {
  try {
    await apiClient.post('/clear', { conversation_id: conversationId })
    return true
  } catch (err) {
    console.error('[API Error] clear failed', err.message)
    return false
  }
}

/**
 * Fetch the authoritative conversational state.
 *
 * Phase 4: the backend owns the state machine, so the HUD renders this rather
 * than inferring state from whether a reply arrived. Guessing is what made the
 * UI look like it was speaking while the assistant was still thinking.
 *
 * @returns {Promise<Object>} state snapshot; `offline: true` if unreachable.
 */
export async function getConversationState() {
  try {
    const { data } = await apiClient.get('/state', { timeout: 5000 })
    return { ...data, offline: false }
  } catch (err) {
    return { state: 'idle', turn_id: null, offline: true }
  }
}

/**
 * Interrupt the assistant mid-sentence (barge-in).
 *
 * Stops playback, cancels the in-flight turn and clears queued audio. Returns
 * the state the assistant settled into.
 */
export async function interruptJarvis() {
  try {
    const { data } = await apiClient.post('/voice/interrupt', {}, { timeout: 5000 })
    return data
  } catch (err) {
    console.error('[API Error] interrupt failed', err.message)
    return { interrupted: false }
  }
}

/**
 * Enable or disable voice input.
 *
 * Turning it off releases the microphone. Text conversation is unaffected
 * either way.
 */
export async function setJarvisVoice(enabled) {
  try {
    const { data } = await apiClient.post('/voice/start', { enabled }, { timeout: 15000 })
    return data
  } catch (err) {
    console.error('[API Error] voice control failed', err.message)
    return { status: 'unavailable', error: true }
  }
}

/**
 * Check backend health status.
 *
 * @returns {Promise<{online: boolean, latency: number}>}
 */
export async function checkBackendHealth() {
  try {
    const { data } = await apiClient.get('/health', { timeout: 5000 })
    return {
      online: true,
      latency: data.latency || 0,
    }
  } catch (err) {
    return {
      online: false,
      latency: err.latency || 0,
    }
  }
}

export default apiClient
