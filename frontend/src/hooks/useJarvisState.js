import { useState, useCallback, useRef, useEffect } from 'react'
import {
  sendJarvisMessage,
  checkBackendHealth,
  clearJarvisConversation,
  getConversationId,
  getConversationState,
  interruptJarvis,
  setJarvisVoice,
} from '../utils/api'

const MAX_MESSAGES = 100
const HEALTH_CHECK_INTERVAL = 30000
// Fast enough that the HUD reflects a state change as it happens, slow enough
// that an idle tab is not polling the backend every frame.
const STATE_POLL_INTERVAL = 400

/**
 * useJarvisState — Central state management hook for JARVIS 2.0.
 *
 * Manages chat history, connection status, and the assistant's conversational
 * state.
 *
 * Phase 0 reliability fixes:
 * - Messages sent while a request is in flight are QUEUED and drained in order.
 *   Previously `if (thinkingRef.current) return` silently discarded them, so
 *   rapid consecutive messages never reached the backend.
 * - The input stays enabled while thinking, so nothing is blocked at the UI.
 * - `clearChat` now also clears the conversation on the backend, so a "clean
 *   slate" actually produces a clean conversation.
 * - Every message is tagged with this tab's conversation id, so conversations
 *   cannot share backend history.
 *
 * Phase 4:
 * - `assistantState` comes from the backend state machine (GET /api/state), not
 *   from a local guess. The previous version set "speaking" the moment a reply
 *   arrived and back to "idle" on a 2s timer, which meant the HUD claimed the
 *   assistant was talking while it was still generating, and kept claiming it
 *   after a barge-in had cut it off.
 * - Voice can be switched on and off, and the user can interrupt mid-sentence.
 */
export function useJarvisState() {
  // --- Core State ---
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hey Boss. Kyuoko Hori online. What do you need?",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ])
  const [isConnected, setIsConnected] = useState(false)
  const [latency, setLatency] = useState(0)
  const [isThinking, setIsThinking] = useState(false)

  // Backend-authoritative conversational state.
  const [backendState, setBackendState] = useState({
    state: 'idle',
    turn_id: null,
    transcript: '',
    voice: { running: false, wake_enabled: false },
    webcam_running: false,
    visual_context: { observations: [], fresh: false },
    last_error: null,
  })

  // Refs for async safety and request ordering
  const mountedRef = useRef(true)
  const queueRef = useRef([])
  const drainingRef = useRef(false)
  const conversationIdRef = useRef(getConversationId())

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  // --- Health Check on Mount + Interval ---
  useEffect(() => {
    let intervalId = null

    const checkHealth = async () => {
      const result = await checkBackendHealth()
      if (mountedRef.current) {
        setIsConnected(result.online)
        setLatency(result.latency)
      }
    }

    checkHealth()
    intervalId = setInterval(checkHealth, HEALTH_CHECK_INTERVAL)

    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [])

  // --- Conversational state polling ---
  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      const snapshot = await getConversationState()
      if (!cancelled && mountedRef.current) {
        setBackendState(snapshot)
      }
    }

    poll()
    const id = setInterval(poll, STATE_POLL_INTERVAL)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const appendMessage = useCallback((msg) => {
    setMessages((prev) => {
      const next = [...prev, msg]
      // Cap history to prevent memory bloat
      return next.length > MAX_MESSAGES ? next.slice(-MAX_MESSAGES) : next
    })
  }, [])

  /**
   * Send one message to the backend and append the reply.
   * Never throws; failures are appended as an error-flagged reply.
   */
  const dispatchMessage = useCallback(async (trimmed) => {
    const result = await sendJarvisMessage(trimmed, conversationIdRef.current)
    if (!mountedRef.current) return

    setLatency(result.latency)
    if (result.error) {
      // A transport/backend failure is not the assistant speaking. Mark it so
      // the UI can distinguish it from a real reply.
      setIsConnected(false)
    }

    appendMessage({
      role: 'assistant',
      content: result.response,
      isError: result.error,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    })
  }, [appendMessage])

  /**
   * Drain the queue strictly in order. Only one request is ever in flight, so
   * messages cannot be reordered on the backend.
   */
  const drainQueue = useCallback(async () => {
    if (drainingRef.current) return
    drainingRef.current = true
    setIsThinking(true)

    try {
      while (queueRef.current.length > 0) {
        const next = queueRef.current.shift()
        await dispatchMessage(next)
      }
    } finally {
      drainingRef.current = false
      if (mountedRef.current) setIsThinking(false)
    }
  }, [dispatchMessage])

  // --- Send Message (queued, ordered, never dropped) ---
  const sendMessage = useCallback((text) => {
    const trimmed = (text ?? '').trim()
    // Blank input is ignored client-side; the backend rejects it too.
    if (!trimmed) return

    appendMessage({
      role: 'user',
      content: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    })

    queueRef.current.push(trimmed)
    drainQueue()
  }, [appendMessage, drainQueue])

  // --- Interrupt (barge-in) ---
  const interrupt = useCallback(async () => {
    const result = await interruptJarvis()
    // The state poll will pick up the new state; nothing to set locally, so the
    // UI can never claim the assistant is still speaking after it was cut off.
    return result
  }, [])

  // --- Toggle voice input ---
  const toggleVoice = useCallback(async () => {
    const next = !backendState.voice?.running
    return setJarvisVoice(next)
  }, [backendState.voice?.running])

  // --- Clear Chat ---
  // Clears the backend conversation too. Previously only the local message list
  // was reset, so the backend kept answering from the old conversation.
  const clearChat = useCallback(async () => {
    queueRef.current = []
    await clearJarvisConversation(conversationIdRef.current)
    if (!mountedRef.current) return
    setMessages([
      {
        role: 'assistant',
        content: "Clean slate, Boss. What's next?",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ])
  }, [])

  const state = backendState.state
  const isListening = state === 'listening' || state === 'follow_up'
  const isSpeaking = state === 'speaking'
  const isVoiceActive = Boolean(backendState.voice?.running)

  return {
    // State
    messages,
    isConnected,
    isListening,
    isSpeaking,
    isThinking,
    isVoiceActive,
    assistantState: state,
    backendState,
    latency,
    // Actions
    sendMessage,
    interrupt,
    toggleVoice,
    clearChat,
  }
}