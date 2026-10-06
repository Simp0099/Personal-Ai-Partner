import { useState, useCallback, useRef, useEffect } from 'react'
import {
  sendJarvisMessage,
  checkBackendHealth,
  clearJarvisConversation,
  getConversationId,
} from '../utils/api'

const MAX_MESSAGES = 100
const SPEAKING_RESET_DELAY = 2000
const HEALTH_CHECK_INTERVAL = 30000

/**
 * useJarvisState — Central state management hook for JARVIS 2.0.
 *
 * Manages chat history, connection status, listening/speaking states,
 * assistant visual state, and telemetry latency.
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
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [assistantState, setAssistantState] = useState('idle')
  const [latency, setLatency] = useState(0)
  const [isThinking, setIsThinking] = useState(false)

  // Refs for async safety and request ordering
  const mountedRef = useRef(true)
  const speakingTimeoutRef = useRef(null)
  const queueRef = useRef([])
  const drainingRef = useRef(false)
  const conversationIdRef = useRef(getConversationId())

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      if (speakingTimeoutRef.current) {
        clearTimeout(speakingTimeoutRef.current)
      }
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

    setAssistantState('speaking')
    setIsSpeaking(true)

    appendMessage({
      role: 'assistant',
      content: result.response,
      isError: result.error,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    })

    if (speakingTimeoutRef.current) {
      clearTimeout(speakingTimeoutRef.current)
    }
    speakingTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) {
        setIsSpeaking(false)
        setAssistantState('idle')
      }
    }, SPEAKING_RESET_DELAY)
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

    setAssistantState('listening')
    appendMessage({
      role: 'user',
      content: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    })

    queueRef.current.push(trimmed)
    drainQueue()
  }, [appendMessage, drainQueue])

  // --- Toggle Listening ---
  const toggleListening = useCallback(() => {
    setIsListening((prev) => {
      const next = !prev
      setAssistantState(next ? 'listening' : 'idle')
      return next
    })
  }, [])

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

  return {
    // State
    messages,
    isConnected,
    isListening,
    isSpeaking,
    assistantState,
    latency,
    isThinking,
    // Actions
    sendMessage,
    toggleListening,
    clearChat,
  }
}
