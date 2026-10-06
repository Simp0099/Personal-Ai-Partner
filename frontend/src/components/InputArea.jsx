import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Mic, MicOff } from 'lucide-react'

/**
 * InputArea — Sleek input bar with ripple feedback and glow states.
 *
 * Features:
 * - Ripple effect on button click
 * - Focus glow ring transitioning to accent color
 * - Mic toggle with active state glow
 * - Smooth hover elevation on send button
 */
export default function InputArea({ onSend, disabled = false }) {
  const [input, setInput] = useState('')
  const [isFocused, setIsFocused] = useState(false)
  const [micActive, setMicActive] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSubmit = () => {
    const trimmed = input.trim()
    if (!trimmed || disabled) return

    onSend(trimmed)
    setInput('')
    inputRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleMicToggle = () => {
    setMicActive((prev) => !prev)
  }

  return (
    <div style={styles.container}>
      {/* Mic button with ripple */}
      <motion.button
        onClick={handleMicToggle}
        disabled={disabled}
        className="ripple"
        style={{
          ...styles.micButton,
          ...(micActive ? styles.micButtonActive : {}),
          opacity: disabled ? 0.4 : 1,
        }}
        whileHover={{ scale: disabled ? 1 : 1.08, y: disabled ? 0 : -2 }}
        whileTap={{ scale: disabled ? 1 : 0.92 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        title={micActive ? 'Stop listening' : 'Start voice input'}
      >
        {micActive ? (
          <Mic size={18} style={{ color: 'var(--color-accent)' }} />
        ) : (
          <MicOff size={18} style={{ color: 'var(--color-text-muted)' }} />
        )}
      </motion.button>

      {/* Text input with focus glow */}
      <div style={styles.inputWrapper}>
        <motion.input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder="Type a command, Boss..."
          disabled={disabled}
          style={{
            ...styles.input,
            ...(isFocused ? styles.inputFocused : {}),
            opacity: disabled ? 0.5 : 1,
          }}
          animate={{
            borderColor: isFocused ? 'var(--color-accent)' : 'rgba(124, 58, 237, 0.2)',
            boxShadow: isFocused
              ? '0 0 0 3px var(--color-accent-glow)'
              : '0 0 0 0px transparent',
          }}
          transition={{ duration: 0.2, ease: 'easeOut' }}
        />
      </div>

      {/* Send button with ripple and hover glow */}
      <motion.button
        onClick={handleSubmit}
        disabled={disabled || !input.trim()}
        className="ripple"
        style={{
          ...styles.sendButton,
          opacity: disabled || !input.trim() ? 0.4 : 1,
        }}
        whileHover={{
          scale: disabled || !input.trim() ? 1 : 1.08,
          y: disabled || !input.trim() ? 0 : -2,
          boxShadow:
            disabled || !input.trim()
              ? 'none'
              : '0 0 20px var(--color-primary-glow)',
        }}
        whileTap={{ scale: disabled || !input.trim() ? 1 : 0.92 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        title="Send message"
      >
        <Send size={18} />
      </motion.button>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-sm)',
    padding: 'var(--spacing-md)',
    background: 'var(--color-surface)',
    borderTop: '1px solid rgba(124, 58, 237, 0.1)',
    flexShrink: 0,
  },
  micButton: {
    width: 40,
    height: 40,
    borderRadius: 'var(--radius-md)',
    background: 'var(--color-surface-raised)',
    border: '1px solid rgba(124, 58, 237, 0.2)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    flexShrink: 0,
  },
  micButtonActive: {
    background: 'rgba(6, 182, 212, 0.1)',
    borderColor: 'var(--color-accent)',
    boxShadow: '0 0 12px var(--color-accent-glow)',
  },
  inputWrapper: {
    flex: 1,
    position: 'relative',
  },
  input: {
    width: '100%',
    padding: 'var(--spacing-sm) var(--spacing-md)',
    background: 'var(--color-bg)',
    border: '1px solid rgba(124, 58, 237, 0.2)',
    borderRadius: 'var(--radius-md)',
    color: 'var(--color-text)',
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--text-sm)',
    outline: 'none',
  },
  inputFocused: {
    borderColor: 'var(--color-accent)',
    boxShadow: '0 0 0 3px var(--color-accent-glow)',
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 'var(--radius-md)',
    background: 'var(--color-primary)',
    color: 'var(--color-text)',
    border: 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    flexShrink: 0,
  },
}
