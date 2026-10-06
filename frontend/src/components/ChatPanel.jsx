import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bot, User } from 'lucide-react'

/**
 * ChatPanel — Scrollable message history with user/assistant bubbles.
 *
 * Auto-scrolls to bottom on new messages. User messages get accent tint,
 * assistant messages get purple sci-fi border accent.
 */
export default function ChatPanel({ messages, isThinking }) {
  const scrollRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isThinking])

  return (
    <div style={styles.container} ref={scrollRef}>
      <AnimatePresence initial={false}>
        {messages.map((msg, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            style={{
              ...styles.messageRow,
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
            }}
          >
            {/* Avatar */}
            <div
              style={{
                ...styles.avatar,
                background:
                  msg.role === 'user'
                    ? 'var(--color-surface-raised)'
                    : 'var(--color-primary)',
              }}
            >
              {msg.role === 'user' ? (
                <User size={14} style={{ color: 'var(--color-accent)' }} />
              ) : (
                <Bot size={14} style={{ color: 'var(--color-text)' }} />
              )}
            </div>

            {/* Bubble */}
            <div
              style={{
                ...styles.bubble,
                ...(msg.role === 'user'
                  ? styles.bubbleUser
                  : styles.bubbleAssistant),
                // Errors are transport/backend failures, not model replies.
                // Marked so a failure is never read as the assistant speaking.
                ...(msg.isError ? styles.bubbleError : {}),
              }}
            >
              <p style={styles.bubbleText}>{msg.content}</p>
              <span style={styles.timestamp}>
                {msg.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Thinking indicator */}
      {isThinking && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          style={{ ...styles.messageRow, justifyContent: 'flex-start' }}
        >
          <div style={{ ...styles.avatar, background: 'var(--color-primary)' }}>
            <Bot size={14} style={{ color: 'var(--color-text)' }} />
          </div>
          <div style={{ ...styles.bubble, ...styles.bubbleAssistant }}>
            <div style={styles.thinkingDots}>
              <span /><span /><span />
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

const styles = {
  container: {
    flex: 1,
    overflowY: 'auto',
    padding: 'var(--spacing-md)',
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--spacing-sm)',
  },
  messageRow: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 'var(--spacing-sm)',
    maxWidth: '85%',
  },
  avatar: {
    width: 28,
    height: 28,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  bubble: {
    padding: 'var(--spacing-sm) var(--spacing-md)',
    borderRadius: 'var(--radius-md)',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  bubbleUser: {
    background: 'var(--color-surface-raised)',
    border: '1px solid rgba(6, 182, 212, 0.25)',
    borderBottomRightRadius: 'var(--radius-sm)',
  },
  bubbleAssistant: {
    background: 'var(--color-surface)',
    border: '1px solid rgba(124, 58, 237, 0.3)',
    borderBottomLeftRadius: 'var(--radius-sm)',
  },
  bubbleError: {
    background: 'rgba(239, 68, 68, 0.06)',
    border: '1px solid rgba(239, 68, 68, 0.35)',
    borderBottomLeftRadius: 'var(--radius-sm)',
  },
  bubbleText: {
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--text-sm)',
    lineHeight: 1.5,
    color: 'var(--color-text)',
    margin: 0,
  },
  timestamp: {
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
    color: 'var(--color-text-muted)',
    alignSelf: 'flex-end',
    marginTop: '2px',
  },
  thinkingDots: {
    display: 'inline-flex',
    gap: '4px',
    padding: '4px 0',
  },
}

// Thinking dot animation styles injected via CSS
const thinkingDotStyles = document.createElement('style')
thinkingDotStyles.textContent = `
  .thinking-dots span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--color-text-muted);
    animation: chatPulse 1.4s ease-in-out infinite;
  }
  .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes chatPulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
  }
`
document.head.appendChild(thinkingDotStyles)
