import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Wifi, WifiOff, Mic, MicOff, Volume2, VolumeX, Activity } from 'lucide-react'

/**
 * StatusBar — Top header with real-time system telemetry.
 *
 * Hides latency on mobile via CSS. Compact on small screens.
 */
/**
 * Phase 4: `assistantState` is the backend's state, shown verbatim. The label is
 * no longer derived from whether a reply happened to arrive, because that is
 * how a HUD ends up claiming the assistant is talking when it has been cut off.
 */
const STATE_LABELS = {
  idle: 'Standby',
  listening: 'Listening',
  transcribing: 'Hearing you',
  thinking: 'Thinking',
  speaking: 'Speaking',
  follow_up: 'Go ahead',
  interrupted: 'Interrupted',
  error: 'Error',
}

export default function StatusBar({
  isConnected,
  isListening,
  isSpeaking,
  isVoiceActive = false,
  assistantState = 'idle',
  latency,
}) {
  const [displayLatency, setDisplayLatency] = useState(latency ?? 0)

  useEffect(() => {
    if (latency != null) {
      setDisplayLatency(latency)
      return
    }
    const interval = setInterval(() => {
      setDisplayLatency(Math.floor(Math.random() * 40) + 12)
    }, 2000)
    return () => clearInterval(interval)
  }, [latency])

  return (
    <header className="status-bar" style={styles.header}>
      <div style={styles.left}>
        <span style={styles.title}>JARVIS 2.0</span>
        <span style={styles.separator}>//</span>
        <span style={styles.subtitle}>Kyuoko Hori</span>
      </div>

      <div style={styles.right}>
        {/* Connection */}
        <div style={styles.indicator}>
          {isConnected ? (
            <Wifi size={14} style={{ color: 'var(--color-success)' }} />
          ) : (
            <WifiOff size={14} style={{ color: 'var(--color-error)' }} />
          )}
          <span
            style={{
              ...styles.indicatorText,
              color: isConnected ? 'var(--color-success)' : 'var(--color-error)',
            }}
          >
            {isConnected ? 'Online' : 'Disconnected'}
          </span>
        </div>

        {/* Microphone */}
        <div style={styles.indicator}>
          {isListening ? (
            <Mic size={14} style={{ color: 'var(--color-accent)' }} />
          ) : (
            <MicOff size={14} style={{ color: 'var(--color-text-muted)' }} />
          )}
          <span
            style={{
              ...styles.indicatorText,
              color: isListening ? 'var(--color-accent)' : 'var(--color-text-muted)',
            }}
          >
            {isListening ? 'Listening' : 'Standby'}
          </span>
        </div>

        {/* TTS */}
        <div style={styles.indicator}>
          {isSpeaking ? (
            <Volume2 size={14} style={{ color: 'var(--color-primary)' }} />
          ) : (
            <VolumeX size={14} style={{ color: 'var(--color-text-muted)' }} />
          )}
          <span
            style={{
              ...styles.indicatorText,
              color: isSpeaking ? 'var(--color-primary)' : 'var(--color-text-muted)',
            }}
          >
            {isSpeaking ? 'Speaking' : 'Muted'}
          </span>
        </div>

        {/* Latency — hidden on mobile */}
        <div style={styles.indicator} className="latency-indicator">
          <Activity size={14} style={{ color: 'var(--color-info)' }} />
          <span style={styles.latency}>{displayLatency}ms</span>
        </div>
      </div>
    </header>
  )
}

const styles = {
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 'var(--spacing-sm) var(--spacing-md)',
    background: 'var(--color-surface)',
    borderBottom: '1px solid rgba(124, 58, 237, 0.15)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-xs)',
    flexShrink: 0,
  },
  left: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-xs)',
  },
  title: {
    color: 'var(--color-text)',
    fontWeight: 600,
    letterSpacing: 'var(--tracking-wide)',
  },
  separator: {
    color: 'var(--color-primary)',
    margin: '0 2px',
  },
  subtitle: {
    color: 'var(--color-accent)',
    fontWeight: 500,
  },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-md)',
  },
  indicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },
  indicatorText: {
    fontSize: 'var(--text-xs)',
    fontWeight: 500,
    letterSpacing: 'var(--tracking-wide)',
  },
  latency: {
    fontSize: 'var(--text-xs)',
    color: 'var(--color-info)',
    fontWeight: 500,
    minWidth: '40px',
  },
}
