import { motion } from 'framer-motion'

/**
 * CharacterAvatar — Central anime character with reactive glow and pulse.
 *
 * Scales down on mobile via CSS. Animations change based on assistant state.
 */
export default function CharacterAvatar({ state = 'idle' }) {
  const quotes = {
    idle: "Standing by, Boss.",
    listening: "I'm all ears.",
    speaking: "On it.",
  }

  const glowColors = {
    idle: 'rgba(124, 58, 237, 0.25)',
    listening: 'rgba(6, 182, 212, 0.5)',
    speaking: 'rgba(124, 58, 237, 0.6)',
  }

  const borderColors = {
    idle: 'var(--color-primary)',
    listening: 'var(--color-accent)',
    speaking: 'var(--color-primary)',
  }

  return (
    <div className="character-avatar-container" style={styles.container}>
      {/* Outer glow ring */}
      <motion.div
        style={{
          ...styles.glowRing,
          background: `radial-gradient(circle, ${glowColors[state]}, transparent 70%)`,
        }}
        animate={{
          scale:
            state === 'speaking'
              ? [1, 1.1, 1]
              : state === 'listening'
              ? [1, 1.05, 1]
              : [1, 1.02, 1],
          opacity: state === 'idle' ? 0.3 : 0.7,
        }}
        transition={{
          duration: state === 'speaking' ? 0.8 : state === 'listening' ? 1.2 : 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />

      {/* Ripple rings when speaking */}
      {state === 'speaking' && (
        <>
          <motion.div
            style={styles.rippleRing}
            initial={{ scale: 1, opacity: 0.5 }}
            animate={{ scale: 1.6, opacity: 0 }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeOut' }}
          />
          <motion.div
            style={{ ...styles.rippleRing, borderColor: 'var(--color-accent)' }}
            initial={{ scale: 1, opacity: 0.3 }}
            animate={{ scale: 1.8, opacity: 0 }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeOut', delay: 0.5 }}
          />
        </>
      )}

      {/* Rotating border ring */}
      <motion.div
        style={{
          ...styles.borderRing,
          borderColor: borderColors[state],
        }}
        animate={{ rotate: 360 }}
        transition={{
          duration: state === 'listening' ? 4 : 8,
          repeat: Infinity,
          ease: 'linear',
        }}
      />

      {/* Inner avatar circle */}
      <motion.div
        style={styles.avatar}
        animate={{
          boxShadow: [
            `0 0 20px ${glowColors[state]}`,
            `0 0 40px ${glowColors[state]}`,
            `0 0 20px ${glowColors[state]}`,
          ],
        }}
        transition={{
          duration: state === 'speaking' ? 0.8 : 1.5,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      >
        <span style={styles.initial}>KH</span>
      </motion.div>

      {/* Character name */}
      <motion.p
        style={styles.name}
        animate={{ y: [0, -2, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
      >
        Kyuoko Hori
      </motion.p>

      {/* Dynamic subtitle */}
      <motion.p
        key={state}
        style={styles.subtitle}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
      >
        {quotes[state]}
      </motion.p>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 'var(--spacing-xl)',
    position: 'relative',
  },
  glowRing: {
    position: 'absolute',
    width: 180,
    height: 180,
    borderRadius: '50%',
    pointerEvents: 'none',
  },
  rippleRing: {
    position: 'absolute',
    width: 140,
    height: 140,
    borderRadius: '50%',
    border: '2px solid var(--color-primary)',
    pointerEvents: 'none',
  },
  borderRing: {
    position: 'absolute',
    width: 140,
    height: 140,
    borderRadius: '50%',
    border: '2px dashed',
    pointerEvents: 'none',
  },
  avatar: {
    width: 120,
    height: 120,
    borderRadius: '50%',
    background: 'var(--color-surface-raised)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    zIndex: 1,
  },
  initial: {
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--text-3xl)',
    fontWeight: 700,
    color: 'var(--color-text)',
    letterSpacing: 'var(--tracking-tight)',
  },
  name: {
    marginTop: 'var(--spacing-md)',
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--text-lg)',
    fontWeight: 600,
    color: 'var(--color-text)',
    letterSpacing: 'var(--tracking-tight)',
  },
  subtitle: {
    marginTop: 'var(--spacing-xs)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    color: 'var(--color-text-muted)',
    fontStyle: 'italic',
  },
}
