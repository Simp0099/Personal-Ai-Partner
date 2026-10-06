import { motion } from 'framer-motion'

const BAR_COUNT = 12

/**
 * ListeningIndicator — Animated audio waveform with fluid keyframes.
 *
 * Bars pulse vertically with staggered timing when active.
 * Includes ripple ring feedback and glow effects.
 */
export default function ListeningIndicator({ isActive = false }) {
  return (
    <div style={styles.container}>
      {/* Ripple ring behind waveform */}
      {isActive && (
        <motion.div
          style={styles.rippleRing}
          initial={{ scale: 0.8, opacity: 0.6 }}
          animate={{ scale: 1.8, opacity: 0 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeOut' }}
        />
      )}

      {/* Waveform bars */}
      <div style={styles.waveform}>
        {Array.from({ length: BAR_COUNT }).map((_, i) => (
          <motion.div
            key={i}
            style={{
              ...styles.bar,
              animationDelay: `${i * 0.06}s`,
            }}
            animate={
              isActive
                ? {
                    scaleY: [0.3, 1, 0.3],
                    opacity: [0.5, 1, 0.5],
                  }
                : { scaleY: 0.3, opacity: 0.4 }
            }
            transition={
              isActive
                ? {
                    duration: 0.6 + Math.random() * 0.4,
                    repeat: Infinity,
                    ease: 'easeInOut',
                    delay: i * 0.06,
                  }
                : { duration: 0.4, ease: 'easeOut' }
            }
          />
        ))}
      </div>

      {/* Label with glow */}
      <motion.span
        style={styles.label}
        animate={{
          opacity: isActive ? [0.6, 1, 0.6] : 0.5,
          textShadow: isActive
            ? [
                '0 0 8px var(--color-accent-glow)',
                '0 0 16px var(--color-accent-glow)',
                '0 0 8px var(--color-accent-glow)',
              ]
            : '0 0 0px transparent',
        }}
        transition={{
          duration: 1.2,
          repeat: isActive ? Infinity : 0,
          ease: 'easeInOut',
        }}
      >
        LISTENING...
      </motion.span>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 'var(--spacing-sm)',
    padding: 'var(--spacing-md)',
    position: 'relative',
  },
  rippleRing: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 60,
    height: 60,
    borderRadius: '50%',
    border: '2px solid var(--color-accent)',
    transform: 'translate(-50%, -50%)',
    pointerEvents: 'none',
  },
  waveform: {
    display: 'flex',
    alignItems: 'center',
    gap: '3px',
    height: '36px',
  },
  bar: {
    width: '3px',
    height: '100%',
    borderRadius: 'var(--radius-full)',
    background: 'var(--color-accent)',
    boxShadow: '0 0 6px var(--color-accent-glow)',
    transformOrigin: 'center',
  },
  label: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-xs)',
    fontWeight: 500,
    letterSpacing: 'var(--tracking-wide)',
    color: 'var(--color-accent)',
  },
}
