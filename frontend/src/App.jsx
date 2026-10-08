import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X } from 'lucide-react'
import StatusBar from './components/StatusBar'
import CharacterAvatar from './components/CharacterAvatar'
import ListeningIndicator from './components/ListeningIndicator'
import ChatPanel from './components/ChatPanel'
import InputArea from './components/InputArea'
import { useJarvisState } from './hooks/useJarvisState'

/**
 * JARVIS 2.0 — Main Application Assembly
 *
 * Responsive layout:
 * - Desktop (1200px+): Fixed sidebar + chat pane side-by-side
 * - Tablet (<1200px): Narrower sidebar, fluid chat
 * - Mobile (<768px): Stacked layout, sidebar becomes slide-out drawer
 */
export default function App() {
  const [drawerOpen, setDrawerOpen] = useState(false)

  const {
    messages,
    isConnected,
    isListening,
    isSpeaking,
    assistantState,
    latency,
    isThinking,
    isVoiceActive,
    backendState,
    sendMessage,
    interrupt,
    toggleVoice,
  } = useJarvisState()

  return (
    <div className="app">
      {/* Top status bar */}
      <StatusBar
        isConnected={isConnected}
        isListening={isListening}
        isSpeaking={isSpeaking}
        isVoiceActive={isVoiceActive}
        assistantState={assistantState}
        conversationState={backendState.conversation_state}
        latency={latency}
      />

      {/* Mobile header (visible < 768px) */}
      <div className="mobile-header">
        <span className="mobile-header-title">JARVIS 2.0</span>
        <div className="mobile-header-actions">
          <button
            className="mobile-icon-btn"
            onClick={isSpeaking ? interrupt : toggleVoice}
            title={isSpeaking ? 'Interrupt' : (isVoiceActive ? 'Stop voice input' : 'Start voice input')}
          >
            {isSpeaking ? '⏹' : (isVoiceActive ? '🎤' : '🎙️')}
          </button>
          <button
            className="mobile-icon-btn"
            onClick={() => setDrawerOpen(true)}
            title="Open menu"
          >
            <Menu size={20} />
          </button>
        </div>
      </div>

      {/* Main content area */}
      <div className="app-main">
        {/* Sidebar — hidden on mobile, visible on desktop */}
        <motion.aside
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className="app-sidebar"
        >
          <CharacterAvatar state={assistantState} />
          <ListeningIndicator isActive={isListening} />
        </motion.aside>

        {/* Chat area */}
        <motion.main
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut', delay: 0.1 }}
          className="app-chat"
        >
          <ChatPanel messages={messages} isThinking={isThinking} />
          {/* Input stays enabled while a reply is pending: messages typed during
              a request are queued and sent in order, not discarded. */}
          <InputArea onSend={sendMessage} disabled={false} />
        </motion.main>
      </div>

      {/* Mobile drawer overlay */}
      <AnimatePresence>
        {drawerOpen && (
          <>
            <motion.div
              className={`drawer-overlay ${drawerOpen ? 'open' : ''}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setDrawerOpen(false)}
            />
            <motion.div
              className={`drawer ${drawerOpen ? 'open' : ''}`}
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
            >
              <div className="drawer-header">
                <span style={{ fontWeight: 600, fontSize: 'var(--text-lg)' }}>
                  JARVIS 2.0
                </span>
                <button
                  className="drawer-close"
                  onClick={() => setDrawerOpen(false)}
                >
                  <X size={20} />
                </button>
              </div>
              <CharacterAvatar state={assistantState} />
              <ListeningIndicator isActive={isListening} />
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
