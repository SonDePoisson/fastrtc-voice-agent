import { useVoiceAgent } from './useVoiceAgent';

const SERVER_URL = 'http://localhost:8000';

function App() {
  const { isConnected, isListening, error, connect, disconnect } = useVoiceAgent({
    serverUrl: SERVER_URL,
  });

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Voice Agent Demo</h1>
      <p style={styles.subtitle}>
        Connect to speak with the AI assistant
      </p>

      <button
        onClick={isConnected ? disconnect : connect}
        style={{
          ...styles.button,
          backgroundColor: isConnected ? '#dc3545' : '#007bff',
        }}
      >
        {isConnected ? 'Stop Conversation' : 'Start Conversation'}
      </button>

      <div style={styles.status}>
        {error && <p style={styles.error}>{error}</p>}
        {isListening && (
          <p style={styles.listening}>
            <span style={styles.dot}>●</span> Listening...
          </p>
        )}
        {!isConnected && !error && (
          <p style={styles.hint}>Click the button to start talking</p>
        )}
      </div>

      <div style={styles.instructions}>
        <h3>Instructions</h3>
        <ol>
          <li>
            Start the voice agent server:
            <code style={styles.code}>
              fastrtc-voice-agent --run --api --port 8000
            </code>
          </li>
          <li>Click "Start Conversation" above</li>
          <li>Allow microphone access when prompted</li>
          <li>Speak naturally - the agent will respond!</li>
        </ol>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: '600px',
    margin: '0 auto',
    padding: '40px 20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    textAlign: 'center',
  },
  title: {
    fontSize: '2rem',
    marginBottom: '8px',
    color: '#333',
  },
  subtitle: {
    color: '#666',
    marginBottom: '32px',
  },
  button: {
    padding: '16px 32px',
    fontSize: '1.1rem',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'opacity 0.2s',
  },
  status: {
    marginTop: '24px',
    minHeight: '50px',
  },
  error: {
    color: '#dc3545',
  },
  listening: {
    color: '#28a745',
    fontSize: '1.2rem',
  },
  dot: {
    animation: 'pulse 1.5s infinite',
  },
  hint: {
    color: '#999',
  },
  instructions: {
    marginTop: '48px',
    textAlign: 'left',
    padding: '20px',
    backgroundColor: '#f8f9fa',
    borderRadius: '8px',
  },
  code: {
    display: 'block',
    backgroundColor: '#e9ecef',
    padding: '8px 12px',
    borderRadius: '4px',
    marginTop: '8px',
    fontFamily: 'monospace',
    fontSize: '0.9rem',
  },
};

export default App;
