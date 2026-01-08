# React Client Example for fastrtc-voice-agent

A minimal React application demonstrating how to connect to the voice agent API server.

## Prerequisites

1. Install the voice agent with API support:
   ```bash
   pip install fastrtc-voice-agent[api]
   ```

2. Start the API server:
   ```bash
   fastrtc-voice-agent --run --api --port 8000
   ```

## Running the Example

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Then open http://localhost:5173 in your browser.

## How It Works

The key component is the `useVoiceAgent` hook in `src/useVoiceAgent.ts`. It handles:

1. **WebRTC Connection**: Creates a peer connection to the server
2. **Audio Capture**: Gets microphone access via `getUserMedia`
3. **Signaling**: Exchanges SDP offers/answers with the server
4. **Audio Playback**: Plays the agent's responses automatically

## Integration in Your Project

Copy `src/useVoiceAgent.ts` to your project and use it like this:

```tsx
import { useVoiceAgent } from './useVoiceAgent';

function VoiceChat() {
  const { isConnected, connect, disconnect, error } = useVoiceAgent({
    serverUrl: 'http://your-server:8000',
  });

  return (
    <div>
      <button onClick={isConnected ? disconnect : connect}>
        {isConnected ? 'Stop' : 'Start'}
      </button>
      {error && <p>Error: {error}</p>}
    </div>
  );
}
```

## API Reference

### `useVoiceAgent(options)`

#### Options
- `serverUrl`: The URL of the voice agent API server (e.g., `http://localhost:8000`)

#### Returns
- `isConnected`: Boolean indicating if connected to the server
- `isListening`: Boolean indicating if actively listening to audio
- `error`: Error message if connection failed, or null
- `connect()`: Async function to start the connection
- `disconnect()`: Function to close the connection
