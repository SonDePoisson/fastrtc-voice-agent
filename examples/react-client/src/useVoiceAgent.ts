import { useState, useRef, useCallback } from 'react';

interface UseVoiceAgentOptions {
  serverUrl: string;
}

interface UseVoiceAgentReturn {
  isConnected: boolean;
  isListening: boolean;
  error: string | null;
  connect: () => Promise<void>;
  disconnect: () => void;
}

/**
 * React hook for connecting to a FastRTC voice agent server.
 *
 * @example
 * ```tsx
 * const { isConnected, connect, disconnect } = useVoiceAgent({
 *   serverUrl: 'http://localhost:8000',
 * });
 *
 * return (
 *   <button onClick={isConnected ? disconnect : connect}>
 *     {isConnected ? 'Stop' : 'Start'}
 *   </button>
 * );
 * ```
 */
export function useVoiceAgent({ serverUrl }: UseVoiceAgentOptions): UseVoiceAgentReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pcRef = useRef<RTCPeerConnection | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const connect = useCallback(async () => {
    try {
      setError(null);

      // Create peer connection
      const pc = new RTCPeerConnection();
      pcRef.current = pc;

      // Capture microphone
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      stream.getTracks().forEach(track => pc.addTrack(track, stream));

      // Create required data channel (FastRTC requirement)
      pc.createDataChannel("data");

      // Handle incoming audio from the agent
      pc.ontrack = (event) => {
        if (!audioRef.current) {
          audioRef.current = new Audio();
          audioRef.current.autoplay = true;
        }
        audioRef.current.srcObject = event.streams[0];
      };

      // Handle connection state changes
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
          setIsConnected(false);
          setIsListening(false);
        }
      };

      // Create and send offer to server
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE gathering to complete
      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
        } else {
          pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === 'complete') {
              resolve();
            }
          };
        }
      });

      // Send offer to server
      const response = await fetch(`${serverUrl}/webrtc/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription?.sdp,
          type: pc.localDescription?.type,
          webrtc_id: crypto.randomUUID(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const answer = await response.json();
      await pc.setRemoteDescription(answer);

      setIsConnected(true);
      setIsListening(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Connection failed';
      setError(message);
      disconnect();
    }
  }, [serverUrl]);

  const disconnect = useCallback(() => {
    // Stop all tracks
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;

    // Close peer connection
    pcRef.current?.close();
    pcRef.current = null;

    // Stop audio
    if (audioRef.current) {
      audioRef.current.srcObject = null;
    }

    setIsConnected(false);
    setIsListening(false);
  }, []);

  return { isConnected, isListening, error, connect, disconnect };
}
