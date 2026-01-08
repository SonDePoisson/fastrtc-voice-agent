"""FastAPI server for voice agent API mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import Stream, ReplyOnPause

if TYPE_CHECKING:
    from .config import AgentConfig


def create_api_server(
    config: AgentConfig | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI server with the voice agent mounted.

    Args:
        config: Agent configuration. If None, uses defaults.
        cors_origins: List of allowed CORS origins. Defaults to ["*"].

    Returns:
        FastAPI app with WebRTC endpoints at:
        - POST /webrtc/offer - WebRTC signaling
        - WS /websocket/offer - WebSocket alternative

    Example:
        ```python
        from voice_agent import create_api_server

        app = create_api_server()
        # Run with: uvicorn main:app --host 0.0.0.0 --port 8000
        ```

        Or mount in an existing app:
        ```python
        from fastapi import FastAPI
        from voice_agent import create_api_server

        main_app = FastAPI()
        voice_app = create_api_server()
        main_app.mount("/voice", voice_app)
        ```
    """
    app = FastAPI(
        title="Voice Agent API",
        description="WebRTC-based voice agent with STT/LLM/TTS pipeline",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import here to avoid circular import
    from .core import VoiceAgent
    from .config import AgentConfig as AgentConfigClass
    from .stt import create_stt_backend
    from .tts import create_tts_backend
    from .llm import create_llm_backend

    config = config or AgentConfigClass()
    stt = create_stt_backend(
        backend=config.stt.backend,
        model_size=config.stt.model_size,
        device=config.stt.device,
    )
    tts = create_tts_backend(
        backend=config.tts.backend,
        voice=config.tts.voice,
    )
    llm = create_llm_backend(
        backend=config.llm.backend,
        model=config.llm.model,
    )
    agent = VoiceAgent(stt=stt, tts=tts, llm=llm, config=config)
    stream = Stream(
        ReplyOnPause(agent.create_fastrtc_handler()),
        modality="audio",
        mode="send-receive",
    )
    stream.mount(app)

    return app
