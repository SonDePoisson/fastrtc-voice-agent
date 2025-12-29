"""Command-line interface for voice_agent."""

import argparse
import sys

from .config import (
    STT_BACKENDS,
    TTS_BACKENDS,
    LLM_BACKENDS,
)


def print_backends() -> None:
    """Print all available backends with their descriptions and defaults."""
    print("\n=== Speech-to-Text (STT) Backends ===\n")
    for name, info in STT_BACKENDS.items():
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Default model: {info['default_model']}")
        print(f"    Default device: {info['default_device']}")
        print()

    print("=== Text-to-Speech (TTS) Backends ===\n")
    for name, info in TTS_BACKENDS.items():
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Default voice: {info['default_voice']}")
        print()

    print("=== Large Language Model (LLM) Backends ===\n")
    for name, info in LLM_BACKENDS.items():
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Default model: {info['default_model']}")
        print()


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fastrtc-voice-agent",
        description="A modular voice agent with swappable STT/TTS/LLM backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fastrtc-voice-agent --list          List all available backends
  fastrtc-voice-agent --run           Run the voice agent with default config
  fastrtc-voice-agent --run --llm claude --stt faster_whisper

For more information, visit: https://github.com/SonDePoisson/voice-agent
        """,
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available backends (STT, TTS, LLM)",
    )

    parser.add_argument(
        "--run",
        "-r",
        action="store_true",
        help="Run the voice agent",
    )

    parser.add_argument(
        "--stt",
        type=str,
        choices=list(STT_BACKENDS.keys()),
        default="faster_whisper",
        help="STT backend to use (default: faster_whisper)",
    )

    parser.add_argument(
        "--tts",
        type=str,
        choices=list(TTS_BACKENDS.keys()),
        default="edge",
        help="TTS backend to use (default: edge)",
    )

    parser.add_argument(
        "--llm",
        type=str,
        choices=list(LLM_BACKENDS.keys()),
        default="ollama",
        help="LLM backend to use (default: ollama)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to use for the LLM backend",
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        help="Model size for STT (default: small)",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["cpu", "cuda"],
        help="Device for STT (default: cpu)",
    )

    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        help="Voice to use for TTS",
    )

    parser.add_argument(
        "--system-prompt",
        "-s",
        type=str,
        help="System prompt for the agent",
    )

    parser.add_argument(
        "--system-prompt-file",
        type=str,
        help="Path to file containing system prompt",
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    if args.list:
        print_backends()
        return 0

    if args.run:
        # Import here to avoid circular imports and speed up --help/--list
        from fastrtc import ReplyOnPause, Stream

        from . import create_agent, AgentConfig, STTConfig, TTSConfig, LLMConfig

        # Resolve actual values from defaults
        stt_defaults = STT_BACKENDS[args.stt]
        tts_defaults = TTS_BACKENDS[args.tts]
        llm_defaults = LLM_BACKENDS[args.llm]

        model_size = args.model_size or stt_defaults["default_model"]
        device = args.device or stt_defaults["default_device"]
        voice = args.voice or tts_defaults["default_voice"]
        model = args.model or llm_defaults["default_model"]

        config = AgentConfig(
            system_prompt=args.system_prompt or "You are a helpful voice assistant.",
            system_prompt_file=args.system_prompt_file,
            stt=STTConfig(backend=args.stt, model_size=model_size, device=device),
            tts=TTSConfig(backend=args.tts, voice=voice),
            llm=LLMConfig(backend=args.llm, model=model),
        )

        print("Starting voice agent with:")
        print(f"  STT: {args.stt} (model: {model_size}, device: {device})")
        print(f"  TTS: {args.tts} (voice: {voice})")
        print(f"  LLM: {args.llm} (model: {model})")
        print()

        agent = create_agent(config)
        stream = Stream(
            ReplyOnPause(agent.create_fastrtc_handler()),
            modality="audio",
            mode="send-receive",
        )
        stream.ui.launch()
        return 0

    # If no action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
