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

        config = AgentConfig(
            system_prompt=args.system_prompt or "You are a helpful voice assistant.",
            system_prompt_file=args.system_prompt_file,
            stt=STTConfig(backend=args.stt),
            tts=TTSConfig(backend=args.tts, voice=args.voice),
            llm=LLMConfig(backend=args.llm, model=args.model),
        )

        print("Starting voice agent with:")
        print(f"  STT: {args.stt}")
        print(f"  TTS: {args.tts} (voice: {args.voice or 'default'})")
        print(f"  LLM: {args.llm} (model: {args.model or 'default'})")
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
