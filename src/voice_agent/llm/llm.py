"""LLM backends for voice_agent."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

import anthropic
import ollama as ollama_client
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a complete response.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Generator:
        """Stream response tokens.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions

        Yields:
            Response chunks (with .response attribute)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass


class OllamaBackend(LLMBackend):
    """LLM backend using Ollama for local inference."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    @property
    def name(self) -> str:
        return f"ollama-{self.model}"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a complete response."""
        response = ollama_client.generate(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
        )
        return response.response

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Generator:
        """Stream response tokens."""
        return ollama_client.generate(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
            stream=True,
        )


class ClaudeBackend(LLMBackend):
    """LLM backend using Anthropic's Claude API."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in .env file or passed as argument")
        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return f"claude-{self.model}"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a complete response."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Generator:
        """Stream response tokens."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield _ClaudeStreamChunk(text)


class _ClaudeStreamChunk:
    """Wrapper to match Ollama's stream chunk interface."""

    def __init__(self, text: str):
        self.response = text


def create_llm_backend(
    backend: str = "ollama",
    model: str | None = None,
) -> LLMBackend:
    """Factory function to create an LLM backend.

    Args:
        backend: Backend type ("ollama" or "claude")
        model: Model identifier (optional, uses defaults if not specified)

    Returns:
        Configured LLM backend instance
    """
    if backend == "ollama":
        return OllamaBackend(model=model or "llama3.2:3b")
    elif backend == "claude":
        return ClaudeBackend(model=model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
