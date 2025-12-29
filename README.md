# fastrtc-voice-agent

A modular voice agent built on [FastRTC](https://github.com/gradio-app/fastrtc)

## Installation

```bash
pip install fastrtc-voice-agent
```

Please install your desired STT and LLM with (for example) :

```bash
pip install "fastrtc-voice-agent[ollama]"
```

or for all optional dependancies :

```bash
pip install "fastrtc-voice-agent[all]"
```

## CLI Usage Example

For default config :

```bash
fastrtc-voice-agent --run
```

Please refere to the help for custom config :

```bash
fastrtc-voice-agent --help
```

## Python Usage Example

```python
from fastrtc import ReplyOnPause, Stream
from voice_agent import create_agent, AgentConfig, STTConfig, TTSConfig, LLMConfig

config = AgentConfig(
    system_prompt="You are a helpful voice assistant.",
    stt=STTConfig(backend="faster_whisper", model_size="small"),
    tts=TTSConfig(backend="edge", voice="en-US-AvaMultilingualNeural"),
    llm=LLMConfig(backend="ollama", model="llama3.2:3b"),
)

agent = create_agent(config)

stream = Stream(
    ReplyOnPause(agent.create_fastrtc_handler()),
    modality="audio",
    mode="send-receive",
)

stream.ui.launch()
```

## Note

To use Anthropic API (may be OpenAI or else later) please copy .env.example as .env file and fill it with your API KEY and the desired model
