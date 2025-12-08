# Swiss Life Voice Insurance Advisor

A real-time voice-based insurance consultation system powered by ElevenLabs speech services, Claude AI, and TF-IDF product retrieval.

## 🎯 Overview

This application provides an interactive voice agent that helps users explore Swiss Life insurance products through natural conversation. The agent:

- **Listens** to user speech via ElevenLabs Scribe (real-time STT)
- **Retrieves** relevant product information using TF-IDF similarity search
- **Generates** contextual responses with Claude AI (Haiku 4.5)
- **Speaks** responses using ElevenLabs TTS (Flash v2.5)
- **Manages** a structured sales consultation with tracked task completion

## 🏗️ Architecture

```
new app/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── start.bat                   # Windows startup script
├── start.sh                    # Unix/Linux/macOS startup script
├── .env.example                # Environment variable template
│
├── config/                     # Configuration layer
│   ├── __init__.py
│   ├── settings.py             # Centralized settings (dataclasses)
│   └── prompts/
│       └── system_prompt.md    # German system prompt for the agent
│
├── core/                       # Domain layer
│   ├── __init__.py
│   ├── models.py               # Domain models (Task, SessionConfig)
│   ├── processors.py           # Response processing utilities
│   └── session.py              # VoiceAgentSession coordinator
│
├── data/                       # Static data
│   └── products.md             # Swiss Life product documentation
│
├── services/                   # External service clients
│   ├── __init__.py
│   ├── llm.py                  # Claude AI client (streaming)
│   ├── product_rag.py          # TF-IDF product retrieval
│   ├── speech_to_text.py       # ElevenLabs Scribe STT
│   └── text_to_speech.py       # ElevenLabs TTS
│
└── web/                        # Presentation layer
    ├── __init__.py
    ├── server.py               # aiohttp HTTP/WebSocket server
    └── static/
        ├── index.html          # Frontend UI with embedded CSS
        └── js/
            └── app.js          # Client-side JavaScript (VAD, audio)
```

## 🔧 Configuration

All configuration is managed through environment variables and the `config/settings.py` module.

### Environment Variables (.env)

```bash
# Required API Keys
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
DEBUG=false

# Optional: TTS Settings
TTS_VOICE_ID=EXAVITQu4vr4xnSDxMaL        # Sarah voice
TTS_MODEL_ID=eleven_flash_v2_5
TTS_SAMPLE_RATE=16000
TTS_SPEED=1.1

# Optional: STT Settings
STT_LANGUAGE=de                           # German
STT_SAMPLE_RATE=16000

# Optional: LLM Settings
LLM_MODEL=claude-haiku-4-5-20241022
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.7

# Optional: RAG Settings
RAG_TOP_K=3
RAG_MAX_FEATURES=5000

# Optional: VAD Settings (client-side)
VAD_THRESHOLD=0.08
VAD_SILENCE_DURATION_MS=1200
VAD_DEBOUNCE_MS=800
```

### Settings Classes

The `config/settings.py` module provides typed configuration via Python dataclasses:

- `TTSSettings` - Text-to-speech configuration
- `STTSettings` - Speech-to-text configuration
- `LLMSettings` - Language model configuration
- `RAGSettings` - Product retrieval configuration
- `VADSettings` - Voice activity detection configuration
- `ServerSettings` - HTTP server configuration
- `Settings` - Main configuration container

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- ElevenLabs API key (with Scribe access)
- Anthropic API key

### Installation

1. **Clone or copy the application:**

   ```bash
   cd "new app"
   ```

2. **Create virtual environment (recommended):**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Unix/Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the application:**

   ```bash
   # Option 1: Direct Python
   python main.py
   ```

6. **Open in browser:**
   ```
   http://localhost:8080
   ```

## 🎙️ How It Works

### Voice Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                        BROWSER                                            │
│                                      (app.js)                                             │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌───────────────────────┐ │
│   │ Microphone  │───▶│ VAD Engine  │───▶│ PCM Encoder     │───▶│ WebSocket Client      │ │
│   │ getUserMedia│    │ - Barge-in  │    │ - 16kHz/16-bit  │    │ - JSON + Base64 audio │ │
│   │ 16kHz mono  │    │ - Silence   │    │ - 1024 samples  │    │ - Events: audio,      │ │
│   └─────────────┘    │   detection │    │   per chunk     │    │   commit, interrupt   │ │
│                      └─────────────┘    └─────────────────┘    └───────────┬───────────┘ │
│                            │                                               │             │
│                            │ (user interrupts agent)                       │             │
│                            ▼                                               │             │
│   ┌─────────────────────────────────────────────────────┐                  │             │
│   │ Audio Playback System                               │                  │             │
│   │ - AudioContext queue with scheduled playback        │                  │             │
│   │ - Gain node for instant muting on barge-in          │                  │             │
│   │ - Real-time buffer management                       │◀─────────────────┼─────────┐   │
│   └─────────────────────────────────────────────────────┘                  │         │   │
│                                                                            │         │   │
└────────────────────────────────────────────────────────────────────────────┼─────────┼───┘
                                                                             │         │
                                         WebSocket Connection (ws:// or wss://)        │
                                                                             │         │
                                                                             ▼         │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                        SERVER                                           │
│                                     (aiohttp + session.py)                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. SPEECH-TO-TEXT                                                                 │  │
│  │    (services/speech_to_text.py)                                                   │  │
│  │    ┌────────────────────┐         ┌─────────────────────────────────────────────┐ │  │
│  │    │ Browser Audio      │────────▶│ ElevenLabs Scribe Realtime API             │ │  │
│  │    │ (PCM Base64)       │         │ - WebSocket streaming                       │ │  │
│  │    └────────────────────┘         │ - Manual commit strategy (browser-controlled)│ │  │
│  │                                   │ - German language (de)                      │ │  │
│  │                                   └──────────────────────┬──────────────────────┘ │  │
│  │                                                          │                        │  │
│  │                                                          ▼                        │  │
│  │                                            ┌─────────────────────────┐            │  │
│  │                                            │ Partial + Final         │            │  │
│  │                                            │ Transcripts             │────────────┼──┼───▶ UI
│  └────────────────────────────────────────────┴─────────────┬───────────┴────────────┘  │
│                                                             │                           │
│                                                             ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 2. RAG RETRIEVAL                                                                  │  │
│  │    (services/product_rag.py)                                                      │  │
│  │    ┌────────────────────┐         ┌─────────────────────────────────────────────┐ │  │
│  │    │ User Query         │────────▶│ TF-IDF Vectorizer                           │ │  │
│  │    │                    │         │ - 1-2 word n-grams                          │ │  │
│  │    └────────────────────┘         │ - Cosine similarity search                  │ │  │
│  │                                   │ - Top-K results (default: 5)                │ │  │
│  │                                   └──────────────────────┬──────────────────────┘ │  │
│  │                                                          │                        │  │
│  │    ┌─────────────────────────────────────────────────────▼──────────────────────┐ │  │
│  │    │ Product Chunks (from data/products.md)                                     │ │  │
│  │    │ - Parsed by section headers                                                │ │  │
│  │    │ - Swiss Life product info: Investment, Retirement, Protection             │ │  │
│  │    └────────────────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┬───────────────────────┘  │
│                                                             │                           │
│                                                             ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. LLM GENERATION                                                                 │  │
│  │    (services/llm.py)                                                              │  │
│  │    ┌────────────────────┐         ┌─────────────────────────────────────────────┐ │  │
│  │    │ System Prompt      │         │ Anthropic Claude API (claude-haiku-4-5)     │ │  │
│  │    │ + RAG Context      │────────▶│ - Streaming SSE response                    │ │  │
│  │    │ + Conversation     │         │ - Prompt caching for efficiency             │ │  │
│  │    │   History          │         │ - Task completion markers [TASK_X_DONE]     │ │  │
│  │    └────────────────────┘         └──────────────────────┬──────────────────────┘ │  │
│  │                                                          │                        │  │
│  │                                                          ▼                        │  │
│  │                                            ┌─────────────────────────┐            │  │
│  │                                            │ Streaming Tokens        │────────────┼──┼───▶ UI
│  │                                            │ (partial response)      │            │  │
│  └────────────────────────────────────────────┴─────────────┬───────────┴────────────┘  │
│                                                             │                           │
│                                                             ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 4. SENTENCE PROCESSING                                                            │  │
│  │    (core/processors.py)                                                           │  │
│  │    ┌────────────────────┐         ┌─────────────────────────────────────────────┐ │  │
│  │    │ Token Stream       │────────▶│ Sentence Splitter                           │ │  │
│  │    │                    │         │ - Buffers until . ! ? detected              │ │  │
│  │    └────────────────────┘         │ - Strips [TASK_X_DONE] markers              │ │  │
│  │                                   │ - Enables incremental TTS                   │ │  │
│  │                                   └──────────────────────┬──────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┼───────────────────────┘  │
│                                                             │                           │
│                                                             ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 5. TEXT-TO-SPEECH                                                                 │  │
│  │    (services/text_to_speech.py)                                                   │  │
│  │    ┌────────────────────┐         ┌─────────────────────────────────────────────┐ │  │
│  │    │ Complete Sentence  │────────▶│ ElevenLabs TTS REST API                     │ │  │
│  │    │                    │         │ - Model: eleven_flash_v2_5                  │ │  │
│  │    └────────────────────┘         │ - Output: PCM 16kHz signed 16-bit           │ │  │
│  │                                   │ - Streaming chunks (1600-3200 bytes)        │ │  │
│  │                                   └──────────────────────┬──────────────────────┘ │  │
│  │                                                          │                        │  │
│  │                                                          ▼                        │  │
│  │                                            ┌─────────────────────────┐            │  │
│  │                                            │ PCM Audio Chunks        │────────────┼──┼───▶ Browser
│  │                                            │ (Base64 via WebSocket)  │            │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ BARGE-IN HANDLING (core/session.py)                                               │  │
│  │ ═══════════════════════════════════════════════════════════════════════════════  │  │
│  │ • Browser detects user speech during playback → sends "user_speaking" event      │  │
│  │ • Server cancels ongoing LLM generation (asyncio.Task.cancel())                  │  │
│  │ • Server sends "clear_audio" → Browser mutes gain node + stops all sources       │  │
│  │ • Echo detection prevents agent's own speech from triggering barge-in            │  │
│  │ • Conversation context preserved for seamless continuation                       │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Message Flow (WebSocket JSON Protocol)

| Direction | Message Type | Payload | Purpose |
|-----------|-------------|---------|---------|
| Browser → Server | `audio` | `{audio: base64}` | Microphone PCM chunks |
| Browser → Server | `commit` | `{}` | VAD silence detected, commit transcript |
| Browser → Server | `user_speaking` | `{interrupted: true}` | User interrupted agent |
| Browser → Server | `audio_status` | `{playing: bool}` | Playback state update |
| Browser → Server | `config` | `{tasks: [...]}` | Session configuration |
| Server → Browser | `status` | `"listening"/"thinking"/"speaking"` | State changes |
| Server → Browser | `partial_transcript` | `"text..."` | Real-time STT |
| Server → Browser | `user_transcript` | `"text..."` | Final user speech |
| Server → Browser | `partial_response` | `"text..."` | Streaming LLM output |
| Server → Browser | `agent_response` | `"text..."` | Complete agent response |
| Server → Browser | `audio` | `{audio: base64}` | TTS PCM chunks |
| Server → Browser | `clear_audio` | `true` | Stop playback (barge-in) |
| Server → Browser | `task_update` | `{id, completed}` | Task completion |

### Key Features

1. **Real-time Speech Recognition**

   - Uses ElevenLabs Scribe with WebSocket streaming
   - Manual commit strategy triggered by client VAD
   - Ultra-low latency transcription

2. **Product Retrieval (RAG)**

   - TF-IDF vectorization with 1-2 n-grams
   - Cosine similarity search
   - Returns top 3 most relevant product chunks

3. **Contextual Response Generation**

   - Claude Haiku 4.5 with prompt caching
   - Streaming responses for low latency
   - Structured task tracking embedded in prompt

4. **Natural Speech Synthesis**

   - ElevenLabs Flash v2.5 for speed
   - 16kHz PCM streaming to browser
   - Sentence-by-sentence synthesis

5. **Interruption Handling**

   - User can speak at any time to interrupt
   - Audio playback stops immediately
   - Context preserved for conversation continuity

6. **Task Management**
   - Three-step sales consultation:
     1. Needs Analysis (Bedarfsermittlung)
     2. Product Recommendation (Produktempfehlung)
     3. Objection Handling (Einwandbehandlung)
   - Visual progress tracking in UI

## 📁 Key Files

### `services/speech_to_text.py`

Manages WebSocket connection to ElevenLabs Scribe API:

- Async audio streaming
- Transcript callbacks
- Manual commit on voice silence

### `services/text_to_speech.py`

REST API client for ElevenLabs TTS:

- Streaming audio chunks
- Configurable voice and speed
- PCM format for browser playback

### `services/llm.py`

Claude AI integration with prompt caching:

- System prompt caching for efficiency
- Streaming token generation
- Conversation history management

### `services/product_rag.py`

TF-IDF based product search:

- Markdown document parsing
- Section-based chunking
- Similarity ranking

### `core/session.py`

Main session coordinator:

- Orchestrates all services
- Handles interruptions
- Manages echo detection
- Processes task completion markers

### `web/server.py`

HTTP and WebSocket server:

- Serves static files
- WebSocket endpoint for voice
- REST endpoint for configuration

## 🔒 Security Notes

- API keys should never be committed to version control
- Use `.env` file for local development
- Use environment variables in production
- The application runs on localhost by default

## 📝 License

This project is proprietary to Swiss Life.

## 🤝 Contributing

Please follow the established code structure:

- Services in `services/`
- Domain logic in `core/`
- Configuration in `config/`
- Web layer in `web/`

All code should follow PEP 8 and include type hints.
