# Swiss Life Voice Insurance Advisor

A German-language voice agent for Swiss Life insurance consultation, featuring real-time speech recognition, LLM-powered responses with product knowledge (RAG), and natural text-to-speech.

## Features

- **Real-time Voice Conversation** - Speak naturally, no push-to-talk needed
- **Client-side VAD** - Browser handles voice activity detection for instant barge-in
- **Product RAG** - TF-IDF search over Swiss Life product documentation
- **Streaming Responses** - LLM tokens stream directly to TTS for low latency
- **German Language** - Optimized for Swiss German business context

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Custom Voice Agent Pipeline                       │
│                                                                      │
│   Browser ──────── WebSocket ──────► Python Server                  │
│      │                                    │                          │
│   Client VAD                         ┌────┴────┐                    │
│   (silence detect)                   │ Session │                    │
│      │                               └────┬────┘                    │
│   Audio chunks ──────────────────────────►│                         │
│                                           ▼                          │
│                              ElevenLabs Scribe STT                  │
│                                    (WebSocket)                      │
│                                           │                          │
│                              ┌────────────┼────────────┐            │
│                              │     Tool Check (LLM)    │            │
│                              │            │            │            │
│                              │     Product RAG ◄───────┤            │
│                              │     (TF-IDF search)     │            │
│                              └────────────┬────────────┘            │
│                                           │                          │
│                              Claude Haiku (streaming)               │
│                              + Prompt Caching                       │
│                                           │                          │
│                              ElevenLabs TTS (REST)                  │
│                              (sentence-by-sentence)                 │
│                                           │                          │
│   ◄────────────── Audio chunks ──────────┘                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add:

```env
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
ELEVENLABS_VOICE_ID=your-german-voice-id  # Optional
```

### 3. Run the Server

```bash
python voice_agent_server.py
```

Open http://localhost:8080 in your browser.

## Project Structure

```
├── voice_agent_server.py  # Main server (STT, LLM, TTS, WebSocket)
├── product_rag.py         # TF-IDF RAG for product search
├── sl_products.md         # Swiss Life product documentation
├── static/
│   └── voice_agent.html   # Web interface with client-side VAD
├── requirements.txt       # Python dependencies
└── .env.example           # Example environment variables
```

## Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| STT | ElevenLabs Scribe Realtime | WebSocket streaming transcription |
| LLM | Claude Haiku 4.5 | Fast responses with prompt caching |
| TTS | ElevenLabs Flash v2.5 | Low-latency German speech synthesis |
| RAG | scikit-learn TF-IDF | Product knowledge retrieval |
| VAD | Browser-side | Instant voice activity detection |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ELEVENLABS_API_KEY` | Yes | ElevenLabs API key |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `ELEVENLABS_VOICE_ID` | No | Custom German voice ID |

## Links

- [ElevenLabs](https://elevenlabs.io/)
- [Anthropic Claude](https://www.anthropic.com/)
