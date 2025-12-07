# Voice Agent - ElevenLabs Conversational AI

A voice agent implementation using ElevenLabs Conversational AI with sub-1-second response latency.

## Two Options

This project provides two ways to run a voice agent:

### Option 1: Web Agent (Recommended)

Uses ElevenLabs' fully managed Conversational AI platform with:

- Built-in VAD (Voice Activity Detection)
- Built-in STT (Speech-to-Text)
- Built-in LLM integration
- Built-in TTS (Text-to-Speech)
- Browser-based interface

### Option 2: CLI Demo

Custom pipeline with individual components:

- ElevenLabs Scribe STT
- WebRTC VAD
- Claude/GPT LLM
- ElevenLabs TTS

---

## Quick Start: Web Agent

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create an Agent in ElevenLabs Dashboard

1. Go to: https://elevenlabs.io/app/conversational-ai
2. Click "Create Agent"
3. Configure your agent:
   - **Voice**: Choose a voice (e.g., your custom voice or a preset)
   - **Model**: Select "Eleven Flash v2.5" (required for English)
   - **LLM**: Choose Claude Haiku, GPT, or their built-in model
   - **System Prompt**: Define your agent's personality
   - **First Message**: What the agent says when conversation starts
4. Save and copy the **Agent ID** from the settings

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add:

```env
ELEVENLABS_API_KEY=your-api-key
ELEVENLABS_AGENT_ID=your-agent-id-from-dashboard
```

### 4. Run the Web Agent

```bash
python web_agent.py
```

Open http://localhost:8080 in your browser.

---

## Quick Start: CLI Demo

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install pyaudio  # For voice mode
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add:

```env
ELEVENLABS_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-anthropic-key  # Or OPENAI_API_KEY
ELEVENLABS_VOICE_ID=your-voice-id     # Optional
```

### 3. Run the Demo

**Text Mode** (no microphone):

```bash
python demo.py --mode text
```

**Benchmark Mode**:

```bash
python demo.py --mode benchmark
```

**Voice Mode** (requires microphone):

```bash
python demo.py --mode voice
```

---

## Architecture

### Web Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ElevenLabs Conversational AI                    │
│                                                                     │
│   Browser ──WebSocket──► ElevenLabs Server                         │
│      │                      │                                       │
│   Audio In               VAD + STT + LLM + TTS                     │
│   Audio Out ◄───────────── Streamed Response                       │
│                                                                     │
│   Everything managed by ElevenLabs - Ultra low latency             │
└─────────────────────────────────────────────────────────────────────┘
```

### CLI Demo

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CUSTOM PIPELINE                               │
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │   STT    │    │   VAD    │    │   LLM    │    │   TTS    │     │
│   │ (Scribe) │    │(WebRTC) │    │ (Claude) │    │(ElevenLabs)│    │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘     │
│        │               │               │               │            │
│   Streaming       End-of-turn      Streaming       Streaming        │
│   transcripts     detection        tokens          audio            │
│                                                                      │
│               ALL COMPONENTS RUN IN PARALLEL                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── web_agent.py      # Web-based agent using ElevenLabs Conversational AI
├── static/
│   └── agent.html    # Web interface
├── demo.py           # CLI demo with custom pipeline
├── pipeline.py       # Pipeline orchestrator
├── stt_client.py     # ElevenLabs Scribe STT client
├── vad_client.py     # WebRTC VAD
├── llm_client.py     # Claude/GPT streaming clients
├── tts_client.py     # ElevenLabs TTS streaming client
├── requirements.txt  # Python dependencies
├── .env.example      # Example environment variables
└── README.md         # This file
```

---

## Troubleshooting

### "ELEVENLABS_AGENT_ID not found"

Create an agent in the ElevenLabs dashboard and add the ID to your `.env` file.

### "Microphone not found" (CLI mode)

Make sure `pyaudio` is installed:

```bash
pip install pyaudio
```

### WebSocket connection errors

Check your API key is valid and you have credits in your ElevenLabs account.

---

## Links

- [ElevenLabs Conversational AI](https://elevenlabs.io/conversational-ai)
- [ElevenLabs Dashboard](https://elevenlabs.io/app)
- [API Documentation](https://elevenlabs.io/docs/conversational-ai)
