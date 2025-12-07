"""
Custom Voice Agent Web Server with ElevenLabs VAD

This implements a voice agent using:
- ElevenLabs Scribe Realtime STT (WebSocket with built-in VAD)
- Claude LLM (Anthropic API with streaming)
- ElevenLabs TTS (REST API for speech synthesis)

The ElevenLabs Scribe API handles voice activity detection automatically,
so users don't need push-to-talk.
"""

import os
import json
import asyncio
import base64
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, AsyncIterator

import aiohttp
from aiohttp import web
import websockets
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for the voice agent"""
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational, under 2-3 sentences."
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel - default voice
    tts_model: str = "eleven_flash_v2_5"
    llm_model: str = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5 (fastest)
    temperature: float = 0.7
    max_tokens: int = 150


class ElevenLabsRealtimeSTT:
    """
    ElevenLabs Scribe Realtime STT with built-in VAD.
    
    Uses WebSocket connection with commit_strategy=vad for automatic
    voice activity detection and turn-taking.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "wss://api.elevenlabs.io"
        self._ws = None
        self._connected = False
        self._receive_task = None
        
        # Callbacks
        self.on_partial_transcript = None
        self.on_final_transcript = None
        self.on_error = None
    
    async def connect(self) -> bool:
        """Connect to ElevenLabs Scribe Realtime WebSocket with manual commit"""
        try:
            # Build URL with manual commit strategy - browser VAD will trigger commits
            # This gives us faster partial transcripts
            params = [
                "model_id=scribe_v2_realtime",
                "encoding=pcm_16000",  # Format: pcm_{sample_rate}
                "sample_rate=16000",
                "commit_strategy=manual",  # Browser VAD will trigger commit
                "language_code=en"
            ]
            url = f"{self.base_url}/v1/speech-to-text/realtime?{'&'.join(params)}"
            
            print(f"[STT] Connecting to: {url}")
            
            self._ws = await websockets.connect(
                url,
                additional_headers={"xi-api-key": self.api_key}
            )
            self._connected = True
            
            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            print("[STT] Connected to ElevenLabs Scribe Realtime")
            return True
            
        except Exception as e:
            print(f"[STT] Connection error: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False
    
    async def send_audio(self, audio_chunk: bytes):
        """Send audio chunk to STT"""
        if not self._connected or not self._ws:
            return
        
        try:
            # Format message as per ElevenLabs SDK
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(audio_chunk).decode("utf-8"),
                "commit": False,
                "sample_rate": 16000
            }
            await self._ws.send(json.dumps(message))
        except Exception as e:
            print(f"[STT] Send error: {e}")
    
    async def commit_transcription(self):
        """Commit the current transcription - called when browser VAD detects silence"""
        if not self._connected or not self._ws:
            return
        
        try:
            # Send an empty audio chunk with commit=True to trigger final transcript
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": 16000
            }
            await self._ws.send(json.dumps(message))
            print("[STT] Commit sent")
        except Exception as e:
            print(f"[STT] Commit error: {e}")
    
    async def _receive_loop(self):
        """Background loop to receive transcripts"""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    msg_type = data.get("message_type", data.get("type", ""))
                    
                    # Debug: log all messages from ElevenLabs
                    if msg_type not in ["session_started"]:
                        print(f"[STT] Received: {msg_type} -> {data}")
                    
                    if msg_type == "partial_transcript":
                        # API returns 'text' field
                        text = data.get("text", "")
                        if text.strip():
                            print(f"[STT] Partial: {text}")
                            if self.on_partial_transcript:
                                self.on_partial_transcript(text)
                    
                    elif msg_type == "transcript":
                        # Alternative message type for partial/interim transcripts
                        text = data.get("text", data.get("transcript", ""))
                        is_final = data.get("is_final", data.get("final", False))
                        if text.strip():
                            if is_final:
                                print(f"[STT] Final (transcript): {text}")
                                if self.on_final_transcript:
                                    asyncio.create_task(self.on_final_transcript(text))
                            else:
                                print(f"[STT] Partial (transcript): {text}")
                                if self.on_partial_transcript:
                                    self.on_partial_transcript(text)
                    
                    elif msg_type == "committed_transcript":
                        # VAD detected end of speech - this is the final transcript
                        # API returns 'text' field
                        text = data.get("text", "")
                        if text.strip():
                            print(f"[STT] Final: {text}")
                            if self.on_final_transcript:
                                self.on_final_transcript(text)
                    
                    elif msg_type == "session_started":
                        print("[STT] Session started successfully")
                    
                    elif msg_type in ["error", "auth_error", "rate_limited"]:
                        error_msg = data.get("message", str(data))
                        print(f"[STT] Error: {error_msg}")
                        if self.on_error:
                            self.on_error(error_msg)
                            
                except json.JSONDecodeError:
                    pass
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[STT] Connection closed: code={e.code}, reason={e.reason}")
        except Exception as e:
            print(f"[STT] Receive loop error: {e}")
        finally:
            self._connected = False
    
    async def close(self):
        """Close the connection"""
        self._connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            try:
                await self._ws.close()
            except:
                pass
            self._ws = None


class ClaudeLLM:
    """Claude LLM client with streaming"""
    
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    async def stream_response(
        self, 
        messages: list, 
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> AsyncIterator[str]:
        """Stream response tokens from Claude"""
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": messages,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    print(f"[LLM] Error: {response.status} - {error}")
                    return
                
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            pass


class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech using REST API with streaming"""
    
    def __init__(self, api_key: str, voice_id: str, model_id: str = "eleven_flash_v2_5"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
    
    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks as they're generated"""
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        params = {
            "output_format": "pcm_16000"
        }
        
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload
                ) as response:
                    if response.status == 200:
                        # Buffer to ensure 2-byte alignment for 16-bit PCM
                        buffer = b""
                        first_chunk = True
                        async for chunk in response.content.iter_chunked(8192):
                            buffer += chunk
                            # First chunk: send smaller for faster start (50ms)
                            # Later chunks: larger for efficiency (100ms)
                            chunk_size = 1600 if first_chunk else 3200
                            while len(buffer) >= chunk_size:
                                yield buffer[:chunk_size]
                                buffer = buffer[chunk_size:]
                                first_chunk = False
                        
                        # Yield remaining buffer (ensure even length for 16-bit alignment)
                        if buffer:
                            if len(buffer) % 2 != 0:
                                buffer = buffer[:-1]  # Drop last byte if odd
                            if buffer:
                                yield buffer
                    else:
                        error = await response.text()
                        print(f"[TTS] Stream error {response.status}: {error}")
        except Exception as e:
            print(f"[TTS] Stream request error: {e}")


class VoiceAgentSession:
    """
    Manages a single voice agent conversation session.
    
    Uses ElevenLabs Scribe Realtime with built-in VAD for automatic
    turn detection - no push-to-talk needed.
    """
    
    def __init__(
        self,
        ws: web.WebSocketResponse,
        elevenlabs_key: str,
        anthropic_key: str,
        config: AgentConfig
    ):
        self.ws = ws
        self.config = config
        self.elevenlabs_key = elevenlabs_key
        
        # Initialize clients
        self.stt = ElevenLabsRealtimeSTT(elevenlabs_key)
        self.llm = ClaudeLLM(anthropic_key, config.llm_model)
        self.tts = ElevenLabsTTS(elevenlabs_key, config.voice_id, config.tts_model)
        
        # Conversation state
        self.conversation_history = []
        self.is_processing = False
        self.is_speaking = False
        self._running = False
        self._interrupted = False  # Flag to signal barge-in interruption
        self._pending_interrupt_text = None  # Store text that interrupted
        self._audio_playing = False  # Track if browser is playing audio
        self._last_shown_transcript = None  # Prevent duplicate UI messages
        self._recent_agent_speech = []  # Track recent agent speech for echo detection
    
    async def start(self):
        """Start the session - connect to STT with VAD"""
        self._running = True
        
        # Set up STT callbacks
        self.stt.on_partial_transcript = self._on_partial_transcript
        self.stt.on_final_transcript = lambda t: asyncio.create_task(self._on_final_transcript(t))
        self.stt.on_error = lambda e: asyncio.create_task(self._send_event("error", e))
        
        # Connect to STT
        if not await self.stt.connect():
            await self._send_event("error", "Failed to connect to speech recognition")
            return False
        
        await self._send_event("status", "listening")
        return True
    
    async def stop(self):
        """Stop the session"""
        self._running = False
        await self.stt.close()
    
    async def handle_audio(self, audio_data: bytes):
        """Handle incoming audio from browser - forward to STT"""
        if not self._running:
            return
        
        # Always send audio to STT for barge-in detection
        await self.stt.send_audio(audio_data)
    
    def _is_echo(self, text: str) -> bool:
        """Check if the transcribed text is likely echo from agent's speech"""
        text_lower = text.lower().strip()
        # Check if the transcript matches or is contained in recent agent speech
        for agent_text in self._recent_agent_speech:
            agent_lower = agent_text.lower()
            # If user text is very similar to what agent just said, it's echo
            if text_lower in agent_lower or agent_lower.startswith(text_lower):
                return True
        return False
    
    def _on_partial_transcript(self, text: str):
        """Handle partial transcript from STT"""
        if not text.strip():
            return
            
        # Debug: log state when we get a partial transcript
        print(f"[Session] Partial: '{text}' | is_speaking={self.is_speaking}, is_processing={self.is_processing}, _audio_playing={self._audio_playing}, _interrupted={self._interrupted}")
        
        # Check for echo - ignore if it matches recent agent speech
        if self._is_echo(text):
            print(f"[Session] Ignoring echo: {text}")
            return
        
        # If already interrupted, just update the pending text (don't re-trigger)
        if self._interrupted:
            self._pending_interrupt_text = text
            asyncio.create_task(self._send_event("partial_transcript", text))
            return
        
        # If agent is speaking/processing or browser is playing audio, signal interruption (ONCE)
        if (self.is_speaking or self.is_processing or self._audio_playing):
            print(f"[Session] Barge-in detected: {text}")
            self._interrupted = True
            self._pending_interrupt_text = text  # Store for later processing
            # Tell browser to stop playing audio immediately
            asyncio.create_task(self._send_event("clear_audio", True))
        
        asyncio.create_task(self._send_event("partial_transcript", text))
    
    async def _on_final_transcript(self, text: str):
        """Handle final transcript from STT - VAD detected end of speech"""
        if not text.strip():
            return
        
        # Check for echo - ignore if it matches recent agent speech
        if self._is_echo(text):
            print(f"[Session] Ignoring echo (final): {text}")
            return
        
        # If we were in interrupted state, NOW process the user's complete input
        if self._interrupted:
            print(f"[Session] Processing barge-in (final): {text}")
            self._interrupted = False
            self._pending_interrupt_text = None
            await self._send_event("user_transcript", text)
            await self._process_user_input(text)
            return
        
        # If agent is speaking/processing or browser is playing audio, interrupt
        if self.is_speaking or self.is_processing or self._audio_playing:
            print(f"[Session] Interrupting agent with: {text}")
            self._interrupted = True
            self._pending_interrupt_text = text
            self._audio_playing = False  # Reset audio state
            # Tell browser to stop playing audio immediately
            await self._send_event("clear_audio", True)
            # Send immediate feedback
            await self._send_event("user_transcript", text)
            await self._send_event("status", "interrupted")
            return
        
        print(f"[Session] User said: {text}")
        await self._send_event("user_transcript", text)
        await self._process_user_input(text)
    
    async def _process_user_input(self, text: str):
        """Process user input and generate response"""
        self.is_processing = True
        self._interrupted = False
        
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": text
            })
            
            # Generate response with streaming TTS
            print("[Session] Calling LLM with streaming TTS...")
            await self._send_event("status", "thinking")
            
            response_text = ""
            sentence_buffer = ""
            sentence_endings = {'.', '!', '?', ':', ';'}
            first_audio = True
            
            async for token in self.llm.stream_response(
                self.conversation_history,
                self.config.system_prompt,
                self.config.temperature,
                self.config.max_tokens
            ):
                # Check for interruption
                if self._interrupted:
                    print("[Session] LLM generation interrupted by user")
                    break
                
                response_text += token
                sentence_buffer += token
                await self._send_event("partial_response", response_text)
                
                # Check if we have a complete sentence to speak
                for ending in sentence_endings:
                    if ending in sentence_buffer:
                        # Find the last sentence ending
                        last_end = max(
                            sentence_buffer.rfind(ending + ' '),
                            sentence_buffer.rfind(ending) if sentence_buffer.endswith(ending) else -1
                        )
                        if last_end >= 0:
                            # Extract complete sentence(s)
                            speak_text = sentence_buffer[:last_end + 1].strip()
                            sentence_buffer = sentence_buffer[last_end + 1:].lstrip()
                            
                            if speak_text and len(speak_text) > 3:
                                if first_audio:
                                    self.is_speaking = True
                                    self._recent_agent_speech = []  # Clear old speech
                                    await self._send_event("status", "speaking")
                                    first_audio = False
                                
                                # Track for echo detection
                                self._recent_agent_speech.append(speak_text)
                                # Keep only last 3 sentences
                                if len(self._recent_agent_speech) > 3:
                                    self._recent_agent_speech.pop(0)
                                
                                print(f"[Session] Speaking: {speak_text}")
                                async for audio_chunk in self.tts.synthesize_streaming(speak_text):
                                    # Check for interruption during TTS
                                    if self._interrupted or not self._running:
                                        print("[Session] TTS interrupted")
                                        break
                                    await self._send_audio(audio_chunk)
                                
                                # If interrupted during TTS, stop everything
                                if self._interrupted:
                                    break
                            break
                
                # Check again after TTS
                if self._interrupted:
                    break
            
            # Speak any remaining text (if not interrupted)
            if not self._interrupted:
                remaining = sentence_buffer.strip()
                if remaining and len(remaining) > 1:
                    if first_audio:
                        self.is_speaking = True
                        await self._send_event("status", "speaking")
                    
                    print(f"[Session] Speaking remaining: {remaining}")
                    async for audio_chunk in self.tts.synthesize_streaming(remaining):
                        if self._interrupted or not self._running:
                            print("[Session] TTS interrupted")
                            break
                        await self._send_audio(audio_chunk)
            
            # Add response to history (even partial if interrupted)
            if response_text:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text + (" [interrupted]" if self._interrupted else "")
                })
                await self._send_event("agent_response", response_text)
            
            await self._send_event("audio_done", True)
            
            # If there's a pending interrupt, DON'T process immediately
            # The final transcript handler will process it when user finishes speaking
            if self._interrupted and self._pending_interrupt_text:
                print(f"[Session] Waiting for user to finish speaking...")
                # Don't reset interrupted state - let final transcript handle it
                self.is_processing = False
                self.is_speaking = False
                self._audio_playing = False
                return  # Exit and wait for final transcript
            
        except Exception as e:
            print(f"[Session] Error: {e}")
            import traceback
            traceback.print_exc()
            await self._send_event("error", str(e))
        
        # Only reset state if not handling an interrupt
        self.is_processing = False
        self.is_speaking = False
        self._audio_playing = False
        self._recent_agent_speech = []  # Clear echo detection buffer
        await self._send_event("status", "listening")
    
    async def _send_event(self, event_type: str, data):
        """Send event to browser"""
        try:
            await self.ws.send_json({
                "type": event_type,
                "data": data
            })
        except Exception as e:
            print(f"[WS] Send error: {e}")
    
    async def _send_audio(self, audio_data: bytes):
        """Send audio chunk to browser"""
        try:
            await self.ws.send_json({
                "type": "audio",
                "audio": base64.b64encode(audio_data).decode("utf-8")
            })
        except Exception as e:
            print(f"[WS] Audio send error: {e}")


class VoiceAgentServer:
    """Web server for the voice agent"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        
        if not self.elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY not found")
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.config = AgentConfig(voice_id=self.voice_id)
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up HTTP routes"""
        static_path = Path(__file__).parent / "static"
        
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/ws", self._handle_websocket)
        if static_path.exists():
            self.app.router.add_static("/static", static_path)
    
    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve main page"""
        html_path = Path(__file__).parent / "static" / "voice_agent.html"
        if html_path.exists():
            return web.FileResponse(html_path)
        return web.Response(text="voice_agent.html not found", status=404)
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection from browser"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        print("[Server] Client connected")
        
        session = VoiceAgentSession(
            ws=ws,
            elevenlabs_key=self.elevenlabs_key,
            anthropic_key=self.anthropic_key,
            config=self.config
        )
        
        if not await session.start():
            return ws
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "audio":
                        # Decode and forward audio to STT
                        audio_b64 = data.get("audio", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            await session.handle_audio(audio_bytes)
                    
                    elif data.get("type") == "user_speaking":
                        # Browser detected user speaking (instant VAD)
                        if data.get("interrupted"):
                            print("[Server] Browser VAD: User interrupted agent")
                            session._interrupted = True
                            session._audio_playing = False
                    
                    elif data.get("type") == "audio_status":
                        # Browser reports audio playback status
                        playing = data.get("playing", False)
                        if playing:  # Only log when starting, not every stop
                            print(f"[Server] Audio status from browser: playing")
                        session._audio_playing = playing
                    
                    elif data.get("type") == "commit":
                        # Browser VAD detected silence - commit the transcription
                        print("[Server] Browser VAD: Committing transcription")
                        await session.stt.commit_transcription()
                    
                    elif data.get("type") == "config":
                        if "system_prompt" in data:
                            session.config.system_prompt = data["system_prompt"]
                        await ws.send_json({"type": "status", "data": "config_updated"})
                    
                    elif data.get("type") == "clear_history":
                        session.conversation_history = []
                        await ws.send_json({"type": "status", "data": "history_cleared"})
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[Server] WebSocket error: {ws.exception()}")
                    break
        finally:
            await session.stop()
            print("[Server] Client disconnected")
        
        return ws
    
    def run(self):
        """Run the server"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║       Voice Agent with ElevenLabs VAD - No Push-to-Talk     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Open in browser: http://{self.host}:{self.port}                        ║
║                                                              ║
║  Components:                                                 ║
║    • STT: ElevenLabs Scribe Realtime (with VAD)             ║
║    • LLM: Claude ({self.config.llm_model[:20]}...)               ║
║    • TTS: ElevenLabs Flash v2.5                             ║
║                                                              ║
║  Just speak naturally - no button needed!                    ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        web.run_app(self.app, host=self.host, port=self.port, print=None)


def main():
    """Main entry point"""
    try:
        server = VoiceAgentServer()
        server.run()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nMake sure your .env file has:")
        print("  ELEVENLABS_API_KEY=your-key")
        print("  ANTHROPIC_API_KEY=your-key")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0


if __name__ == "__main__":
    exit(main())
