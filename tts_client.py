"""
Text-to-Speech Client Module
============================

ElevenLabs TTS with WebSocket streaming for ultra-low latency (~75ms TTFB).
Supports multi-context for seamless interruption handling.
"""

import asyncio
import json
import base64
import time
import websockets
from dataclasses import dataclass
from typing import AsyncGenerator, Optional


@dataclass
class VoiceSettings:
    """Voice synthesis settings"""
    stability: float = 0.5        # 0-1, higher = more consistent
    similarity_boost: float = 0.75  # 0-1, higher = closer to original
    style: float = 0.0            # 0-1, style exaggeration
    speed: float = 1.0            # 0.25-4.0, speaking speed


class ElevenLabsTTSClient:
    """
    ElevenLabs TTS with WebSocket streaming.
    
    Why WebSocket over REST:
    - Persistent connection (no reconnection overhead)
    - Bidirectional streaming (send text, receive audio simultaneously)
    - Lower TTFB (~75ms vs ~200ms for REST)
    - Multi-context support for interruption handling
    
    Models:
    - eleven_flash_v2_5: ~75ms TTFB, good quality (RECOMMENDED)
    - eleven_turbo_v2_5: ~150ms TTFB, better quality
    - eleven_multilingual_v2: ~300ms TTFB, best quality
    """
    
    WEBSOCKET_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
    MULTI_CONTEXT_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input"
    
    # Popular voice IDs
    VOICES = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",   # American female
        "drew": "29vD33N1CtxCmqQRPOHJ",     # American male
        "clyde": "2EiwWnXFnvU5JabPnv8n",    # British male
        "sarah": "EXAVITQu4vr4xnSDxMaL",    # American female
        "domi": "AZnzlk1XvdvUeBnXmlld",     # American female
        "bella": "EXAVITQu4vr4xnSDxMaL",    # American female
    }
    
    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_flash_v2_5",
        voice_settings: Optional[VoiceSettings] = None,
        output_format: str = "mp3_44100_128"
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.voice_settings = voice_settings or VoiceSettings()
        self.output_format = output_format
        
        self._websocket = None
        self._is_connected = False
        self._audio_queue = asyncio.Queue()
        self._current_context_id = None
        self._receive_task = None
    
    async def connect(self, multi_context: bool = False):
        """Establish WebSocket connection"""
        if self._is_connected:
            return
        
        # Choose URL based on mode
        if multi_context:
            url = self.MULTI_CONTEXT_URL.format(voice_id=self.voice_id)
        else:
            url = self.WEBSOCKET_URL.format(voice_id=self.voice_id)
        
        url += f"?model_id={self.model_id}&output_format={self.output_format}"
        
        headers = {"xi-api-key": self.api_key}
        
        self._websocket = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10
        )
        self._is_connected = True
        
        # Send initialization message
        init_message = {
            "text": " ",  # Space required to initialize
            "voice_settings": {
                "stability": self.voice_settings.stability,
                "similarity_boost": self.voice_settings.similarity_boost,
                "speed": self.voice_settings.speed
            },
            "generation_config": {
                # Chunk schedule balances latency vs quality
                # Smaller first chunk = faster TTFB
                "chunk_length_schedule": [120, 160, 250, 290]
            }
        }
        
        await self._websocket.send(json.dumps(init_message))
        
        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())
    
    async def _receive_loop(self):
        """Background task to receive audio chunks"""
        try:
            async for message in self._websocket:
                data = json.loads(message)
                
                audio_b64 = data.get("audio")
                is_final = data.get("isFinal", False)
                
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    await self._audio_queue.put((audio_bytes, is_final))
                
                elif is_final:
                    await self._audio_queue.put((None, True))
        
        except websockets.exceptions.ConnectionClosed:
            self._is_connected = False
        except asyncio.CancelledError:
            pass
    
    async def stream_speech(
        self,
        text: str,
        flush: bool = False,
        context_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech.
        
        Args:
            text: Text to synthesize
            flush: Force generation of buffered text (use at sentence end)
            context_id: Optional context ID for multi-context mode
        
        Yields:
            Audio bytes as they're generated
        
        Key insight: Call this incrementally as LLM generates tokens.
        Don't wait for full LLM response!
        """
        if not self._is_connected:
            await self.connect()
        
        # Send text (don't clear queue - let audio accumulate)
        message = {
            "text": text + " ",  # Trailing space helps prosody
            "try_trigger_generation": True
        }
        
        if flush:
            message["flush"] = True
        
        if context_id:
            message["context_id"] = context_id
        
        await self._websocket.send(json.dumps(message))
        
        # Yield any available audio chunks without blocking long
        start_time = time.time()
        first_chunk = True
        
        # Only wait briefly for audio - don't block the LLM streaming
        timeout = 0.5 if not flush else 2.0
        
        while True:
            try:
                audio_bytes, is_final = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=timeout
                )
                
                if audio_bytes is None:
                    break
                
                if first_chunk:
                    ttfb = (time.time() - start_time) * 1000
                    print(f"[TTS TTFB: {ttfb:.0f}ms]")
                    first_chunk = False
                
                yield audio_bytes
                
                if is_final:
                    break
                    
            except asyncio.TimeoutError:
                break
    
    async def flush_audio(self) -> AsyncGenerator[bytes, None]:
        """Flush and collect any remaining audio in the queue"""
        if not self._is_connected:
            return
            
        # Send flush signal
        await self._websocket.send(json.dumps({"text": "", "flush": True}))
        
        # Collect remaining audio with longer timeout
        while True:
            try:
                audio_bytes, is_final = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=2.0
                )
                
                if audio_bytes is None or is_final:
                    if audio_bytes:
                        yield audio_bytes
                    break
                
                yield audio_bytes
                    
            except asyncio.TimeoutError:
                break
    
    async def stop(self):
        """Stop current synthesis (for interruption)"""
        if self._is_connected and self._websocket:
            try:
                # Send empty text to close current generation
                await self._websocket.send(json.dumps({"text": ""}))
            except:
                pass
            
            # Clear queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"text": ""}))
            except:
                pass
            await self._websocket.close()
            self._websocket = None
            self._is_connected = False


class TextChunker:
    """
    Utility for chunking text for optimal TTS streaming.
    
    Why chunk:
    - TTS needs some text to determine prosody
    - Too little text = unnatural speech
    - Too much text = high latency
    
    Strategy: Flush at sentence boundaries for natural pauses
    """
    
    SENTENCE_ENDINGS = {'.', '!', '?', ':', ';'}
    CLAUSE_ENDINGS = {',', '-', '—'}
    
    @staticmethod
    def should_flush(text: str) -> bool:
        """Determine if text should be flushed to TTS"""
        text = text.rstrip()
        
        # Flush at sentence boundaries
        if any(text.endswith(p) for p in TextChunker.SENTENCE_ENDINGS):
            return True
        
        # Flush if too long (balance latency vs quality)
        if len(text) > 100:
            return True
        
        # Optionally flush at clause boundaries if long enough
        if len(text) > 50 and any(text.endswith(p) for p in TextChunker.CLAUSE_ENDINGS):
            return True
        
        return False
    
    @staticmethod
    def chunk_by_sentence(text: str) -> list[str]:
        """Split text into sentences"""
        chunks = []
        current = ""
        
        for char in text:
            current += char
            if char in TextChunker.SENTENCE_ENDINGS:
                if current.strip():
                    chunks.append(current.strip())
                current = ""
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks


class InterruptionHandler:
    """
    Handles interruptions using multi-context WebSockets.
    
    How it works:
    1. Each response uses a unique context_id
    2. When interrupted, close current context
    3. Create new context for next response
    4. Track which text was actually played using alignment data
    """
    
    def __init__(self, tts_client: ElevenLabsTTSClient):
        self.tts = tts_client
        self._context_counter = 0
        self._played_text = ""
        self._alignment_buffer = []
    
    def new_context(self) -> str:
        """Create new context for response"""
        self._context_counter += 1
        self._played_text = ""
        return f"ctx_{self._context_counter}"
    
    async def stream_with_tracking(
        self,
        text: str,
        context_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS while tracking what was played.
        
        Word-level alignment tells us exactly which words
        were converted to audio. If interrupted, we know
        what the user actually heard.
        """
        async for chunk in self.tts.stream_speech(text, context_id=context_id):
            # Track played text (simplified - full implementation would use alignment data)
            self._played_text += text
            yield chunk
    
    async def handle_interruption(self, context_id: str) -> str:
        """
        Handle interruption for context.
        
        1. Close current context (stops generation)
        2. Return what was actually played
        """
        # Stop TTS
        await self.tts.stop()
        
        # Return what was played
        played = self._played_text
        self._played_text = ""
        
        return played
