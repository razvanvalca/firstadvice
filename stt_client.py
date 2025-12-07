"""
Speech-to-Text Client Module
============================

ElevenLabs Scribe API client for real-time speech-to-text with WebSocket streaming.
Provides ~100-150ms latency with continuous transcription while user speaks.
"""

import asyncio
import json
import base64
import websockets
from dataclasses import dataclass
from typing import AsyncGenerator, Optional


@dataclass
class TranscriptResult:
    """Result from STT processing"""
    text: str
    is_final: bool
    confidence: float = 1.0
    words: list = None  # Word-level timestamps
    language: Optional[str] = None


class ElevenLabsSTTClient:
    """
    ElevenLabs Scribe API client for real-time speech-to-text.
    
    Why WebSocket over REST:
    - Persistent connection reduces per-request overhead
    - Bidirectional streaming for real-time partial results
    - Lower latency (~150ms vs ~500ms for REST)
    """
    
    WEBSOCKET_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    
    def __init__(
        self,
        api_key: str,
        model: str = "scribe_v2_realtime",
        language: Optional[str] = None,  # Auto-detect if None
        sample_rate: int = 16000
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._transcript_buffer = ""
        self._is_connected = False
        self._result_queue = asyncio.Queue()
        self._receive_task = None
    
    async def connect(self):
        """Establish WebSocket connection for real-time streaming"""
        if self._is_connected:
            return
        
        # Build URL with query parameters
        url = f"{self.WEBSOCKET_URL}?model_id={self.model}"
        if self.language:
            url += f"&language_code={self.language}"
        
        headers = {"xi-api-key": self.api_key}
        
        self._websocket = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10
        )
        self._is_connected = True
        
        # Start background task to receive transcripts
        self._receive_task = asyncio.create_task(self._receive_loop())
    
    async def _receive_loop(self):
        """Background task to receive transcription results"""
        try:
            async for message in self._websocket:
                data = json.loads(message)
                msg_type = data.get("type", "")
                
                if msg_type == "transcript":
                    # Partial transcript - update buffer
                    self._transcript_buffer = data.get("text", "")
                    await self._result_queue.put(TranscriptResult(
                        text=self._transcript_buffer,
                        is_final=False
                    ))
                    
                elif msg_type == "transcript_final":
                    # Final transcript for this segment
                    final_text = data.get("text", "")
                    await self._result_queue.put(TranscriptResult(
                        text=final_text,
                        is_final=True
                    ))
                    self._transcript_buffer = ""
                    
        except websockets.exceptions.ConnectionClosed:
            self._is_connected = False
        except asyncio.CancelledError:
            pass
    
    async def send_audio(self, audio_chunk: bytes):
        """
        Send audio chunk for transcription.
        
        Args:
            audio_chunk: Raw PCM audio (16kHz, 16-bit, mono)
        """
        if not self._is_connected:
            await self.connect()
        
        # Encode audio as base64 (required by API)
        audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
        
        await self._websocket.send(json.dumps({"audio": audio_b64}))
    
    async def get_transcript(self, timeout: float = 1.0) -> Optional[TranscriptResult]:
        """Get next transcript result from queue"""
        try:
            return await asyncio.wait_for(self._result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def commit(self):
        """
        Manually commit current transcript segment.
        
        Use at logical breakpoints (end of utterance) to force final transcript.
        """
        if self._is_connected and self._websocket:
            await self._websocket.send(json.dumps({"type": "commit"}))
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            self._is_connected = False


class LocalWhisperSTT:
    """
    Local STT using faster-whisper for offline use.
    
    Why use local:
    - No network latency
    - Data privacy (audio never leaves device)
    - Works offline
    
    Tradeoff: Higher compute requirements, potentially higher latency on CPU
    """
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device)
        except ImportError:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")
        self._audio_buffer = bytearray()
    
    async def transcribe(self, audio: bytes) -> TranscriptResult:
        import numpy as np
        
        # Convert to float array
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        segments, info = self.model.transcribe(audio_array, beam_size=5)
        text = " ".join([seg.text for seg in segments])
        
        return TranscriptResult(text=text.strip(), is_final=True, language=info.language)
