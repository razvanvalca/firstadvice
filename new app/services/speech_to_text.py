"""
ElevenLabs Speech-to-Text Service

Real-time speech transcription using ElevenLabs Scribe API via WebSocket.
"""

import json
import base64
import asyncio
from typing import Callable, Optional

import websockets

from config.settings import ElevenLabsConfig


class SpeechToTextService:
    """
    Real-time speech-to-text service using ElevenLabs Scribe.
    
    Uses WebSocket connection for streaming transcription with
    manual commit strategy for browser-controlled VAD.
    """
    
    def __init__(self, config: ElevenLabsConfig):
        """
        Initialize the STT service.
        
        Args:
            config: ElevenLabs configuration
        """
        self._config = config
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_partial_transcript: Optional[Callable[[str], None]] = None
        self.on_final_transcript: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to the STT service."""
        return self._connected
    
    async def connect(self) -> bool:
        """
        Connect to ElevenLabs Scribe Realtime WebSocket.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = self._build_connection_url()
            print(f"[STT] Connecting to: {url}")
            
            self._ws = await websockets.connect(
                url,
                additional_headers={"xi-api-key": self._config.api_key}
            )
            self._connected = True
            
            # Start background receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            print("[STT] Connected to ElevenLabs Scribe Realtime")
            return True
            
        except Exception as e:
            print(f"[STT] Connection error: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def _build_connection_url(self) -> str:
        """Build the WebSocket connection URL with parameters."""
        params = [
            f"model_id={self._config.stt_model}",
            f"encoding=pcm_{self._config.stt_sample_rate}",
            f"sample_rate={self._config.stt_sample_rate}",
            f"commit_strategy={self._config.stt_commit_strategy}",
            f"language_code={self._config.stt_language}",
        ]
        return f"{self._config.stt_base_url}/v1/speech-to-text/realtime?{'&'.join(params)}"
    
    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send an audio chunk to the STT service.
        
        Args:
            audio_chunk: Raw audio bytes (PCM 16-bit)
        """
        if not self._connected or not self._ws:
            return
        
        try:
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(audio_chunk).decode("utf-8"),
                "commit": False,
                "sample_rate": self._config.stt_sample_rate,
            }
            await self._ws.send(json.dumps(message))
        except Exception as e:
            print(f"[STT] Send error: {e}")
    
    async def commit_transcription(self) -> None:
        """
        Commit the current transcription.
        
        Called when browser VAD detects end of speech.
        """
        if not self._connected or not self._ws:
            return
        
        try:
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": self._config.stt_sample_rate,
            }
            await self._ws.send(json.dumps(message))
            print("[STT] Commit sent")
        except Exception as e:
            print(f"[STT] Commit error: {e}")
    
    async def _receive_loop(self) -> None:
        """Background loop to receive and process transcripts."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    pass
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[STT] Connection closed: code={e.code}, reason={e.reason}")
        except Exception as e:
            print(f"[STT] Receive loop error: {e}")
        finally:
            self._connected = False
    
    async def _handle_message(self, data: dict) -> None:
        """Handle a message from the STT service."""
        msg_type = data.get("message_type", data.get("type", ""))
        
        if msg_type == "partial_transcript":
            text = data.get("text", "")
            if text.strip():
                print(f"[STT] Partial: {text}")
                if self.on_partial_transcript:
                    self.on_partial_transcript(text)
        
        elif msg_type == "transcript":
            text = data.get("text", data.get("transcript", ""))
            is_final = data.get("is_final", data.get("final", False))
            if text.strip():
                if is_final:
                    print(f"[STT] Final (transcript): {text}")
                    if self.on_final_transcript:
                        self.on_final_transcript(text)
                else:
                    print(f"[STT] Partial (transcript): {text}")
                    if self.on_partial_transcript:
                        self.on_partial_transcript(text)
        
        elif msg_type == "committed_transcript":
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
    
    async def close(self) -> None:
        """Close the connection to the STT service."""
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
            except Exception:
                pass
            self._ws = None
