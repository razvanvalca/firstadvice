"""
ElevenLabs Text-to-Speech Service

Streaming text-to-speech synthesis using ElevenLabs API.
"""

from typing import AsyncIterator

import aiohttp

from config.settings import ElevenLabsConfig


class TextToSpeechService:
    """
    Text-to-speech service using ElevenLabs REST API.
    
    Provides streaming audio synthesis for low-latency playback.
    """
    
    def __init__(self, config: ElevenLabsConfig):
        """
        Initialize the TTS service.
        
        Args:
            config: ElevenLabs configuration
        """
        self._config = config
    
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """
        Synthesize speech from text with streaming output.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks (PCM 16-bit @ 16kHz)
        """
        url = f"{self._config.tts_base_url}/v1/text-to-speech/{self._config.voice_id}/stream"
        
        headers = {
            "xi-api-key": self._config.api_key,
            "Content-Type": "application/json",
        }
        
        params = {
            "output_format": self._config.tts_output_format,
        }
        
        payload = {
            "text": text,
            "model_id": self._config.tts_model,
            "voice_settings": {
                "stability": self._config.voice_stability,
                "similarity_boost": self._config.voice_similarity_boost,
                "speed": self._config.voice_speed,
            },
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload,
                ) as response:
                    if response.status == 200:
                        async for chunk in self._stream_audio(response):
                            yield chunk
                    else:
                        error = await response.text()
                        print(f"[TTS] Error {response.status}: {error}")
        except Exception as e:
            print(f"[TTS] Request error: {e}")
    
    async def _stream_audio(self, response: aiohttp.ClientResponse) -> AsyncIterator[bytes]:
        """
        Stream audio chunks from the response.
        
        Uses adaptive chunk sizes: smaller for first chunk (faster start),
        larger for subsequent chunks (efficiency).
        
        Args:
            response: HTTP response with audio stream
            
        Yields:
            Audio chunks aligned to 16-bit samples
        """
        buffer = b""
        first_chunk = True
        chunk_count = 0
        
        async for chunk in response.content.iter_chunked(8192):
            buffer += chunk
            
            # First chunk: smaller for faster start (~50ms at 16kHz)
            # Later chunks: larger for efficiency (~100ms at 16kHz)
            chunk_size = 1600 if first_chunk else 3200
            
            while len(buffer) >= chunk_size:
                chunk_count += 1
                yield buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                first_chunk = False
        
        # Yield remaining buffer (ensure 16-bit alignment)
        if buffer:
            if len(buffer) % 2 != 0:
                buffer = buffer[:-1]  # Drop odd byte
            if buffer:
                chunk_count += 1
                yield buffer
        
        if chunk_count == 0:
            print(f"[TTS] Warning: No audio chunks for text: {text[:50]}...")
