"""
ElevenLabs Conversational AI Agent
==================================

Uses ElevenLabs' Conversational AI WebSocket API which includes:
- Built-in STT (Scribe)
- Proprietary VAD and turn-taking
- LLM integration (or custom LLM)
- TTS with ultra-low latency

This provides the full ElevenLabs voice agent experience.
"""

import asyncio
import json
import base64
import time
import os
from dataclasses import dataclass
from typing import Optional, Callable, AsyncGenerator
import aiohttp
import websockets


@dataclass
class ConversationConfig:
    """Configuration for the conversational agent"""
    agent_id: Optional[str] = None  # Use pre-configured agent
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default voice
    model_id: str = "eleven_flash_v2_5"  # TTS model
    
    # LLM settings (if not using agent_id)
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and natural."
    llm_model: str = "gpt-4o-mini"  # or claude-3-haiku, gemini-1.5-flash
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # Voice settings
    stability: float = 0.5
    similarity_boost: float = 0.75
    
    # Turn-taking settings
    turn_detection_mode: str = "server_vad"  # server_vad or manual
    silence_threshold_ms: int = 500
    
    # Audio settings
    input_sample_rate: int = 16000
    output_sample_rate: int = 16000
    output_format: str = "pcm_16000"  # pcm_16000, pcm_22050, pcm_24000, mp3_44100


class ElevenLabsConversationalAgent:
    """
    ElevenLabs Conversational AI client.
    
    This uses the full Conversational AI API which provides:
    - Automatic speech recognition
    - Voice activity detection with smart turn-taking
    - LLM integration
    - Low-latency text-to-speech
    
    All in one WebSocket connection.
    """
    
    # WebSocket endpoint for Conversational AI
    CONV_AI_URL = "wss://api.elevenlabs.io/v1/convai/conversation"
    
    def __init__(
        self,
        api_key: str,
        config: Optional[ConversationConfig] = None
    ):
        self.api_key = api_key
        self.config = config or ConversationConfig()
        
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._is_connected = False
        self._conversation_id: Optional[str] = None
        
        # Event callbacks
        self.on_transcript: Optional[Callable[[str, bool], None]] = None  # text, is_final
        self.on_response_text: Optional[Callable[[str], None]] = None  # AI response text
        self.on_audio: Optional[Callable[[bytes], None]] = None  # Audio data
        self.on_agent_speaking: Optional[Callable[[bool], None]] = None  # is_speaking
        self.on_user_speaking: Optional[Callable[[bool], None]] = None  # is_speaking
        self.on_error: Optional[Callable[[str], None]] = None
        
        self._receive_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """
        Connect to ElevenLabs Conversational AI.
        
        Returns True if connection successful.
        """
        if self._is_connected:
            return True
        
        try:
            # Build connection URL
            url = self.CONV_AI_URL
            if self.config.agent_id:
                url += f"?agent_id={self.config.agent_id}"
            
            headers = {
                "xi-api-key": self.api_key
            }
            
            self._websocket = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Send initialization config
            init_config = self._build_init_config()
            await self._websocket.send(json.dumps(init_config))
            
            # Wait for connection acknowledgment
            response = await asyncio.wait_for(self._websocket.recv(), timeout=10)
            data = json.loads(response)
            
            if data.get("type") == "conversation_initiation_metadata":
                self._conversation_id = data.get("conversation_id")
                self._is_connected = True
                
                # Start receive loop
                self._receive_task = asyncio.create_task(self._receive_loop())
                
                return True
            else:
                print(f"Unexpected init response: {data}")
                return False
                
        except Exception as e:
            print(f"Connection error: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def _build_init_config(self) -> dict:
        """Build initialization configuration message"""
        config = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {
                "tts": {
                    "voice_id": self.config.voice_id,
                    "model_id": self.config.model_id,
                    "voice_settings": {
                        "stability": self.config.stability,
                        "similarity_boost": self.config.similarity_boost
                    },
                    "output_format": self.config.output_format
                },
                "turn_detection": {
                    "mode": self.config.turn_detection_mode,
                    "silence_threshold_ms": self.config.silence_threshold_ms
                }
            }
        }
        
        # If no agent_id, configure LLM
        if not self.config.agent_id:
            config["conversation_config_override"]["agent"] = {
                "prompt": {
                    "prompt": self.config.system_prompt
                },
                "llm": {
                    "model": self.config.llm_model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
            }
        
        return config
    
    async def _receive_loop(self):
        """Background task to receive and process messages"""
        try:
            async for message in self._websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            self._is_connected = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Receive error: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "user_transcript":
                # User speech transcribed
                text = data.get("user_transcript", "")
                is_final = data.get("is_final", False)
                if self.on_transcript:
                    self.on_transcript(text, is_final)
            
            elif msg_type == "agent_response":
                # AI response text
                text = data.get("agent_response", "")
                if self.on_response_text:
                    self.on_response_text(text)
            
            elif msg_type == "audio":
                # Audio data from TTS
                audio_b64 = data.get("audio", "")
                if audio_b64 and self.on_audio:
                    audio_bytes = base64.b64decode(audio_b64)
                    self.on_audio(audio_bytes)
            
            elif msg_type == "agent_speaking":
                # Agent started/stopped speaking
                is_speaking = data.get("is_speaking", False)
                if self.on_agent_speaking:
                    self.on_agent_speaking(is_speaking)
            
            elif msg_type == "user_speaking":
                # User started/stopped speaking (VAD)
                is_speaking = data.get("is_speaking", False)
                if self.on_user_speaking:
                    self.on_user_speaking(is_speaking)
            
            elif msg_type == "interruption":
                # User interrupted the agent
                pass  # Audio will stop automatically
            
            elif msg_type == "error":
                error_msg = data.get("message", "Unknown error")
                print(f"Agent error: {error_msg}")
                if self.on_error:
                    self.on_error(error_msg)
            
            elif msg_type == "ping":
                # Respond to ping
                await self._websocket.send(json.dumps({"type": "pong"}))
                
        except json.JSONDecodeError:
            # Might be binary audio data
            if isinstance(message, bytes) and self.on_audio:
                self.on_audio(message)
    
    async def send_audio(self, audio_chunk: bytes):
        """
        Send audio chunk to the agent.
        
        Args:
            audio_chunk: Raw PCM audio (16kHz, 16-bit, mono)
        """
        if not self._is_connected:
            return
        
        # Encode as base64
        audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
        
        message = {
            "type": "audio",
            "audio": audio_b64
        }
        
        await self._websocket.send(json.dumps(message))
    
    async def send_text(self, text: str):
        """
        Send text input (skip STT, go directly to LLM).
        
        Useful for testing or text-based interaction.
        """
        if not self._is_connected:
            await self.connect()
        
        message = {
            "type": "user_message",
            "text": text
        }
        
        await self._websocket.send(json.dumps(message))
    
    async def interrupt(self):
        """Interrupt the agent while speaking"""
        if not self._is_connected:
            return
        
        message = {"type": "interrupt"}
        await self._websocket.send(json.dumps(message))
    
    async def end_turn(self):
        """Manually signal end of user turn (if not using server VAD)"""
        if not self._is_connected:
            return
        
        message = {"type": "end_of_turn"}
        await self._websocket.send(json.dumps(message))
    
    async def disconnect(self):
        """Close the conversation"""
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
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def conversation_id(self) -> Optional[str]:
        return self._conversation_id


async def create_signed_url(api_key: str, agent_id: Optional[str] = None) -> Optional[str]:
    """
    Create a signed URL for client-side WebSocket connection.
    
    This allows the browser to connect directly to ElevenLabs
    without exposing the API key.
    """
    url = "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url"
    
    if agent_id:
        url += f"?agent_id={agent_id}"
    
    headers = {
        "xi-api-key": api_key
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("signed_url")
            else:
                error = await response.text()
                print(f"Failed to get signed URL: {error}")
                return None


async def create_agent(
    api_key: str,
    name: str = "Voice Assistant",
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational.",
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    llm_model: str = "claude-3-haiku",
    first_message: str = "Hello! How can I help you today?"
) -> Optional[str]:
    """
    Create an ElevenLabs Conversational AI agent dynamically.
    
    Returns the agent_id if successful, None otherwise.
    """
    url = "https://api.elevenlabs.io/v1/convai/agents/create"
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Use flash v2.5 for English which is required by ElevenLabs
    payload = {
        "name": name,
        "conversation_config": {
            "agent": {
                "prompt": {
                    "prompt": system_prompt,
                    "llm": llm_model,
                    "temperature": 0.7,
                    "max_tokens": 150
                },
                "first_message": first_message,
                "language": "en"
            },
            "tts": {
                "voice_id": voice_id,
                "model_id": "eleven_flash_v2_5"
            }
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                agent_id = data.get("agent_id")
                print(f"Created agent: {agent_id}")
                return agent_id
            else:
                error = await response.text()
                print(f"Failed to create agent: {response.status} - {error}")
                return None


async def get_or_create_agent(
    api_key: str,
    agent_id: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Get existing agent_id or create a new agent.
    
    If agent_id is provided and valid, returns it.
    Otherwise creates a new agent.
    """
    if agent_id:
        # Verify the agent exists
        url = f"https://api.elevenlabs.io/v1/convai/agents/{agent_id}"
        headers = {"xi-api-key": api_key}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return agent_id
                else:
                    print(f"Agent {agent_id} not found, creating new one...")
    
    # Create a new agent
    return await create_agent(api_key, **kwargs)
