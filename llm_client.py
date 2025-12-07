"""
LLM Client Module
=================

Streaming LLM clients for Anthropic Claude, OpenAI GPT, and Google Gemini.
All clients support streaming to minimize Time To First Token (TTFB).
"""

import asyncio
import json
import time
import aiohttp
from typing import AsyncGenerator, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    finish_reason: Optional[str] = None
    ttfb_ms: Optional[float] = None


class AnthropicLLMClient:
    """
    Anthropic Claude client with streaming.
    
    Why Claude:
    - Excellent instruction following (important for voice agents)
    - Good at being concise (we want short responses)
    - Safe/helpful balance
    
    Streaming protocol:
    - SSE (Server-Sent Events) over HTTP
    - Events: content_block_delta contains text tokens
    """
    
    API_URL = "https://api.anthropic.com/v1/messages"
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def stream_completion(
        self,
        messages: list[dict],
        system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion tokens.
        
        Yields tokens as they're generated for immediate TTS processing.
        This is the key to low latency - we don't wait for full response.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Anthropic separates system prompt from messages
        # Extract system from messages if present
        conversation = []
        extracted_system = system_prompt
        
        for m in messages:
            if m["role"] == "system":
                extracted_system = m["content"]
            else:
                conversation.append(m)
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,  # CRITICAL: Enable streaming
            "messages": conversation
        }
        
        if extracted_system:
            payload["system"] = extracted_system
        
        start_time = time.time()
        first_token = True
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.API_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Anthropic API error: {response.status} - {error}")
                
                # Process Server-Sent Events stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if not line or not line.startswith('data: '):
                        continue
                    
                    data = line[6:]  # Remove 'data: ' prefix
                    
                    if data == '[DONE]':
                        break
                    
                    try:
                        event = json.loads(data)
                        
                        # Look for text delta events
                        if event.get('type') == 'content_block_delta':
                            delta = event.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                text = delta.get('text', '')
                                
                                if first_token:
                                    ttfb = (time.time() - start_time) * 1000
                                    print(f"[LLM TTFB: {ttfb:.0f}ms]")
                                    first_token = False
                                
                                if text:
                                    yield text
                    
                    except json.JSONDecodeError:
                        continue


class OpenAILLMClient:
    """
    OpenAI GPT client with streaming.
    
    Similar to Anthropic but different message format.
    """
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def stream_completion(
        self,
        messages: list[dict],
        system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens from OpenAI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add system prompt if provided and not in messages
        all_messages = messages.copy()
        if system_prompt and not any(m["role"] == "system" for m in messages):
            all_messages.insert(0, {"role": "system", "content": system_prompt})
        
        payload = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True
        }
        
        start_time = time.time()
        first_token = True
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.API_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if not line or not line.startswith('data: '):
                        continue
                    
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    
                    try:
                        event = json.loads(data)
                        content = event.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        
                        if first_token and content:
                            ttfb = (time.time() - start_time) * 1000
                            print(f"[LLM TTFB: {ttfb:.0f}ms]")
                            first_token = False
                        
                        if content:
                            yield content
                    
                    except json.JSONDecodeError:
                        continue


class GeminiLLMClient:
    """
    Google Gemini client - fastest option.
    
    Why Gemini Flash:
    - Lowest TTFB (~150ms)
    - Good quality for conversational tasks
    - Competitive pricing
    """
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def stream_completion(
        self,
        messages: list[dict],
        system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens from Gemini"""
        url = self.API_URL.format(model=self.model) + f"?key={self.api_key}&alt=sse"
        
        # Convert to Gemini format
        contents = []
        system_instruction = system_prompt
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        start_time = time.time()
        first_token = True
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Gemini API error: {response.status} - {error}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if not line or not line.startswith('data: '):
                        continue
                    
                    data = line[6:]
                    
                    try:
                        event = json.loads(data)
                        
                        for candidate in event.get('candidates', []):
                            for part in candidate.get('content', {}).get('parts', []):
                                text = part.get('text', '')
                                
                                if first_token and text:
                                    ttfb = (time.time() - start_time) * 1000
                                    print(f"[LLM TTFB: {ttfb:.0f}ms]")
                                    first_token = False
                                
                                if text:
                                    yield text
                    
                    except json.JSONDecodeError:
                        continue


class FastLLMRouter:
    """
    Routes to fastest responding LLM (model racing).
    
    Why model racing:
    - LLM latency varies based on load
    - Some models may be slow or fail
    - First response wins = consistent low latency
    
    This is what ElevenLabs does internally with their hosted LLMs.
    """
    
    def __init__(self, clients: list, timeout_ms: int = 5000):
        self.clients = clients
        self.timeout = timeout_ms / 1000
    
    async def stream_completion(
        self,
        messages: list[dict],
        system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        """Race multiple LLM clients and stream from the fastest"""
        if not self.clients:
            raise ValueError("No LLM clients configured")
        
        # Create queues for each client
        queues = [asyncio.Queue() for _ in self.clients]
        
        async def stream_to_queue(client, queue):
            try:
                async for token in client.stream_completion(messages, system_prompt):
                    await queue.put(("token", token))
                await queue.put(("done", None))
            except Exception as e:
                await queue.put(("error", str(e)))
        
        # Start all clients racing
        tasks = [
            asyncio.create_task(stream_to_queue(client, queue))
            for client, queue in zip(self.clients, queues)
        ]
        
        # Wait for first token from any client
        winner_idx = None
        start_time = time.time()
        
        while winner_idx is None and (time.time() - start_time) < self.timeout:
            for idx, queue in enumerate(queues):
                try:
                    msg_type, content = queue.get_nowait()
                    if msg_type == "token" and content is not None:
                        winner_idx = idx
                        yield content
                        break
                    elif msg_type == "error":
                        continue  # Try other clients
                except asyncio.QueueEmpty:
                    continue
            
            if winner_idx is None:
                await asyncio.sleep(0.01)
        
        if winner_idx is None:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            raise TimeoutError("No LLM responded in time")
        
        # Cancel losers
        for idx, task in enumerate(tasks):
            if idx != winner_idx:
                task.cancel()
        
        # Continue streaming from winner
        winner_queue = queues[winner_idx]
        while True:
            msg_type, content = await winner_queue.get()
            if msg_type == "done" or msg_type == "error":
                break
            if content:
                yield content
        
        # Cleanup
        for task in tasks:
            if not task.done():
                task.cancel()
