"""
Anthropic Claude LLM Service

Streaming LLM responses with prompt caching support.
"""

import json
from typing import AsyncIterator

import aiohttp

from config.settings import AnthropicConfig


class LlmService:
    """
    LLM service using Anthropic Claude API.
    
    Provides streaming responses with prompt caching for efficiency.
    """
    
    def __init__(self, config: AnthropicConfig):
        """
        Initialize the LLM service.
        
        Args:
            config: Anthropic API configuration
        """
        self._config = config
    
    async def generate_stream(
        self,
        messages: list,
        system_prompt: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: Conversation history
            system_prompt: System prompt for the agent
            temperature: Sampling temperature (default from config)
            max_tokens: Maximum tokens to generate (default from config)
            
        Yields:
            Response tokens as they are generated
        """
        headers = {
            "x-api-key": self._config.api_key,
            "anthropic-version": self._config.api_version,
            "anthropic-beta": "prompt-caching-2024-07-31",
            "content-type": "application/json",
        }
        
        payload = {
            "model": self._config.model,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature or self._config.temperature,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
            "stream": True,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._config.api_url,
                headers=headers,
                json=payload,
            ) as response:
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
    
    async def classify_intent(
        self,
        user_message: str,
        system_prompt: str,
    ) -> str:
        """
        Quick LLM call for intent classification (non-streaming).
        
        Used for tool routing decisions (e.g., should we search RAG?).
        
        Args:
            user_message: The user's message to classify
            system_prompt: Classification instructions
            
        Returns:
            Classification result as string
        """
        headers = {
            "x-api-key": self._config.api_key,
            "anthropic-version": self._config.api_version,
            "content-type": "application/json",
        }
        
        payload = {
            "model": self._config.tool_router_model,
            "max_tokens": self._config.tool_router_max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.api_url,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["content"][0]["text"].strip()
                    else:
                        print(f"[LLM] Classification failed: {response.status}")
                        return ""
        except Exception as e:
            print(f"[LLM] Classification error: {e}")
            return ""
