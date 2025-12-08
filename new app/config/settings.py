"""
Application Settings

Central configuration for all application settings.
Loads from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class ServerConfig:
    """Web server configuration"""
    host: str = "localhost"
    port: int = 8080
    static_dir: str = "web/static"


@dataclass(frozen=True)
class ElevenLabsConfig:
    """ElevenLabs API configuration for STT and TTS"""
    api_key: str = ""
    stt_base_url: str = "wss://api.elevenlabs.io"
    tts_base_url: str = "https://api.elevenlabs.io"
    
    # STT settings
    stt_model: str = "scribe_v2_realtime"
    stt_language: str = "de"
    stt_sample_rate: int = 16000
    stt_commit_strategy: str = "manual"
    
    # TTS settings
    tts_model: str = "eleven_flash_v2_5"
    tts_output_format: str = "pcm_16000"
    voice_id: str = "j08ENmQlEinPmKqg3LUg"  # German voice
    voice_stability: float = 0.5
    voice_similarity_boost: float = 0.75
    voice_speed: float = 1.1


@dataclass(frozen=True)
class AnthropicConfig:
    """Anthropic Claude API configuration"""
    api_key: str = ""
    api_url: str = "https://api.anthropic.com/v1/messages"
    api_version: str = "2023-06-01"
    
    # Model settings
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Tool router model (for RAG decision)
    tool_router_model: str = "claude-haiku-4-5-20251001"
    tool_router_max_tokens: int = 50


@dataclass(frozen=True)
class RagConfig:
    """RAG (Retrieval-Augmented Generation) configuration"""
    documents_path: str = "data/products.md"
    
    # TF-IDF settings
    ngram_range: tuple = (1, 2)
    max_features: int = 5000
    sublinear_tf: bool = True
    
    # Search settings
    top_k: int = 5
    min_score_threshold: float = 0.01


@dataclass(frozen=True)
class VadConfig:
    """Voice Activity Detection configuration (client-side)"""
    threshold: float = 0.08
    speech_threshold: float = 0.03
    debounce_time_ms: int = 800
    silence_commit_delay_ms: int = 1200


@dataclass
class AgentConfig:
    """Voice agent behavior configuration"""
    # Trigger keywords for RAG (comma-separated, empty = always check)
    rag_trigger_keywords: str = (
        "produkt,empfehlen,sparen,ersparnisse,rente,pension,investieren,"
        "investition,vorsorge,säule,3a,3b,versicherung,vorschlag,option,plan,"
        "was haben sie,was bieten sie,product,recommend,save,savings,retire,"
        "retirement,invest,investment,pension,pillar,insurance,suggest"
    )
    
    # System prompt file path
    system_prompt_file: str = "config/prompts/system_prompt.md"
    
    # Default tasks for conversation tracking
    default_tasks: List[dict] = field(default_factory=lambda: [
        {"id": 1, "description": "Begrüssung und Vorstellung als Swiss Life Berater"},
        {"id": 2, "description": "Namen des Kunden erfragen"},
        {"id": 3, "description": "Versicherungssituation und finanzielle Ziele verstehen"},
        {"id": 4, "description": "Passendes Swiss Life Produkt empfehlen und erklären"},
        {"id": 5, "description": "Interesse an Angebot bestätigen"},
        {"id": 6, "description": "Notwendige Daten für Angebot erfassen"},
        {"id": 7, "description": "Nächste Schritte erklären und verabschieden"},
    ])


@dataclass
class AppSettings:
    """
    Main application settings container.
    
    Loads configuration from environment variables and provides
    sensible defaults for all settings.
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    @classmethod
    def load(cls) -> "AppSettings":
        """
        Load settings from environment variables.
        
        Returns:
            AppSettings instance with loaded configuration
            
        Raises:
            ValueError: If required environment variables are missing
        """
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        if not elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        return cls(
            server=ServerConfig(
                host=os.getenv("SERVER_HOST", "localhost"),
                port=int(os.getenv("SERVER_PORT", "8080")),
            ),
            elevenlabs=ElevenLabsConfig(
                api_key=elevenlabs_key,
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", "ogdlaxy0T9rCSVdH0VJM"),
                tts_model=os.getenv("TTS_MODEL_ID", "eleven_flash_v2_5"),
                stt_language=os.getenv("STT_LANGUAGE", "de"),
                stt_sample_rate=int(os.getenv("STT_SAMPLE_RATE", "16000")),
                voice_speed=float(os.getenv("TTS_SPEED", "1.1")),
            ),
            anthropic=AnthropicConfig(
                api_key=anthropic_key,
                model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            ),
            rag=RagConfig(
                top_k=int(os.getenv("RAG_TOP_K", "5")),
                max_features=int(os.getenv("RAG_MAX_FEATURES", "5000")),
            ),
            vad=VadConfig(
                threshold=float(os.getenv("VAD_THRESHOLD", "0.08")),
                silence_commit_delay_ms=int(os.getenv("VAD_SILENCE_DURATION_MS", "1200")),
                debounce_time_ms=int(os.getenv("VAD_DEBOUNCE_MS", "800")),
            ),
            agent=AgentConfig(),
        )


# Global settings instance (lazy loaded)
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """
    Get the global settings instance.
    
    Returns:
        AppSettings instance
    """
    global _settings
    if _settings is None:
        _settings = AppSettings.load()
    return _settings
