#!/usr/bin/env python3
"""
Swiss Life Voice Insurance Advisor - Main Entry Point

This module serves as the application entry point, loading configuration
and starting the voice agent server.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from config.settings import AppSettings, get_settings
from web.server import VoiceAgentServer


def setup_logging(debug: bool = False) -> None:
    """Configure application logging.
    
    Args:
        debug: If True, set log level to DEBUG; otherwise INFO.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def print_banner() -> None:
    """Display application startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     🎙️  Swiss Life Voice Insurance Advisor                        ║
    ║                                                                   ║
    ║     Real-time voice-based insurance consultation                  ║
    ║     Powered by ElevenLabs + Claude + TF-IDF RAG                   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_environment(settings: AppSettings) -> bool:
    """Validate that required environment variables are set.
    
    Args:
        settings: Application settings instance.
        
    Returns:
        True if all required variables are set, False otherwise.
    """
    errors = []
    
    if not settings.elevenlabs.api_key:
        errors.append("ELEVENLABS_API_KEY is not set")
    
    if not settings.anthropic.api_key:
        errors.append("ANTHROPIC_API_KEY is not set")
    
    if errors:
        logging.error("Environment validation failed:")
        for error in errors:
            logging.error(f"  - {error}")
        logging.error("\nPlease set the required environment variables in your .env file.")
        logging.error("See .env.example for reference.")
        return False
    
    return True


async def main() -> int:
    """Main application entry point.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Load settings from environment
    settings = get_settings()
    
    # Setup logging
    debug = os.getenv("DEBUG", "false").lower() == "true"
    setup_logging(debug=debug)
    
    logger = logging.getLogger(__name__)
    
    # Print startup banner
    print_banner()
    
    # Validate environment
    if not validate_environment(settings):
        return 1
    
    logger.info("Starting Swiss Life Voice Insurance Advisor...")
    logger.info(f"Server will be available at http://{settings.server.host}:{settings.server.port}")
    
    # Create and run server
    server = VoiceAgentServer(settings)
    
    try:
        await server.run_async()
    except KeyboardInterrupt:
        logger.info("Shutdown requested via keyboard interrupt")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        return 1
    
    logger.info("Server stopped. Goodbye!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested. Goodbye!")
        sys.exit(0)
