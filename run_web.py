#!/usr/bin/env python3
"""
Main entry point for the ElevenLabs Web Voice Agent.

This starts the web server that provides:
- A browser-based voice interface at http://localhost:8080
- WebSocket relay between browser and ElevenLabs Conversational AI API
- Real-time audio streaming with proprietary VAD

Usage:
    python run_web.py [--port PORT] [--host HOST]

Environment variables required:
    ELEVENLABS_API_KEY - Your ElevenLabs API key
    ANTHROPIC_API_KEY - Your Anthropic API key (for the LLM)
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Verify required environment variables are set."""
    required_vars = ['ELEVENLABS_API_KEY']
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        logger.error("Missing required environment variables:")
        for var in missing:
            logger.error(f"  - {var}")
        logger.error("\nPlease set these in your .env file or environment.")
        return False
    
    # Check optional but recommended
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.warning("ANTHROPIC_API_KEY not set - using ElevenLabs default LLM")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run the ElevenLabs Web Voice Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_web.py                    # Start on default port 8080
    python run_web.py --port 3000        # Start on port 3000
    python run_web.py --host 0.0.0.0     # Allow external connections
        """
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8080,
        help='Port to run the server on (default: 8080)'
    )
    parser.add_argument(
        '--host', '-H',
        type=str,
        default='localhost',
        help='Host to bind to (default: localhost)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Print startup banner
    print("\n" + "="*60)
    print("  🎙️  ElevenLabs Web Voice Agent")
    print("="*60)
    print(f"\n  Server starting on http://{args.host}:{args.port}")
    print(f"\n  Open this URL in your browser to start talking!")
    print("\n  Features:")
    print("    ✓ Real-time voice conversation")
    print("    ✓ ElevenLabs proprietary VAD (Voice Activity Detection)")
    print("    ✓ Ultra-low latency audio streaming")
    print("    ✓ Live transcription display")
    print("\n  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Import and run the server
    from web_server import WebServer
    
    server = WebServer(host=args.host, port=args.port)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
