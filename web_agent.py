"""
ElevenLabs Conversational AI Web Agent

This provides a web interface for ElevenLabs Conversational AI.
The browser connects directly to ElevenLabs using a signed URL.

Prerequisites:
1. Create an agent in the ElevenLabs dashboard: https://elevenlabs.io/app/conversational-ai
2. Add the agent ID to your .env file as ELEVENLABS_AGENT_ID

Usage:
    python web_agent.py
"""

import os
import asyncio
from pathlib import Path
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()


class WebAgentServer:
    """Simple web server that provides signed URLs for ElevenLabs Conversational AI."""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.agent_id = os.getenv("ELEVENLABS_AGENT_ID", "")
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")
        if not self.agent_id:
            raise ValueError(
                "ELEVENLABS_AGENT_ID not found in environment.\n"
                "Please create an agent at https://elevenlabs.io/app/conversational-ai\n"
                "and add ELEVENLABS_AGENT_ID=your_agent_id to your .env file"
            )
        
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up HTTP routes"""
        static_path = Path(__file__).parent / "static"
        
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/api/config", self._handle_config)
        self.app.router.add_static("/static", static_path, name="static")
    
    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the main HTML page"""
        static_path = Path(__file__).parent / "static" / "agent.html"
        if static_path.exists():
            return web.FileResponse(static_path)
        else:
            return web.Response(text="agent.html not found", status=404)
    
    async def _handle_config(self, request: web.Request) -> web.Response:
        """Return agent configuration for the frontend"""
        # Return the agent_id - the frontend will connect directly to ElevenLabs
        return web.json_response({
            "agent_id": self.agent_id
        })
    
    async def start(self):
        """Start the web server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║          ElevenLabs Voice Agent - Web Interface              ║
╠══════════════════════════════════════════════════════════════╣
║  Open in browser: http://{self.host}:{self.port}                        ║
║                                                              ║
║  Agent ID: {self.agent_id[:20]}...                       ║
║                                                              ║
║  The browser connects directly to ElevenLabs using their     ║
║  Conversational AI WebSocket API.                            ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()
    
    def run(self):
        """Run the server (blocking)"""
        asyncio.run(self.start())


def main():
    """Main entry point"""
    try:
        server = WebAgentServer()
        server.run()
    except ValueError as e:
        print(f"\n❌ Configuration Error:\n{e}")
        print("\nSteps to fix:")
        print("1. Go to https://elevenlabs.io/app/conversational-ai")
        print("2. Create a new agent with your desired voice and settings")
        print("3. Copy the Agent ID from the agent settings")
        print("4. Add to your .env file: ELEVENLABS_AGENT_ID=your_agent_id_here")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0


if __name__ == "__main__":
    exit(main())
