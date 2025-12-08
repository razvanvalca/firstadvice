"""
Voice Agent Web Server

HTTP and WebSocket server for the voice agent application.
"""

import asyncio
import json
import base64
from pathlib import Path

import aiohttp
from aiohttp import web

from config.settings import AppSettings
from core.session import VoiceAgentSession
from services.product_rag import initialize_rag_service


class VoiceAgentServer:
    """
    Web server for the voice agent application.
    
    Handles:
    - Static file serving
    - WebSocket connections for voice sessions
    - RAG initialization at startup
    """
    
    def __init__(self, settings: AppSettings):
        """
        Initialize the server.
        
        Args:
            settings: Application settings
        """
        self._settings = settings
        self._product_summary = ""
        
        # Create aiohttp application
        self._app = web.Application()
        self._setup_routes()
        self._app.on_startup.append(self._on_startup)
    
    def _setup_routes(self) -> None:
        """Configure HTTP routes."""
        static_path = Path(__file__).parent / "static"
        
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/ws", self._handle_websocket)
        
        if static_path.exists():
            self._app.router.add_static("/static", static_path)
    
    async def _on_startup(self, app: web.Application) -> None:
        """Initialize services at startup."""
        print("[Server] Initializing RAG system...")
        rag = await initialize_rag_service(self._settings.rag)
        self._product_summary = rag.product_summary
        print(f"[Server] RAG ready with {rag.chunk_count} product chunks")
    
    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the main page."""
        html_path = Path(__file__).parent / "static" / "index.html"
        if html_path.exists():
            return web.FileResponse(html_path)
        return web.Response(text="index.html not found", status=404)
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection from browser."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        print("[Server] Client connected")
        
        # Create session
        session = VoiceAgentSession(
            websocket=ws,
            settings=self._settings,
            product_summary=self._product_summary,
        )
        
        if not await session.start():
            return ws
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(session, ws, json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[Server] WebSocket error: {ws.exception()}")
                    break
        finally:
            await session.stop()
            print("[Server] Client disconnected")
        
        return ws
    
    async def _handle_message(
        self,
        session: VoiceAgentSession,
        ws: web.WebSocketResponse,
        data: dict,
    ) -> None:
        """
        Handle a WebSocket message from the browser.
        
        Args:
            session: The voice agent session
            ws: WebSocket connection
            data: Parsed message data
        """
        msg_type = data.get("type")
        
        if msg_type == "audio":
            # Audio chunk from microphone
            audio_b64 = data.get("audio", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                await session.handle_audio(audio_bytes)
        
        elif msg_type == "commit":
            # Browser VAD detected silence
            print("[Server] Browser VAD: Committing transcription")
            await session.handle_commit()
        
        elif msg_type == "user_speaking":
            # Browser detected user speaking during playback
            if data.get("interrupted"):
                print("[Server] Browser VAD: User interrupted agent")
                session.handle_user_speaking_interrupt()
        
        elif msg_type == "audio_status":
            # Browser reports audio playback status
            playing = data.get("playing", False)
            if playing:
                print("[Server] Audio status: playing")
            session.handle_audio_status(playing)
        
        elif msg_type == "config":
            # Configuration update
            session.configure(
                system_prompt=data.get("system_prompt"),
                tasks=data.get("tasks"),
            )
            
            # Send tasks list if configured
            if data.get("tasks"):
                await ws.send_json({
                    "type": "tasks",
                    "data": session.get_tasks(),
                })
            
            await ws.send_json({"type": "status", "data": "config_updated"})
        
        elif msg_type == "clear_history":
            # Clear conversation history
            session._state.conversation_history = []
            await ws.send_json({"type": "status", "data": "history_cleared"})
    
    def run(self) -> None:
        """Start the server (blocking)."""
        host = self._settings.server.host
        port = self._settings.server.port
        model = self._settings.anthropic.model
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║         Swiss Life Voice Insurance Advisor                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Open in browser: http://{host}:{port}                        ║
║                                                              ║
║  Components:                                                 ║
║    • STT: ElevenLabs Scribe Realtime                        ║
║    • LLM: Claude ({model[:20]}...)     ║
║    • TTS: ElevenLabs Flash v2.5                             ║
║    • RAG: TF-IDF (scikit-learn)                             ║
║    • VAD: Client-side (browser)                             ║
║                                                              ║
║  Just speak naturally - no button needed!                    ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        web.run_app(self._app, host=host, port=port, print=None)

    async def run_async(self) -> None:
        """Start the server (async, for use in existing event loop)."""
        host = self._settings.server.host
        port = self._settings.server.port
        model = self._settings.anthropic.model
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║         Swiss Life Voice Insurance Advisor                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Open in browser: http://{host}:{port}                        ║
║                                                              ║
║  Components:                                                 ║
║    • STT: ElevenLabs Scribe Realtime                        ║
║    • LLM: Claude ({model[:20]}...)     ║
║    • TTS: ElevenLabs Flash v2.5                             ║
║    • RAG: TF-IDF (scikit-learn)                             ║
║    • VAD: Client-side (browser)                             ║
║                                                              ║
║  Just speak naturally - no button needed!                    ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        # Keep running until cancelled
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep for an hour
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()
