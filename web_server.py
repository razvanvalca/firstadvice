"""
Web Server for Voice Agent
==========================

Provides:
1. HTTP server for the web interface
2. WebSocket endpoint to relay audio between browser and ElevenLabs
3. Signed URL generation for direct browser-to-ElevenLabs connection
"""

import asyncio
import json
import os
import base64
from pathlib import Path
from aiohttp import web
import aiohttp
from dotenv import load_dotenv

from elevenlabs_agent import ElevenLabsConversationalAgent, ConversationConfig, create_signed_url, get_or_create_agent

load_dotenv()

# Get configuration from environment
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")  # Will be auto-created if not set
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Determine LLM to use - ElevenLabs model IDs (not Anthropic API IDs)
if ANTHROPIC_API_KEY:
    LLM_MODEL = "claude-3-haiku"  # ElevenLabs uses short model names
elif OPENAI_API_KEY:
    LLM_MODEL = "gpt-4o-mini"
else:
    LLM_MODEL = "gpt-4o-mini"  # Default, will use ElevenLabs' hosted LLM


class WebServer:
    """Web server for the voice agent interface"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self._setup_routes()
        
        # Active WebSocket connections
        self.connections = {}
        
        # Agent ID (will be set on first connection if not configured)
        self.agent_id = ELEVENLABS_AGENT_ID
        self._agent_id_lock = asyncio.Lock()
    
    def _setup_routes(self):
        """Set up HTTP and WebSocket routes"""
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/api/signed-url", self._handle_signed_url)
        self.app.router.add_get("/api/config", self._handle_config)
        self.app.router.add_get("/ws", self._handle_websocket)
        
        # Serve static files
        static_path = Path(__file__).parent / "static"
        if static_path.exists():
            self.app.router.add_static("/static", static_path)
    
    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the main HTML page"""
        html_path = Path(__file__).parent / "static" / "index.html"
        
        if html_path.exists():
            return web.FileResponse(html_path)
        else:
            # Return embedded HTML if static file doesn't exist
            return web.Response(
                text=self._get_embedded_html(),
                content_type="text/html"
            )
    
    async def _handle_signed_url(self, request: web.Request) -> web.Response:
        """Generate signed URL for direct browser connection"""
        if not ELEVENLABS_API_KEY:
            return web.json_response(
                {"error": "ELEVENLABS_API_KEY not configured"},
                status=500
            )
        
        signed_url = await create_signed_url(
            ELEVENLABS_API_KEY,
            ELEVENLABS_AGENT_ID
        )
        
        if signed_url:
            return web.json_response({"signed_url": signed_url})
        else:
            return web.json_response(
                {"error": "Failed to generate signed URL"},
                status=500
            )
    
    async def _handle_config(self, request: web.Request) -> web.Response:
        """Return client configuration"""
        return web.json_response({
            "voice_id": ELEVENLABS_VOICE_ID,
            "agent_id": ELEVENLABS_AGENT_ID,
            "has_api_key": bool(ELEVENLABS_API_KEY)
        })
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """
        Handle WebSocket connection from browser.
        
        Acts as a relay between browser and ElevenLabs Conversational AI.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        connection_id = id(ws)
        print(f"[WS] Client connected: {connection_id}")
        
        # Ensure we have an agent_id (create one if needed)
        async with self._agent_id_lock:
            if not self.agent_id:
                print("[WS] No agent configured, creating one...")
                self.agent_id = await get_or_create_agent(
                    api_key=ELEVENLABS_API_KEY,
                    name="Voice Assistant",
                    system_prompt="""You are a helpful, friendly voice assistant. 
Keep your responses concise and natural - ideally 1-2 sentences.
Speak conversationally as if talking to a friend.""",
                    voice_id=ELEVENLABS_VOICE_ID,
                    llm_model=LLM_MODEL,
                    first_message="Hello! How can I help you today?"
                )
                if self.agent_id:
                    print(f"[WS] Created agent: {self.agent_id}")
                else:
                    await ws.send_json({"type": "error", "message": "Failed to create ElevenLabs agent"})
                    await ws.close()
                    return ws
        
        # Create ElevenLabs agent for this connection
        config = ConversationConfig(
            agent_id=self.agent_id,
            voice_id=ELEVENLABS_VOICE_ID,
            system_prompt="""You are a helpful, friendly voice assistant. 
Keep your responses concise and natural - ideally 1-2 sentences.
Speak conversationally as if talking to a friend.""",
            llm_model=LLM_MODEL,
            output_format="pcm_16000"
        )
        
        agent = ElevenLabsConversationalAgent(ELEVENLABS_API_KEY, config)
        self.connections[connection_id] = {"ws": ws, "agent": agent}
        
        # Set up agent callbacks
        async def send_to_client(msg_type: str, data: dict):
            try:
                await ws.send_json({"type": msg_type, **data})
            except:
                pass
        
        agent.on_transcript = lambda text, final: asyncio.create_task(
            send_to_client("transcript", {"text": text, "is_final": final})
        )
        agent.on_response_text = lambda text: asyncio.create_task(
            send_to_client("response", {"text": text})
        )
        agent.on_audio = lambda audio: asyncio.create_task(
            send_to_client("audio", {"audio": base64.b64encode(audio).decode()})
        )
        agent.on_agent_speaking = lambda speaking: asyncio.create_task(
            send_to_client("agent_speaking", {"is_speaking": speaking})
        )
        agent.on_user_speaking = lambda speaking: asyncio.create_task(
            send_to_client("user_speaking", {"is_speaking": speaking})
        )
        agent.on_error = lambda error: asyncio.create_task(
            send_to_client("error", {"message": error})
        )
        
        try:
            # Connect to ElevenLabs
            connected = await agent.connect()
            if connected:
                await ws.send_json({"type": "connected", "conversation_id": agent.conversation_id})
            else:
                await ws.send_json({"type": "error", "message": "Failed to connect to ElevenLabs"})
                return ws
            
            # Handle messages from browser
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")
                    
                    if msg_type == "audio":
                        # Relay audio to ElevenLabs
                        audio_b64 = data.get("audio", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            await agent.send_audio(audio_bytes)
                    
                    elif msg_type == "text":
                        # Send text message
                        text = data.get("text", "")
                        if text:
                            await agent.send_text(text)
                    
                    elif msg_type == "interrupt":
                        await agent.interrupt()
                    
                    elif msg_type == "end_turn":
                        await agent.end_turn()
                
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary audio data
                    await agent.send_audio(msg.data)
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[WS] Error: {ws.exception()}")
                    break
        
        except Exception as e:
            print(f"[WS] Error: {e}")
        
        finally:
            # Cleanup
            await agent.disconnect()
            if connection_id in self.connections:
                del self.connections[connection_id]
            print(f"[WS] Client disconnected: {connection_id}")
        
        return ws
    
    def _get_embedded_html(self) -> str:
        """Return embedded HTML if static file not found"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Voice Agent</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container { 
            max-width: 600px; 
            width: 90%; 
            text-align: center;
            padding: 2rem;
        }
        h1 { margin-bottom: 2rem; font-weight: 300; }
        .status { 
            margin: 1rem 0; 
            padding: 0.5rem 1rem;
            border-radius: 20px;
            background: rgba(255,255,255,0.1);
            display: inline-block;
        }
        .status.connected { background: rgba(76, 175, 80, 0.3); }
        .status.speaking { background: rgba(33, 150, 243, 0.3); }
        .mic-button {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 2rem;
            cursor: pointer;
            margin: 2rem 0;
            transition: all 0.3s ease;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        }
        .mic-button:hover { transform: scale(1.05); }
        .mic-button.active { 
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            animation: pulse 1.5s infinite;
        }
        .mic-button:disabled { opacity: 0.5; cursor: not-allowed; }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(245, 87, 108, 0.5); }
            50% { box-shadow: 0 0 0 20px rgba(245, 87, 108, 0); }
        }
        .transcript {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            min-height: 60px;
            text-align: left;
        }
        .transcript.user { border-left: 3px solid #667eea; }
        .transcript.agent { border-left: 3px solid #f5576c; }
        .label { font-size: 0.8rem; color: #888; margin-bottom: 0.5rem; }
        .visualizer {
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
        }
        .bar {
            width: 4px;
            background: #667eea;
            border-radius: 2px;
            transition: height 0.1s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice Agent</h1>
        
        <div class="status" id="status">Connecting...</div>
        
        <div class="visualizer" id="visualizer">
            <div class="bar" style="height: 20px"></div>
            <div class="bar" style="height: 35px"></div>
            <div class="bar" style="height: 25px"></div>
            <div class="bar" style="height: 40px"></div>
            <div class="bar" style="height: 30px"></div>
        </div>
        
        <button class="mic-button" id="micButton" disabled>🎤</button>
        
        <div class="transcript user" id="userTranscript">
            <div class="label">You</div>
            <div id="userText">-</div>
        </div>
        
        <div class="transcript agent" id="agentTranscript">
            <div class="label">Agent</div>
            <div id="agentText">-</div>
        </div>
    </div>
    
    <script>
        let ws;
        let audioContext;
        let mediaStream;
        let processor;
        let isRecording = false;
        let audioQueue = [];
        let isPlaying = false;
        
        const micButton = document.getElementById('micButton');
        const status = document.getElementById('status');
        const userText = document.getElementById('userText');
        const agentText = document.getElementById('agentText');
        const bars = document.querySelectorAll('.bar');
        
        // Connect to WebSocket
        function connect() {
            ws = new WebSocket(`ws://${location.host}/ws`);
            
            ws.onopen = () => {
                status.textContent = 'Connected';
                status.classList.add('connected');
            };
            
            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                
                switch(data.type) {
                    case 'connected':
                        micButton.disabled = false;
                        status.textContent = 'Ready - Click to speak';
                        break;
                    
                    case 'transcript':
                        userText.textContent = data.text;
                        break;
                    
                    case 'response':
                        agentText.textContent = data.text;
                        break;
                    
                    case 'audio':
                        const audioData = base64ToArrayBuffer(data.audio);
                        playAudio(audioData);
                        break;
                    
                    case 'agent_speaking':
                        if (data.is_speaking) {
                            status.textContent = 'Agent speaking...';
                            status.classList.add('speaking');
                        } else {
                            status.textContent = 'Ready - Click to speak';
                            status.classList.remove('speaking');
                        }
                        break;
                    
                    case 'user_speaking':
                        if (data.is_speaking) {
                            status.textContent = 'Listening...';
                        }
                        break;
                    
                    case 'error':
                        status.textContent = 'Error: ' + data.message;
                        console.error(data.message);
                        break;
                }
            };
            
            ws.onclose = () => {
                status.textContent = 'Disconnected';
                status.classList.remove('connected');
                micButton.disabled = true;
                setTimeout(connect, 2000);
            };
            
            ws.onerror = (e) => {
                console.error('WebSocket error:', e);
            };
        }
        
        // Microphone handling
        micButton.onclick = async () => {
            if (!isRecording) {
                await startRecording();
            } else {
                stopRecording();
            }
        };
        
        async function startRecording() {
            try {
                audioContext = new AudioContext({ sampleRate: 16000 });
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!isRecording) return;
                    
                    const input = e.inputBuffer.getChannelData(0);
                    const pcm = floatTo16BitPCM(input);
                    
                    // Update visualizer
                    updateVisualizer(input);
                    
                    // Send to server
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'audio',
                            audio: arrayBufferToBase64(pcm)
                        }));
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                micButton.classList.add('active');
                micButton.textContent = '⏹️';
                status.textContent = 'Listening...';
                
            } catch (e) {
                console.error('Microphone error:', e);
                status.textContent = 'Microphone access denied';
            }
        }
        
        function stopRecording() {
            isRecording = false;
            
            if (processor) processor.disconnect();
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
            if (audioContext) audioContext.close();
            
            micButton.classList.remove('active');
            micButton.textContent = '🎤';
            status.textContent = 'Ready - Click to speak';
            
            // Signal end of turn
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'end_turn' }));
            }
        }
        
        // Audio playback
        function playAudio(pcmData) {
            audioQueue.push(pcmData);
            if (!isPlaying) processAudioQueue();
        }
        
        async function processAudioQueue() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }
            
            isPlaying = true;
            const pcmData = audioQueue.shift();
            
            const ctx = new AudioContext({ sampleRate: 16000 });
            const audioBuffer = ctx.createBuffer(1, pcmData.byteLength / 2, 16000);
            const channel = audioBuffer.getChannelData(0);
            const view = new Int16Array(pcmData);
            
            for (let i = 0; i < view.length; i++) {
                channel[i] = view[i] / 32768;
            }
            
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);
            source.onended = () => {
                ctx.close();
                processAudioQueue();
            };
            source.start();
        }
        
        // Utility functions
        function floatTo16BitPCM(input) {
            const buffer = new ArrayBuffer(input.length * 2);
            const view = new DataView(buffer);
            for (let i = 0; i < input.length; i++) {
                const s = Math.max(-1, Math.min(1, input[i]));
                view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            return buffer;
        }
        
        function arrayBufferToBase64(buffer) {
            const bytes = new Uint8Array(buffer);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            return btoa(binary);
        }
        
        function base64ToArrayBuffer(base64) {
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            return bytes.buffer;
        }
        
        function updateVisualizer(input) {
            const rms = Math.sqrt(input.reduce((a, b) => a + b * b, 0) / input.length);
            const level = Math.min(1, rms * 10);
            
            bars.forEach((bar, i) => {
                const height = 10 + level * 50 * (0.5 + Math.random() * 0.5);
                bar.style.height = height + 'px';
            });
        }
        
        // Start
        connect();
    </script>
</body>
</html>"""
    
    async def start(self):
        """Start the web server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"\n{'='*60}")
        print(f"Voice Agent Web Interface")
        print(f"{'='*60}")
        print(f"\n🌐 Open in browser: http://localhost:{self.port}")
        print(f"\nPress Ctrl+C to stop\n")
        
        # Keep running
        while True:
            await asyncio.sleep(3600)
    
    async def run(self):
        """Alias for start() - runs the web server"""
        await self.start()


async def main():
    """Main entry point"""
    if not ELEVENLABS_API_KEY:
        print("ERROR: ELEVENLABS_API_KEY not set in environment")
        print("Set it in your .env file")
        return
    
    server = WebServer(port=8080)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
