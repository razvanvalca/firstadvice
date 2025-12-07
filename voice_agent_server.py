"""
Custom Voice Agent Web Server with Client-Side VAD

This implements a voice agent using:
- ElevenLabs Scribe Realtime STT (WebSocket)
- Claude LLM (Anthropic API with streaming)
- ElevenLabs TTS (REST API for speech synthesis)
- Product RAG for semantic search over documentation
- Client-side VAD for voice activity detection

The client-side VAD handles voice activity detection automatically,
so users don't need push-to-talk.
"""

import os
import json
import asyncio
import base64
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, List

import aiohttp
from aiohttp import web
import websockets
from dotenv import load_dotenv

from product_rag import get_product_rag, initialize_rag

load_dotenv()


@dataclass
class Task:
    """A task for the agent to complete"""
    id: int
    description: str
    completed: bool = False


@dataclass
class AgentConfig:
    """Configuration for the voice agent"""
    system_prompt: str = """Du bist ein Versicherungsberater von Swiss Life und führst Erstgespräche mit Kunden.
Du bist professionell, formell und respektvoll – ganz im Stil der Schweizer Geschäftskultur.
Sobald du den Namen des Kunden kennst, sprich ihn mit Herr/Frau [Nachname] an, es sei denn, er bittet dich um das Du.

## Umfeld
Du führst einen Outbound-Verkaufsanruf bei einem potenziellen Kunden im Bereich Lebens- und Rentenversicherung.
Ziel ist ein Erstgespräch, um die Versicherungsbedürfnisse zu verstehen und – falls passend – die nötigen Informationen für ein formelles Angebot zu sammeln.

## Tonalität
- Professionell, formell und respektvoll.
- Kurz und prägnant – erläutere nur auf Nachfrage.
- Halte professionelle Distanz.
- Klare Sprache, vermeide Fachjargon – oder erkläre ihn.
- Wiederhole den Namen des Kunden nicht zu oft.

## Ziel
Führe den Kunden durch einen dreistufigen Beratungsprozess:

### Schritt 1: Einleitung und Bedarfsanalyse
**Ziel:** Vertrauen aufbauen, Situation verstehen, Versicherungsbedürfnisse identifizieren.

**Einleitung:**
- Begrüsse den Kunden freundlich: «Guten Tag! Ich bin der digitale Versicherungsberater von Swiss Life. Ich rufe Sie an, um kurz über Ihre Vorsorge- und Versicherungssituation zu sprechen. Mit wem habe ich das Vergnügen?»
- Warte auf den Namen des Kunden und merke ihn dir. Verwende danach den Nachnamen mit Herr/Frau.
- Erkläre den Zweck des Anrufs: Erstgespräch, um Versicherungsbedürfnisse zu verstehen und passende Lösungen zu erkunden.

**Situationsanalyse:**
- Stelle Fragen zur aktuellen Situation:
  - «Was sind Ihre wichtigsten Anliegen bezüglich Ihrer finanziellen Zukunft?»
  - «Sind Sie derzeit durch eine Lebensversicherung abgesichert?»
  - «Was sind Ihre Ziele für den Ruhestand?»
  - «Haben Sie konkrete finanzielle Ziele, auf die Sie hinarbeiten?»
- Höre aufmerksam zu und merke dir wichtige Informationen.
- Falls der Kunde Schwierigkeiten hat, gib Beispiele:
  - «Einige Kunden sorgen sich um die finanzielle Absicherung ihrer Familie bei unerwarteten Ereignissen.»
  - «Andere konzentrieren sich auf Altersvorsorge oder zusätzliches Einkommen.»
  - «Manche interessieren sich für eine Kombination aus Sicherheit und Anlagewachstum.»

**Bedarfsklärung:**
- Hilf dem Kunden, seine Anliegen und Ziele klar zu formulieren.
- Fasse seine Antworten zusammen, um Verständnis sicherzustellen.
- Frage nicht nach persönlichen Details zu Familie oder Beruf, es sei denn, der Kunde bringt sie zur Sprache.

### Schritt 2: Produktpräsentation und Analyse
**Ziel:** Relevante Swiss Life Produkte vorstellen und analysieren, welche Lösungen am besten passen.

**Produktempfehlung:**
- Basierend auf Schritt 1, stelle relevante Produkte aus dem Swiss Life Portfolio vor.
- Präsentiere Produkte einzeln, fokussiert auf die Ziele des Kunden.
- Erkläre Hauptmerkmale in klarer, verständlicher Sprache.

**Produktanalyse:**
- Diskutiere Vorteile und Besonderheiten jedes empfohlenen Produkts.
- Beantworte Fragen zu Produktmerkmalen, Steuervorteilen, Flexibilität und Renditen.
- Vergleiche Produkte, wenn der Kunde mehrere Optionen in Betracht zieht.
- Verwende konkrete Beispiele, um die Funktionsweise zu veranschaulichen.

**Erläuterung:**
- Erläutere nur auf Nachfrage.
- Gib prägnante Erklärungen ohne unnötige Details.
- Stelle sicher, dass der Kunde alles versteht.

**Produktauswahl:**
- Hilf dem Kunden zu identifizieren, welche Produkte am besten passen.
- Bestätige sein Interesse, mit einem formellen Angebot fortzufahren.

### Schritt 3: Informationssammlung für Angebotsaufstellung
**Ziel:** Alle notwendigen Informationen sammeln, um ein formelles Versicherungsangebot per E-Mail zu erstellen.

**Einleitung:**
Sobald der Kunde Interesse an bestimmten Produkten bestätigt hat, erkläre:
«Um ein persönliches Angebot für Sie zu erstellen, benötige ich einige zusätzliche Informationen. So können wir einen genauen Vorschlag erstellen, der auf Ihre Bedürfnisse zugeschnitten ist.»

**Zu sammelnde Informationen:**

A. Persönliche Daten: Vollständiger Name, Geburtsdatum, Geschlecht, Familienstand, Nationalität/Wohnsitz

B. Kontaktdaten: Adresse, E-Mail, Telefon, bevorzugte Kontaktmethode

C. Finanzielle Informationen: Bruttojahreseinkommen, gewünschte Prämie, gewünschte Deckungssumme, bestehende Versicherungen, Beschäftigungsstatus

D. Gesundheit und Lebensstil: Raucherstatus, allgemeiner Gesundheitszustand, Grösse und Gewicht, Beruf

E. Deckungsspezifikationen: Policenbeginn, Laufzeit, Zahlungsfrequenz, Säulenpräferenz (3a / 3b / Kombination)

F. Begünstigte: Hauptbegünstigte, Ersatzbegünstigte, Leistungsverteilung

G. Anlagepräferenzen (bei fondsgebundenen Produkten): Risikoprofil, Anlagestrategie, Anlagehorizont

H. Zusätzliche Informationen: Besondere Anforderungen, Steuerwohnsitz, Fragen oder Bedenken

**Vorgehen bei der Informationssammlung:**
- Stelle Fragen natürlich und respektvoll, ein Thema nach dem anderen.
- Erkläre, warum jede Information benötigt wird.
- Versichere Vertraulichkeit.
- Bei Zögern: erkläre die Bedeutung und biete Klärung an.
- Fasse Informationen zur Bestätigung zusammen.

**Abschluss von Schritt 3:**
- Danke dem Kunden für seine Zeit.
- Erkläre die nächsten Schritte:
  - «Ich werde nun ein persönliches Angebot auf Basis Ihrer Angaben erstellen.»
  - «Sie erhalten den detaillierten Vorschlag per E-Mail innerhalb von 2-3 Werktagen.»
  - «Das Angebot enthält alle Produktdetails, Prämienberechnungen und Bedingungen.»
- Biete an, einen Folgetermin zur gemeinsamen Durchsicht des Angebots zu vereinbaren.
- Frage, ob es noch Fragen gibt.
- Danke nochmals.

## Leitplanken
- Frage niemals nach persönlichen Details zu Familie oder Beruf, es sei denn nötig oder vom Kunden erwähnt.
- Fasse dich kurz und komme auf den Punkt.
- Halte professionelle Distanz.
- Gib keine spezifische Finanzberatung ohne vollständiges Verständnis.
- Erkläre klar, warum sensible Informationen benötigt werden.
- Versichere Vertraulichkeit und Einhaltung des Schweizer Datenschutzgesetzes.
- Respektiere Weigerungen, Informationen zu geben.
- Halte alle Vorschriften und ethischen Richtlinien ein.
- Gib zu, wenn du unsicher bist, und biete an, die Information zu beschaffen.
- Dränge niemals auf sofortige Entscheidungen.
- Gehe nur zu Schritt 3 über, wenn der Kunde klar Interesse zeigt.

Antworte immer auf Deutsch. Halte deine Antworten kurz und gesprächig, maximal 2-3 Sätze pro Antwort."""
    voice_id: str = "j08ENmQlEinPmKqg3LUg"  # German voice
    tts_model: str = "eleven_flash_v2_5"
    llm_model: str = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5 (fastest)
    temperature: float = 0.7
    max_tokens: int = 500  # Increased for longer responses
    tts_speed: float = 1.1  # Voice speed (0.7-1.2, 1.0 = normal)
    trigger_on_keywords: str = ""  # Comma-separated keywords to trigger tool check. Empty = always check
    tasks: list = None  # List of Task objects
    product_summary: str = ""  # Product names + short descriptions for context
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
    
    def get_full_system_prompt(self) -> str:
        """Build system prompt with task instructions and product knowledge"""
        prompt = self.system_prompt
        
        # Add product summary if available
        if self.product_summary:
            prompt += f"""

## Available Products
You have access to information about these Swiss Life products:
{self.product_summary}

When recommending products, be specific about product names and their key benefits. Match products to the customer's stated goals and situation.
"""
        
        # Add task instructions if tasks exist
        if self.tasks:
            task_list = "\n".join([
                f"  {t.id}. {'[DONE]' if t.completed else '[TODO]'} {t.description}"
                for t in self.tasks
            ])
            
            prompt += f"""

## Your Tasks
You have the following tasks to complete during this conversation:
{task_list}

## Task Completion Rules
- Work through tasks naturally in conversation - don't rush or be robotic
- When you have successfully completed a task, include exactly this marker in your response: [TASK_DONE:X] where X is the task number
- Only mark a task done when you have genuinely accomplished it (e.g., obtained the information, provided the recommendation, etc.)
- You can complete multiple tasks in one response if appropriate
- Keep responses concise (2-3 sentences) while working toward your tasks
"""
        
        return prompt


class ElevenLabsRealtimeSTT:
    """
    ElevenLabs Scribe Realtime STT.
    
    Uses WebSocket connection for real-time transcription.
    VAD is handled client-side for turn-taking.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "wss://api.elevenlabs.io"
        self._ws = None
        self._connected = False
        self._receive_task = None
        
        # Callbacks
        self.on_partial_transcript = None
        self.on_final_transcript = None
        self.on_error = None
    
    async def connect(self) -> bool:
        """Connect to ElevenLabs Scribe Realtime WebSocket with manual commit"""
        try:
            # Build URL with manual commit strategy - browser VAD will trigger commits
            # This gives us faster partial transcripts
            params = [
                "model_id=scribe_v2_realtime",
                "encoding=pcm_16000",  # Format: pcm_{sample_rate}
                "sample_rate=16000",
                "commit_strategy=manual",  # Browser VAD will trigger commit
                "language_code=de"  # German
            ]
            url = f"{self.base_url}/v1/speech-to-text/realtime?{'&'.join(params)}"
            
            print(f"[STT] Connecting to: {url}")
            
            self._ws = await websockets.connect(
                url,
                additional_headers={"xi-api-key": self.api_key}
            )
            self._connected = True
            
            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            print("[STT] Connected to ElevenLabs Scribe Realtime")
            return True
            
        except Exception as e:
            print(f"[STT] Connection error: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False
    
    async def send_audio(self, audio_chunk: bytes):
        """Send audio chunk to STT"""
        if not self._connected or not self._ws:
            return
        
        try:
            # Format message as per ElevenLabs SDK
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(audio_chunk).decode("utf-8"),
                "commit": False,
                "sample_rate": 16000
            }
            await self._ws.send(json.dumps(message))
        except Exception as e:
            print(f"[STT] Send error: {e}")
    
    async def commit_transcription(self):
        """Commit the current transcription - called when browser VAD detects silence"""
        if not self._connected or not self._ws:
            return
        
        try:
            # Send an empty audio chunk with commit=True to trigger final transcript
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": 16000
            }
            await self._ws.send(json.dumps(message))
            print("[STT] Commit sent")
        except Exception as e:
            print(f"[STT] Commit error: {e}")
    
    async def _receive_loop(self):
        """Background loop to receive transcripts"""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    msg_type = data.get("message_type", data.get("type", ""))
                    
                    # Debug: log all messages from ElevenLabs
                    if msg_type not in ["session_started"]:
                        print(f"[STT] Received: {msg_type} -> {data}")
                    
                    if msg_type == "partial_transcript":
                        # API returns 'text' field
                        text = data.get("text", "")
                        if text.strip():
                            print(f"[STT] Partial: {text}")
                            if self.on_partial_transcript:
                                self.on_partial_transcript(text)
                    
                    elif msg_type == "transcript":
                        # Alternative message type for partial/interim transcripts
                        text = data.get("text", data.get("transcript", ""))
                        is_final = data.get("is_final", data.get("final", False))
                        if text.strip():
                            if is_final:
                                print(f"[STT] Final (transcript): {text}")
                                if self.on_final_transcript:
                                    asyncio.create_task(self.on_final_transcript(text))
                            else:
                                print(f"[STT] Partial (transcript): {text}")
                                if self.on_partial_transcript:
                                    self.on_partial_transcript(text)
                    
                    elif msg_type == "committed_transcript":
                        # VAD detected end of speech - this is the final transcript
                        # API returns 'text' field
                        text = data.get("text", "")
                        if text.strip():
                            print(f"[STT] Final: {text}")
                            if self.on_final_transcript:
                                self.on_final_transcript(text)
                    
                    elif msg_type == "session_started":
                        print("[STT] Session started successfully")
                    
                    elif msg_type in ["error", "auth_error", "rate_limited"]:
                        error_msg = data.get("message", str(data))
                        print(f"[STT] Error: {error_msg}")
                        if self.on_error:
                            self.on_error(error_msg)
                            
                except json.JSONDecodeError:
                    pass
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[STT] Connection closed: code={e.code}, reason={e.reason}")
        except Exception as e:
            print(f"[STT] Receive loop error: {e}")
        finally:
            self._connected = False
    
    async def close(self):
        """Close the connection"""
        self._connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            try:
                await self._ws.close()
            except:
                pass
            self._ws = None


class ClaudeLLM:
    """Claude LLM client with streaming and prompt caching"""
    
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    async def stream_response(
        self, 
        messages: list, 
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> AsyncIterator[str]:
        """Stream response tokens from Claude with prompt caching"""
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "content-type": "application/json"
        }
        
        # Use prompt caching for the system prompt (static content)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "messages": messages,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
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


class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech using REST API with streaming"""
    
    def __init__(self, api_key: str, voice_id: str, model_id: str = "eleven_flash_v2_5", speed: float = 1.0):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.speed = max(0.7, min(1.2, speed))  # Clamp to valid range
    
    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks as they're generated"""
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        params = {
            "output_format": "pcm_16000"
        }
        
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": self.speed
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload
                ) as response:
                    if response.status == 200:
                        # Buffer to ensure 2-byte alignment for 16-bit PCM
                        buffer = b""
                        first_chunk = True
                        chunk_count = 0
                        async for chunk in response.content.iter_chunked(8192):
                            buffer += chunk
                            # First chunk: send smaller for faster start (50ms)
                            # Later chunks: larger for efficiency (100ms)
                            chunk_size = 1600 if first_chunk else 3200
                            while len(buffer) >= chunk_size:
                                chunk_count += 1
                                yield buffer[:chunk_size]
                                buffer = buffer[chunk_size:]
                                first_chunk = False
                        
                        # Yield remaining buffer (ensure even length for 16-bit alignment)
                        if buffer:
                            if len(buffer) % 2 != 0:
                                buffer = buffer[:-1]  # Drop last byte if odd
                            if buffer:
                                chunk_count += 1
                                yield buffer
                        
                        if chunk_count == 0:
                            print(f"[TTS] WARNING: No audio chunks produced for text: {text[:50]}...")
                    else:
                        error = await response.text()
                        print(f"[TTS] Stream error {response.status}: {error}")
        except Exception as e:
            print(f"[TTS] Stream request error: {e}")


class VoiceAgentSession:
    """
    Manages a single voice agent conversation session.
    
    Uses ElevenLabs Scribe Realtime for STT with client-side VAD
    for automatic turn detection - no push-to-talk needed.
    """
    
    def __init__(
        self,
        ws: web.WebSocketResponse,
        elevenlabs_key: str,
        anthropic_key: str,
        config: AgentConfig
    ):
        self.ws = ws
        self.config = config
        self.elevenlabs_key = elevenlabs_key
        
        # Initialize clients
        self.stt = ElevenLabsRealtimeSTT(elevenlabs_key)
        self.llm = ClaudeLLM(anthropic_key, config.llm_model)
        self.tts = ElevenLabsTTS(elevenlabs_key, config.voice_id, config.tts_model, config.tts_speed)
        
        # Conversation state
        self.conversation_history = []
        self.is_processing = False
        self.is_speaking = False
        self._running = False
        self._interrupted = False  # Flag to signal barge-in interruption
        self._pending_interrupt_text = None  # Store text that interrupted
        self._audio_playing = False  # Track if browser is playing audio
        self._last_shown_transcript = None  # Prevent duplicate UI messages
        self._recent_agent_speech = []  # Track recent agent speech for echo detection
        self._pending_user_text = None  # Track additional user speech during processing
        self._processing_task = None  # Track current processing task
    
    async def start(self):
        """Start the session - connect to STT with VAD"""
        self._running = True
        
        # Set up STT callbacks
        self.stt.on_partial_transcript = self._on_partial_transcript
        self.stt.on_final_transcript = lambda t: asyncio.create_task(self._on_final_transcript(t))
        self.stt.on_error = lambda e: asyncio.create_task(self._send_event("error", e))
        
        # Connect to STT
        if not await self.stt.connect():
            await self._send_event("error", "Failed to connect to speech recognition")
            return False
        
        await self._send_event("status", "listening")
        return True
    
    async def stop(self):
        """Stop the session"""
        self._running = False
        await self.stt.close()
    
    async def handle_audio(self, audio_data: bytes):
        """Handle incoming audio from browser - forward to STT"""
        if not self._running:
            return
        
        # Always send audio to STT for barge-in detection
        await self.stt.send_audio(audio_data)
    
    def _is_echo(self, text: str) -> bool:
        """Check if the transcribed text is likely echo from agent's speech"""
        text_lower = text.lower().strip()
        # Check if the transcript matches or is contained in recent agent speech
        for agent_text in self._recent_agent_speech:
            agent_lower = agent_text.lower()
            # If user text is very similar to what agent just said, it's echo
            if text_lower in agent_lower or agent_lower.startswith(text_lower):
                return True
        return False
    
    def _on_partial_transcript(self, text: str):
        """Handle partial transcript from STT"""
        if not text.strip():
            return
            
        # Debug: log state when we get a partial transcript
        print(f"[Session] Partial: '{text}' | is_speaking={self.is_speaking}, is_processing={self.is_processing}, _audio_playing={self._audio_playing}, _interrupted={self._interrupted}")
        
        # Check for echo - ignore if it matches recent agent speech
        if self._is_echo(text):
            print(f"[Session] Ignoring echo: {text}")
            return
        
        # If already interrupted, just update the pending text (don't re-trigger)
        if self._interrupted:
            self._pending_interrupt_text = text
            asyncio.create_task(self._send_event("partial_transcript", text))
            return
        
        # Only trigger barge-in if agent is SPEAKING (not just processing)
        # If only processing (LLM thinking, no speech yet), user might still be talking
        if (self.is_speaking or self._audio_playing):
            print(f"[Session] Barge-in detected: {text}")
            self._interrupted = True
            self._pending_interrupt_text = text  # Store for later processing
            # Cancel any ongoing processing task
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                print("[Session] Cancelled ongoing LLM processing")
            # Tell browser to stop playing audio immediately
            asyncio.create_task(self._send_event("clear_audio", True))
        
        asyncio.create_task(self._send_event("partial_transcript", text))
    
    async def _on_final_transcript(self, text: str):
        """Handle final transcript from STT - VAD detected end of speech"""
        if not text.strip():
            return
        
        # Check for echo - ignore if it matches recent agent speech
        if self._is_echo(text):
            print(f"[Session] Ignoring echo (final): {text}")
            return
        
        # If we were in interrupted state (agent was speaking), NOW process the user's complete input
        if self._interrupted:
            print(f"[Session] Processing barge-in (final): {text}")
            self._interrupted = False
            self._pending_interrupt_text = None
            await self._send_event("user_transcript", text)
            # Wait a moment for any ongoing tasks to fully cancel
            await asyncio.sleep(0.05)
            self._processing_task = asyncio.create_task(self._process_user_input(text))
            await self._processing_task
            return
        
        # If agent is currently speaking (not just processing), this is a barge-in
        if self.is_speaking or self._audio_playing:
            print(f"[Session] Interrupting agent speech with: {text}")
            self._interrupted = True
            self._pending_interrupt_text = text
            self._audio_playing = False  # Reset audio state
            # Cancel any ongoing processing task
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                print("[Session] Cancelled ongoing LLM processing")
            # Tell browser to stop playing audio immediately
            await self._send_event("clear_audio", True)
            # Send immediate feedback
            await self._send_event("user_transcript", text)
            await self._send_event("status", "interrupted")
            return
        
        # If we're processing but NOT speaking yet, accumulate user text
        # (They're continuing their thought before agent started responding)
        if self.is_processing and not self.is_speaking:
            print(f"[Session] User continued while processing (queuing): {text}")
            if self._pending_user_text:
                self._pending_user_text += " " + text
            else:
                self._pending_user_text = text
            await self._send_event("user_transcript", text)
            return
        
        print(f"[Session] User said: {text}")
        await self._send_event("user_transcript", text)
        self._processing_task = asyncio.create_task(self._process_user_input(text))
        await self._processing_task
    
    async def _process_user_input(self, text: str):
        """Process user input and generate response with optional RAG"""
        self.is_processing = True
        self._interrupted = False
        self._pending_user_text = None  # Clear any pending text - we're processing fresh
        
        try:
            # Check if this looks like a product-related query
            rag_context = await self._maybe_get_rag_context(text)
            
            # Build the user message, optionally with RAG context
            if rag_context:
                # Inject RAG context as part of the user message
                user_message = f"{text}\n\n[System context - relevant product information:]\n{rag_context}"
            else:
                user_message = text
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Generate response with streaming TTS
            print("[Session] Calling LLM with streaming TTS...")
            await self._send_event("status", "thinking")
            
            response_text = ""
            sentence_buffer = ""
            sentence_endings = {'.', '!', '?', ':', ';'}
            first_audio = True
            
            async for token in self.llm.stream_response(
                self.conversation_history,
                self.config.get_full_system_prompt(),
                self.config.temperature,
                self.config.max_tokens
            ):
                # Check for interruption
                if self._interrupted:
                    print("[Session] LLM generation interrupted by user")
                    break
                
                response_text += token
                sentence_buffer += token
                
                # Send to UI
                display_text = self._strip_task_markers(response_text)
                await self._send_event("partial_response", display_text)
                
                # Check if we have a complete sentence to speak
                for ending in sentence_endings:
                    if ending in sentence_buffer:
                        # Find the last sentence ending
                        last_end = max(
                            sentence_buffer.rfind(ending + ' '),
                            sentence_buffer.rfind(ending) if sentence_buffer.endswith(ending) else -1
                        )
                        if last_end >= 0:
                            # Extract complete sentence(s)
                            speak_text = sentence_buffer[:last_end + 1].strip()
                            sentence_buffer = sentence_buffer[last_end + 1:].lstrip()
                            
                            if speak_text and len(speak_text) > 3:
                                if first_audio:
                                    self.is_speaking = True
                                    self._recent_agent_speech = []  # Clear old speech
                                    await self._send_event("status", "speaking")
                                    first_audio = False
                                
                                # Parse task completions and strip markers before TTS
                                await self._parse_task_completions(speak_text)
                                speak_text = self._strip_task_markers(speak_text)
                                
                                # Skip if nothing left after stripping markers
                                if not speak_text.strip():
                                    continue
                                
                                # Track for echo detection
                                self._recent_agent_speech.append(speak_text)
                                if len(self._recent_agent_speech) > 3:
                                    self._recent_agent_speech.pop(0)
                                
                                print(f"[Session] Speaking: {speak_text}")
                                async for audio_chunk in self.tts.synthesize_streaming(speak_text):
                                    if self._interrupted or not self._running:
                                        print("[Session] TTS interrupted")
                                        break
                                    await self._send_audio(audio_chunk)
                                
                                if self._interrupted:
                                    break
                            break
                
                if self._interrupted:
                    break
            
            # Speak any remaining text
            if not self._interrupted and sentence_buffer.strip():
                remaining = sentence_buffer.strip()
                await self._parse_task_completions(remaining)
                remaining = self._strip_task_markers(remaining)
                
                if remaining.strip():
                    if first_audio:
                        self.is_speaking = True
                        await self._send_event("status", "speaking")
                    
                    print(f"[Session] Speaking remaining: {remaining}")
                    async for audio_chunk in self.tts.synthesize_streaming(remaining):
                        if self._interrupted or not self._running:
                            print("[Session] TTS interrupted")
                            break
                        await self._send_audio(audio_chunk)
            
            # Add response to history
            if response_text:
                await self._parse_task_completions(response_text)
                clean_response = self._strip_task_markers(response_text)
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": clean_response + (" [interrupted]" if self._interrupted else "")
                })
                await self._send_event("agent_response", clean_response)
            
            await self._send_event("audio_done", True)
            
            # If there's a pending interrupt, wait for final transcript
            if self._interrupted and self._pending_interrupt_text:
                print(f"[Session] Waiting for user to finish speaking...")
                self.is_processing = False
                self.is_speaking = False
                self._audio_playing = False
                return
            
        except asyncio.CancelledError:
            print("[Session] Processing cancelled due to barge-in")
            self.is_processing = False
            self.is_speaking = False
            self._audio_playing = False
            return
            
        except Exception as e:
            print(f"[Session] Error: {e}")
            import traceback
            traceback.print_exc()
            await self._send_event("error", str(e))
        
        # Check if user continued speaking while we were processing
        if self._pending_user_text and not self._interrupted:
            pending = self._pending_user_text
            self._pending_user_text = None
            print(f"[Session] Processing queued user text: {pending}")
            self.is_processing = False
            self.is_speaking = False
            self._audio_playing = False
            await self._process_user_input(pending)
            return
        
        # Reset state
        self.is_processing = False
        self.is_speaking = False
        self._audio_playing = False
        self._recent_agent_speech = []
        await self._send_event("status", "listening")
    
    async def _maybe_get_rag_context(self, text: str) -> str:
        """Use LLM to decide if RAG is needed, then fetch context if so"""
        
        # If keywords are configured, only run tool check if input matches
        if self.config.trigger_on_keywords:
            keywords = [k.strip().lower() for k in self.config.trigger_on_keywords.split(',') if k.strip()]
            text_lower = text.lower()
            if not any(kw in text_lower for kw in keywords):
                print(f"[Session] Skipping tool check (no keyword match)")
                return ""
        
        # Fast LLM call to check if product lookup is needed
        tool_check_prompt = """You are a tool router. Based on the user's message, decide if you need to look up product information.

Respond with ONLY one of:
- SEARCH: <query> - if user is asking about products, recommendations, or needs specific product details
- NONE - if user is just chatting, giving their name, or not asking about products

Examples:
- "My name is John" → NONE
- "I want to save for retirement" → SEARCH: retirement savings products
- "What products do you recommend?" → SEARCH: product recommendations
- "Hello" → NONE
- "I'm 35 and want growth" → SEARCH: growth investment products for 35 year old"""

        import time
        start_time = time.time()
        
        try:
            headers = {
                "x-api-key": self.llm.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            payload = {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 50,
                "system": tool_check_prompt,
                "messages": [{"role": "user", "content": text}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        decision = data["content"][0]["text"].strip()
                        elapsed = (time.time() - start_time) * 1000
                        print(f"[Session] Tool check ({elapsed:.0f}ms): {decision}")
                        
                        if decision.startswith("SEARCH:"):
                            query = decision[7:].strip()
                            print(f"[Session] RAG triggered: {query}")
                            
                            rag = get_product_rag()
                            results = await rag.search(query, top_k=5)
                            
                            if results:
                                # Log detailed results to console
                                print(f"[Session] RAG returned {len(results)} results:")
                                for i, r in enumerate(results, 1):
                                    print(f"  {i}. {r.product_name} (score: {r.score:.3f})")
                                
                                # Send RAG results to UI (show more content for better display)
                                rag_display = [
                                    {"product": r.product_name, "score": round(r.score, 3), "snippet": r.content[:400] + ("..." if len(r.content) > 400 else "")}
                                    for r in results
                                ]
                                await self._send_event("rag_results", {"query": query, "results": rag_display})
                                
                                context = "\n\n".join([
                                    f"### {r.product_name}\n{r.content}"
                                    for r in results
                                ])
                                return context
                    else:
                        print(f"[Session] Tool check failed: {response.status}")
                    
        except Exception as e:
            print(f"[Session] Tool check error: {e}")
        
        return ""
    
    async def _parse_task_completions(self, response_text: str):
        """Parse response for task completion markers and update tasks"""
        import re
        pattern = r'\[TASK_DONE:(\d+)\]'
        matches = re.findall(pattern, response_text)
        
        for task_id_str in matches:
            task_id = int(task_id_str)
            for task in self.config.tasks:
                if task.id == task_id and not task.completed:
                    task.completed = True
                    print(f"[Session] Task {task_id} completed: {task.description}")
                    # Notify frontend
                    await self._send_event("task_update", {
                        "id": task_id,
                        "description": task.description,
                        "completed": True
                    })
    
    def _strip_task_markers(self, text: str) -> str:
        """Remove task completion markers from text for display - handles partial markers too"""
        import re
        # Remove complete markers
        text = re.sub(r'\s*\[TASK_DONE:\d+\]\s*', ' ', text)
        # Remove partial markers like "[TASK_DONE:" or just the number and bracket
        text = re.sub(r'\s*\[TASK_DONE:\d*$', '', text)  # Incomplete at end
        text = re.sub(r'^\d*\]\s*', '', text)  # Just the end part at start
        return text.strip()
    
    async def _send_event(self, event_type: str, data):
        """Send event to browser"""
        try:
            await self.ws.send_json({
                "type": event_type,
                "data": data
            })
        except Exception as e:
            print(f"[WS] Send error: {e}")
    
    async def _send_audio(self, audio_data: bytes):
        """Send audio chunk to browser"""
        try:
            if self.ws.closed:
                print("[WS] Cannot send audio - WebSocket closed")
                return
            await self.ws.send_json({
                "type": "audio",
                "audio": base64.b64encode(audio_data).decode("utf-8")
            })
        except Exception as e:
            print(f"[WS] Audio send error: {e}")


class VoiceAgentServer:
    """Web server for the voice agent"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        # Use German voice from env or default (set in AgentConfig)
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "j08ENmQlEinPmKqg3LUg")  # German voice
        
        if not self.elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY not found")
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.config = AgentConfig(
            voice_id=self.voice_id,  # German voice
            # Keywords that trigger tool check. Empty = always check
            trigger_on_keywords="produkt,empfehlen,sparen,ersparnisse,rente,pension,investieren,investition,vorsorge,säule,3a,3b,versicherung,vorschlag,option,plan,was haben sie,was bieten sie,product,recommend,save,savings,retire,retirement,invest,investment,pension,pillar,insurance,suggest"
        )
        self.app = web.Application()
        self._setup_routes()
        
        # Register startup handler for RAG initialization
        self.app.on_startup.append(self._on_startup)
    
    async def _on_startup(self, app):
        """Initialize RAG system at startup"""
        print("[Server] Initializing RAG system...")
        rag = await initialize_rag()
        # Store product summary in default config
        self.config.product_summary = rag.get_product_summary()
        print(f"[Server] RAG ready with {len(rag.chunks)} product chunks")
    
    def _setup_routes(self):
        """Set up HTTP routes"""
        static_path = Path(__file__).parent / "static"
        
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/ws", self._handle_websocket)
        if static_path.exists():
            self.app.router.add_static("/static", static_path)
    
    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve main page"""
        html_path = Path(__file__).parent / "static" / "voice_agent.html"
        if html_path.exists():
            return web.FileResponse(html_path)
        return web.Response(text="voice_agent.html not found", status=404)
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection from browser"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        print("[Server] Client connected")
        
        # Create session with a copy of config (so each session has its own state)
        session_config = AgentConfig(
            system_prompt=self.config.system_prompt,
            voice_id=self.config.voice_id,
            tts_model=self.config.tts_model,
            llm_model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tts_speed=self.config.tts_speed,
            trigger_on_keywords=self.config.trigger_on_keywords,
            product_summary=self.config.product_summary  # Include product summary
        )
        
        session = VoiceAgentSession(
            ws=ws,
            elevenlabs_key=self.elevenlabs_key,
            anthropic_key=self.anthropic_key,
            config=session_config
        )
        
        if not await session.start():
            return ws
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "audio":
                        # Decode and forward audio to STT
                        audio_b64 = data.get("audio", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            await session.handle_audio(audio_bytes)
                    
                    elif data.get("type") == "user_speaking":
                        # Browser detected user speaking (instant VAD)
                        if data.get("interrupted"):
                            print("[Server] Browser VAD: User interrupted agent")
                            session._interrupted = True
                            session._audio_playing = False
                    
                    elif data.get("type") == "audio_status":
                        # Browser reports audio playback status
                        playing = data.get("playing", False)
                        if playing:  # Only log when starting, not every stop
                            print(f"[Server] Audio status from browser: playing")
                        session._audio_playing = playing
                    
                    elif data.get("type") == "commit":
                        # Browser VAD detected silence - commit the transcription
                        print("[Server] Browser VAD: Committing transcription")
                        await session.stt.commit_transcription()
                    
                    elif data.get("type") == "config":
                        if "system_prompt" in data:
                            session.config.system_prompt = data["system_prompt"]
                        if "tasks" in data:
                            # Create Task objects from the task data
                            session.config.tasks = [
                                Task(id=t.get("id", i+1), description=t.get("description", ""), completed=t.get("completed", False))
                                for i, t in enumerate(data["tasks"])
                            ]
                            # Send initial tasks list to frontend
                            await ws.send_json({
                                "type": "tasks",
                                "data": [
                                    {"id": t.id, "description": t.description, "completed": t.completed}
                                    for t in session.config.tasks
                                ]
                            })
                        # Ensure product summary is set from RAG
                        if not session.config.product_summary:
                            session.config.product_summary = self.config.product_summary
                        await ws.send_json({"type": "status", "data": "config_updated"})
                    
                    elif data.get("type") == "clear_history":
                        session.conversation_history = []
                        await ws.send_json({"type": "status", "data": "history_cleared"})
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[Server] WebSocket error: {ws.exception()}")
                    break
        finally:
            await session.stop()
            print("[Server] Client disconnected")
        
        return ws
    
    def run(self):
        """Run the server"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║       Voice Agent with Client-Side VAD + Product RAG        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Open in browser: http://{self.host}:{self.port}                        ║
║                                                              ║
║  Components:                                                 ║
║    • STT: ElevenLabs Scribe Realtime                        ║
║    • LLM: Claude ({self.config.llm_model[:20]}...) + Caching    ║
║    • TTS: ElevenLabs Flash v2.5                             ║
║    • RAG: TF-IDF (scikit-learn, async)                      ║
║    • VAD: Client-side (browser)                             ║
║                                                              ║
║  Just speak naturally - no button needed!                    ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        web.run_app(self.app, host=self.host, port=self.port, print=None)


def main():
    """Main entry point"""
    try:
        server = VoiceAgentServer()
        server.run()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nMake sure your .env file has:")
        print("  ELEVENLABS_API_KEY=your-key")
        print("  ANTHROPIC_API_KEY=your-key")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0


if __name__ == "__main__":
    exit(main())
