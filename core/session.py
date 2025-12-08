"""
Voice Agent Session

Manages a single voice conversation session, coordinating STT, LLM, TTS,
and RAG services with proper state management and interruption handling.
"""

import asyncio
import base64
import time
from pathlib import Path
from typing import Optional, List

from aiohttp import web

from config.settings import AppSettings
from core.models import Task, SessionState, PromptBuilder
from core.processors import ResponseProcessor, ToolRouter
from services.speech_to_text import SpeechToTextService
from services.text_to_speech import TextToSpeechService
from services.llm import LlmService
from services.product_rag import get_rag_service


class VoiceAgentSession:
    """
    Manages a single voice agent conversation session.
    
    Coordinates:
    - Speech-to-text transcription
    - LLM response generation with RAG
    - Text-to-speech synthesis
    - Barge-in (interruption) handling
    - Task completion tracking
    """
    
    def __init__(
        self,
        websocket: web.WebSocketResponse,
        settings: AppSettings,
        product_summary: str = "",
    ):
        """
        Initialize a new session.
        
        Args:
            websocket: WebSocket connection to the browser
            settings: Application settings
            product_summary: Pre-loaded product summary for system prompt
        """
        self._ws = websocket
        self._settings = settings
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        
        # Initialize services
        self._stt = SpeechToTextService(settings.elevenlabs)
        self._tts = TextToSpeechService(settings.elevenlabs)
        self._llm = LlmService(settings.anthropic)
        self._rag = get_rag_service(settings.rag)
        
        # Initialize processors
        self._response_processor = ResponseProcessor()
        self._tool_router = ToolRouter(settings.agent.rag_trigger_keywords)
        
        # Initialize state
        self._state = SessionState()
        
        # Build system prompt
        prompt_path = Path(__file__).parent.parent / settings.agent.system_prompt_file
        self._prompt_builder = PromptBuilder.from_file(prompt_path)
        self._prompt_builder.with_product_summary(product_summary)
    
    async def start(self) -> bool:
        """
        Start the session.
        
        Connects to STT service and sets up callbacks.
        
        Returns:
            True if session started successfully
        """
        self._running = True
        
        # Set up STT callbacks
        self._stt.on_partial_transcript = self._on_partial_transcript
        self._stt.on_final_transcript = lambda t: asyncio.create_task(self._on_final_transcript(t))
        self._stt.on_error = lambda e: asyncio.create_task(self._send_event("error", e))
        
        # Connect to STT
        if not await self._stt.connect():
            await self._send_event("error", "Failed to connect to speech recognition")
            return False
        
        await self._send_event("status", "listening")
        return True
    
    async def stop(self) -> None:
        """Stop the session and clean up resources."""
        self._running = False
        await self._stt.close()
    
    async def handle_audio(self, audio_data: bytes) -> None:
        """
        Handle incoming audio from the browser.
        
        Args:
            audio_data: Raw PCM audio bytes
        """
        if not self._running:
            return
        
        await self._stt.send_audio(audio_data)
    
    async def handle_commit(self) -> None:
        """Handle a commit request from browser VAD."""
        await self._stt.commit_transcription()
    
    def handle_audio_status(self, playing: bool) -> None:
        """
        Handle audio playback status from browser.
        
        Args:
            playing: Whether audio is currently playing
        """
        self._state.is_audio_playing = playing
    
    def handle_user_speaking_interrupt(self) -> None:
        """Handle user speaking interrupt notification from browser."""
        self._state.is_interrupted = True
        self._state.is_audio_playing = False
    
    def configure(
        self,
        system_prompt: Optional[str] = None,
        tasks: Optional[List[dict]] = None,
    ) -> None:
        """
        Update session configuration.
        
        Args:
            system_prompt: Override for system prompt
            tasks: Task definitions
        """
        if system_prompt:
            self._prompt_builder.base_prompt = system_prompt
        
        if tasks:
            self._state.tasks = [
                Task(
                    id=t.get("id", i + 1),
                    description=t.get("description", ""),
                    completed=t.get("completed", False),
                )
                for i, t in enumerate(tasks)
            ]
            self._prompt_builder.with_tasks(self._state.tasks)
    
    def get_tasks(self) -> List[dict]:
        """Get current tasks as serializable dicts."""
        return [
            {"id": t.id, "description": t.description, "completed": t.completed}
            for t in self._state.tasks
        ]
    
    # =========================================================================
    # Private: STT Callbacks
    # =========================================================================
    
    def _on_partial_transcript(self, text: str) -> None:
        """Handle partial transcript from STT."""
        if not text.strip():
            return
        
        # Debug logging
        print(f"[Session] Partial: '{text}' | speaking={self._state.is_speaking}, processing={self._state.is_processing}")
        
        # Check for echo
        if self._state.is_echo(text):
            print(f"[Session] Ignoring echo: {text}")
            return
        
        # If already interrupted, just update pending text
        if self._state.is_interrupted:
            self._state.pending_interrupt_text = text
            asyncio.create_task(self._send_event("partial_transcript", text))
            return
        
        # Trigger barge-in if agent is speaking
        if self._state.is_speaking or self._state.is_audio_playing:
            print(f"[Session] Barge-in detected: {text}")
            self._state.is_interrupted = True
            self._state.pending_interrupt_text = text
            
            # Cancel ongoing processing
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                print("[Session] Cancelled ongoing LLM processing")
            
            # Tell browser to stop audio
            asyncio.create_task(self._send_event("clear_audio", True))
        
        asyncio.create_task(self._send_event("partial_transcript", text))
    
    async def _on_final_transcript(self, text: str) -> None:
        """Handle final transcript from STT (end of speech)."""
        if not text.strip():
            return
        
        # Check for echo
        if self._state.is_echo(text):
            print(f"[Session] Ignoring echo (final): {text}")
            return
        
        # Process barge-in completion
        if self._state.is_interrupted:
            print(f"[Session] Processing barge-in (final): {text}")
            self._state.is_interrupted = False
            self._state.pending_interrupt_text = None
            await self._send_event("user_transcript", text)
            await asyncio.sleep(0.05)  # Let cancellation complete
            self._processing_task = asyncio.create_task(self._process_user_input(text))
            await self._processing_task
            return
        
        # Barge-in during speech
        if self._state.is_speaking or self._state.is_audio_playing:
            print(f"[Session] Interrupting agent speech with: {text}")
            self._state.is_interrupted = True
            self._state.pending_interrupt_text = text
            self._state.is_audio_playing = False
            
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
            
            await self._send_event("clear_audio", True)
            await self._send_event("user_transcript", text)
            await self._send_event("status", "interrupted")
            return
        
        # Queue text if processing but not speaking yet
        if self._state.is_processing and not self._state.is_speaking:
            print(f"[Session] User continued while processing: {text}")
            if self._state.pending_user_text:
                self._state.pending_user_text += " " + text
            else:
                self._state.pending_user_text = text
            await self._send_event("user_transcript", text)
            return
        
        # Normal processing
        print(f"[Session] User said: {text}")
        await self._send_event("user_transcript", text)
        self._processing_task = asyncio.create_task(self._process_user_input(text))
        await self._processing_task
    
    # =========================================================================
    # Private: Response Processing
    # =========================================================================
    
    async def _process_user_input(self, text: str) -> None:
        """Process user input and generate response."""
        self._state.is_processing = True
        self._state.is_interrupted = False
        self._state.pending_user_text = None
        
        try:
            # Check for RAG context
            rag_context = await self._maybe_get_rag_context(text)
            
            # Build user message with optional RAG context
            if rag_context:
                user_message = f"{text}\n\n[System context - relevant product information:]\n{rag_context}"
            else:
                user_message = text
            
            self._state.add_user_message(user_message)
            
            # Generate response
            print("[Session] Calling LLM with streaming TTS...")
            await self._send_event("status", "thinking")
            
            response_text = ""
            sentence_buffer = ""
            first_audio = True
            
            system_prompt = self._prompt_builder.build()
            
            async for token in self._llm.generate_stream(
                self._state.get_messages_for_llm(),
                system_prompt,
            ):
                if self._state.is_interrupted:
                    print("[Session] LLM generation interrupted")
                    break
                
                response_text += token
                sentence_buffer += token
                
                # Send partial response to UI
                display_text = self._response_processor.strip_task_markers(response_text)
                await self._send_event("partial_response", display_text)
                
                # Check for complete sentences to speak
                complete, sentence_buffer = self._response_processor.extract_complete_sentences(sentence_buffer)
                
                if complete and len(complete) > 3:
                    if first_audio:
                        self._state.is_speaking = True
                        self._state.clear_agent_speech_history()
                        await self._send_event("status", "speaking")
                        first_audio = False
                    
                    await self._process_and_speak(complete)
                    
                    if self._state.is_interrupted:
                        break
            
            # Speak remaining text
            if not self._state.is_interrupted and sentence_buffer.strip():
                remaining = sentence_buffer.strip()
                
                if first_audio:
                    self._state.is_speaking = True
                    await self._send_event("status", "speaking")
                
                await self._process_and_speak(remaining)
            
            # Save response to history
            if response_text:
                await self._finalize_response(response_text)
            
            await self._send_event("audio_done", True)
            
            # Handle pending interrupt
            if self._state.is_interrupted and self._state.pending_interrupt_text:
                print("[Session] Waiting for user to finish speaking...")
                self._state.reset_processing_state()
                return
                
        except asyncio.CancelledError:
            print("[Session] Processing cancelled due to barge-in")
            self._state.reset_processing_state()
            return
            
        except Exception as e:
            print(f"[Session] Error: {e}")
            import traceback
            traceback.print_exc()
            await self._send_event("error", str(e))
        
        # Process any queued user text
        if self._state.pending_user_text and not self._state.is_interrupted:
            pending = self._state.pending_user_text
            self._state.pending_user_text = None
            print(f"[Session] Processing queued text: {pending}")
            self._state.reset_processing_state()
            await self._process_user_input(pending)
            return
        
        self._state.reset_processing_state()
        await self._send_event("status", "listening")
    
    async def _process_and_speak(self, text: str) -> None:
        """Process text for task markers and speak it."""
        # Extract and handle task completions
        task_ids = self._response_processor.extract_task_completions(text)
        for task_id in task_ids:
            task = self._state.mark_task_completed(task_id)
            if task:
                print(f"[Session] Task {task_id} completed: {task.description}")
                await self._send_event("task_update", {
                    "id": task_id,
                    "description": task.description,
                    "completed": True,
                })
        
        # Clean text for TTS
        speak_text = self._response_processor.strip_task_markers(text)
        
        if not speak_text.strip():
            return
        
        # Track for echo detection
        self._state.track_agent_speech(speak_text)
        
        print(f"[Session] Speaking: {speak_text}")
        
        # Stream TTS
        async for audio_chunk in self._tts.synthesize(speak_text):
            if self._state.is_interrupted or not self._running:
                print("[Session] TTS interrupted")
                break
            await self._send_audio(audio_chunk)
    
    async def _finalize_response(self, response_text: str) -> None:
        """Finalize and save the response."""
        # Process any remaining task markers
        task_ids = self._response_processor.extract_task_completions(response_text)
        for task_id in task_ids:
            task = self._state.mark_task_completed(task_id)
            if task:
                await self._send_event("task_update", {
                    "id": task_id,
                    "description": task.description,
                    "completed": True,
                })
        
        clean_response = self._response_processor.strip_task_markers(response_text)
        
        if self._state.is_interrupted:
            clean_response += " [interrupted]"
        
        self._state.add_assistant_message(clean_response)
        await self._send_event("agent_response", clean_response)
    
    # =========================================================================
    # Private: RAG Integration
    # =========================================================================
    
    async def _maybe_get_rag_context(self, text: str) -> str:
        """
        Check if RAG is needed and retrieve context if so.
        
        Args:
            text: User's message
            
        Returns:
            RAG context string, or empty if not needed
        """
        # Quick keyword check
        if not self._tool_router.should_check_rag(text):
            print("[Session] Skipping RAG (no keyword match)")
            return ""
        
        # LLM classification
        start_time = time.time()
        
        try:
            decision = await self._llm.classify_intent(
                text,
                ToolRouter.CLASSIFICATION_PROMPT,
            )
            
            elapsed = (time.time() - start_time) * 1000
            print(f"[Session] Tool check ({elapsed:.0f}ms): {decision}")
            
            should_search, query = self._tool_router.parse_classification(decision)
            
            if should_search:
                print(f"[Session] RAG triggered: {query}")
                results = await self._rag.search(query)
                
                if results:
                    # Log results
                    print(f"[Session] RAG returned {len(results)} results:")
                    for i, r in enumerate(results, 1):
                        print(f"  {i}. {r.product_name} (score: {r.score:.3f})")
                    
                    # Send to UI
                    rag_display = [
                        {
                            "product": r.product_name,
                            "score": round(r.score, 3),
                            "snippet": r.content[:400] + ("..." if len(r.content) > 400 else ""),
                        }
                        for r in results
                    ]
                    await self._send_event("rag_results", {"query": query, "results": rag_display})
                    
                    # Build context
                    context = "\n\n".join([
                        f"### {r.product_name}\n{r.content}"
                        for r in results
                    ])
                    return context
                    
        except Exception as e:
            print(f"[Session] RAG error: {e}")
        
        return ""
    
    # =========================================================================
    # Private: WebSocket Communication
    # =========================================================================
    
    async def _send_event(self, event_type: str, data) -> None:
        """Send an event to the browser."""
        try:
            await self._ws.send_json({
                "type": event_type,
                "data": data,
            })
        except Exception as e:
            print(f"[WS] Send error: {e}")
    
    async def _send_audio(self, audio_data: bytes) -> None:
        """Send an audio chunk to the browser."""
        try:
            if self._ws.closed:
                print("[WS] Cannot send audio - WebSocket closed")
                return
            await self._ws.send_json({
                "type": "audio",
                "audio": base64.b64encode(audio_data).decode("utf-8"),
            })
        except Exception as e:
            print(f"[WS] Audio send error: {e}")
