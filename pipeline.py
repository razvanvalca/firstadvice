"""
Voice Agent Pipeline Orchestrator
==================================

Main orchestrator for the voice agent pipeline.
Manages component lifecycle, routes data between components,
handles state transitions, and coordinates interruptions.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Optional
from enum import Enum

from tts_client import TextChunker


class AgentState(Enum):
    """Pipeline state machine"""
    IDLE = "idle"           # Waiting for user
    LISTENING = "listening"  # Receiving audio, running STT
    PROCESSING = "processing"  # LLM generating response
    SPEAKING = "speaking"    # TTS playing audio
    INTERRUPTED = "interrupted"  # User interrupted, stopping


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    was_interrupted: bool = False


@dataclass
class PipelineMetrics:
    """Latency tracking"""
    stt_latency_ms: float = 0
    vad_latency_ms: float = 0
    llm_ttfb_ms: float = 0
    tts_ttfb_ms: float = 0
    total_latency_ms: float = 0


class VoiceAgentPipeline:
    """
    Main orchestrator for voice agent pipeline.
    
    Responsibilities:
    1. Initialize and manage all components
    2. Route audio → STT → VAD → LLM → TTS → audio
    3. Handle state transitions
    4. Coordinate interruptions
    5. Track conversation history
    6. Measure latency
    """
    
    def __init__(
        self,
        stt_client,
        vad_client,
        llm_client,
        tts_client,
        system_prompt: str = "You are a helpful voice assistant. Keep responses concise."
    ):
        self.stt = stt_client
        self.vad = vad_client
        self.llm = llm_client
        self.tts = tts_client
        
        self.system_prompt = system_prompt
        self.conversation_history: list[ConversationTurn] = []
        
        self.state = AgentState.IDLE
        self.current_metrics = PipelineMetrics()
        
        # Interruption coordination
        self._interrupt_event = asyncio.Event()
        
        # Callbacks for UI
        self.on_state_change: Optional[Callable] = None
        self.on_transcript: Optional[Callable] = None
        self.on_response_chunk: Optional[Callable] = None
        self.on_audio_chunk: Optional[Callable] = None
    
    def _set_state(self, new_state: AgentState):
        """Update state and notify listeners"""
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(new_state)
    
    async def process_audio_stream(self, audio_stream: AsyncGenerator[bytes, None]):
        """
        Main entry point for processing audio.
        
        This is the core pipeline:
        1. Receive audio chunks from microphone
        2. Feed to STT (runs in parallel while user speaks)
        3. Feed to VAD (detects end-of-turn)
        4. When turn complete, process transcript through LLM
        5. Stream LLM tokens to TTS
        6. Play TTS audio
        7. Handle interruptions
        """
        self._set_state(AgentState.LISTENING)
        
        pipeline_start = time.time()
        transcript_buffer = ""
        
        try:
            async for audio_chunk in audio_stream:
                # Check for interruption while speaking
                if self.state == AgentState.SPEAKING:
                    if await self._detect_interruption(audio_chunk):
                        await self._handle_interruption()
                        self._set_state(AgentState.LISTENING)
                        continue
                
                # Feed to STT
                stt_start = time.time()
                await self.stt.send_audio(audio_chunk)
                transcript = await self.stt.get_transcript(timeout=0.1)
                
                if transcript:
                    transcript_buffer = transcript.text
                    self.current_metrics.stt_latency_ms = (time.time() - stt_start) * 1000
                    
                    if self.on_transcript:
                        self.on_transcript(transcript_buffer, transcript.is_final)
                
                # Feed to VAD
                vad_start = time.time()
                turn_complete = await self.vad.process_chunk(audio_chunk)
                self.current_metrics.vad_latency_ms = (time.time() - vad_start) * 1000
                
                # If turn complete, process response
                if turn_complete and transcript_buffer.strip():
                    self._set_state(AgentState.PROCESSING)
                    
                    # Add to history
                    self.conversation_history.append(
                        ConversationTurn(role="user", content=transcript_buffer)
                    )
                    
                    # Generate and speak response
                    await self._process_and_respond(transcript_buffer, pipeline_start)
                    
                    # Reset for next turn
                    transcript_buffer = ""
                    pipeline_start = time.time()
                    self.vad.reset()
                    self._set_state(AgentState.LISTENING)
        
        except asyncio.CancelledError:
            pass
        finally:
            self._set_state(AgentState.IDLE)
    
    async def _process_and_respond(self, user_input: str, pipeline_start: float):
        """Process user input and generate spoken response"""
        
        # Build message history for LLM
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.conversation_history[-10:]:  # Last 10 turns
            messages.append({"role": turn.role, "content": turn.content})
        
        self._set_state(AgentState.SPEAKING)
        self._interrupt_event.clear()
        
        response_text = ""
        text_buffer = ""
        first_token = True
        first_audio = True
        
        # Stream LLM response
        llm_start = time.time()
        
        async for token in self.llm.stream_completion(messages):
            # Check for interruption
            if self._interrupt_event.is_set():
                break
            
            # Track TTFB
            if first_token:
                self.current_metrics.llm_ttfb_ms = (time.time() - llm_start) * 1000
                first_token = False
            
            response_text += token
            text_buffer += token
            
            if self.on_response_chunk:
                self.on_response_chunk(token)
            
            # Stream to TTS at sentence boundaries
            if TextChunker.should_flush(text_buffer):
                tts_start = time.time()
                
                async for audio in self.tts.stream_speech(text_buffer):
                    if self._interrupt_event.is_set():
                        break
                    
                    if first_audio:
                        self.current_metrics.tts_ttfb_ms = (time.time() - tts_start) * 1000
                        first_audio = False
                    
                    if self.on_audio_chunk:
                        self.on_audio_chunk(audio)
                
                text_buffer = ""
        
        # Flush remaining text
        if text_buffer and not self._interrupt_event.is_set():
            async for audio in self.tts.stream_speech(text_buffer, flush=True):
                if self._interrupt_event.is_set():
                    break
                if self.on_audio_chunk:
                    self.on_audio_chunk(audio)
        
        # Record metrics
        self.current_metrics.total_latency_ms = (time.time() - pipeline_start) * 1000
        
        # Add to history
        self.conversation_history.append(
            ConversationTurn(
                role="assistant",
                content=response_text,
                was_interrupted=self._interrupt_event.is_set()
            )
        )
    
    async def _detect_interruption(self, audio_chunk: bytes) -> bool:
        """Detect if user is interrupting while agent speaks"""
        vad_result = await self.vad.detect_speech(audio_chunk)
        
        # Require sustained speech to avoid false positives
        # (background noise, echo, etc.)
        if vad_result.confidence > 0.7 and vad_result.duration_ms > 200:
            return True
        
        return False
    
    async def _handle_interruption(self):
        """Handle user interruption gracefully"""
        self._interrupt_event.set()
        self._set_state(AgentState.INTERRUPTED)
        
        # Stop TTS immediately
        await self.tts.stop()
        
        # Mark last turn as interrupted
        if self.conversation_history and self.conversation_history[-1].role == "assistant":
            self.conversation_history[-1].was_interrupted = True
    
    async def process_text_input(self, text: str) -> AsyncGenerator[str, None]:
        """
        Process text input directly (for testing without microphone).
        
        Yields response tokens as they're generated.
        """
        self._set_state(AgentState.PROCESSING)
        
        # Add to history
        self.conversation_history.append(
            ConversationTurn(role="user", content=text)
        )
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.conversation_history[-10:]:
            messages.append({"role": turn.role, "content": turn.content})
        
        response_text = ""
        
        async for token in self.llm.stream_completion(messages):
            response_text += token
            yield token
        
        # Add response to history
        self.conversation_history.append(
            ConversationTurn(role="assistant", content=response_text)
        )
        
        self._set_state(AgentState.IDLE)
    
    def get_metrics_summary(self) -> str:
        """Get formatted metrics summary"""
        m = self.current_metrics
        return (
            f"STT: {m.stt_latency_ms:.0f}ms | "
            f"VAD: {m.vad_latency_ms:.0f}ms | "
            f"LLM TTFB: {m.llm_ttfb_ms:.0f}ms | "
            f"TTS TTFB: {m.tts_ttfb_ms:.0f}ms | "
            f"Total: {m.total_latency_ms:.0f}ms"
        )


class LatencyMonitor:
    """
    Monitor and report latency breakdown.
    
    Use this to identify bottlenecks.
    """
    
    def __init__(self):
        self.metrics = []
    
    def record(self, metrics: PipelineMetrics):
        self.metrics.append({
            'timestamp': time.time(),
            'stt': metrics.stt_latency_ms,
            'vad': metrics.vad_latency_ms,
            'llm_ttfb': metrics.llm_ttfb_ms,
            'tts_ttfb': metrics.tts_ttfb_ms,
            'total': metrics.total_latency_ms
        })
    
    def report(self) -> str:
        if not self.metrics:
            return "No metrics recorded"
        
        # Calculate averages
        avg = {
            'stt': sum(m['stt'] for m in self.metrics) / len(self.metrics),
            'vad': sum(m['vad'] for m in self.metrics) / len(self.metrics),
            'llm_ttfb': sum(m['llm_ttfb'] for m in self.metrics) / len(self.metrics),
            'tts_ttfb': sum(m['tts_ttfb'] for m in self.metrics) / len(self.metrics),
            'total': sum(m['total'] for m in self.metrics) / len(self.metrics),
        }
        
        recommendations = self._get_recommendations(avg)
        
        return f"""
Latency Report (avg of {len(self.metrics)} samples)
================================================
STT:      {avg['stt']:.0f}ms
VAD:      {avg['vad']:.0f}ms
LLM TTFB: {avg['llm_ttfb']:.0f}ms
TTS TTFB: {avg['tts_ttfb']:.0f}ms
------------------------------------------------
TOTAL:    {avg['total']:.0f}ms

Target: <1000ms
Status: {'✓ PASS' if avg['total'] < 1000 else '✗ FAIL'}

Recommendations:
{recommendations}
"""
    
    def _get_recommendations(self, avg: dict) -> str:
        recs = []
        
        if avg['llm_ttfb'] > 500:
            recs.append("- LLM is slow. Try gemini-flash or gpt-4o-mini")
        
        if avg['tts_ttfb'] > 150:
            recs.append("- TTS is slow. Use eleven_flash_v2_5 model")
        
        if avg['stt'] > 200:
            recs.append("- STT is slow. Ensure using WebSocket streaming")
        
        if not recs:
            recs.append("- Performance looks good!")
        
        return '\n'.join(recs)
