#!/usr/bin/env python3
"""
Voice Agent Demo
================

A low-latency conversational voice agent using:
- ElevenLabs Scribe for STT (~100ms)
- Silero VAD for voice activity detection
- Claude/GPT/Gemini for LLM (streaming)
- ElevenLabs Flash for TTS (~75ms TTFB)

Usage:
    python demo.py --mode text      # Text input simulation
    python demo.py --mode benchmark # Latency testing
    python demo.py --mode voice     # Microphone input (requires pyaudio)
"""

import asyncio
import argparse
import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_pipeline():
    """Initialize all components"""
    from stt_client import ElevenLabsSTTClient
    from vad_client import SileroVADClient, WebRTCVADClient
    from llm_client import AnthropicLLMClient, OpenAILLMClient, GeminiLLMClient
    from tts_client import ElevenLabsTTSClient
    from pipeline import VoiceAgentPipeline
    
    # Get API keys
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not elevenlabs_key:
        print("ERROR: Set ELEVENLABS_API_KEY environment variable")
        print("Get your key at: https://elevenlabs.io/")
        return None
    
    if not (anthropic_key or openai_key or gemini_key):
        print("ERROR: Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")
        return None
    
    print("Initializing pipeline...")
    
    # STT
    stt = ElevenLabsSTTClient(api_key=elevenlabs_key)
    print("  ✓ STT: ElevenLabs Scribe")
    
    # VAD - try Silero first, fall back to WebRTC
    vad = None
    try:
        vad = SileroVADClient()
        print("  ✓ VAD: Silero")
    except Exception as e:
        print(f"  ⚠ Silero VAD failed ({e}), trying WebRTC...")
        try:
            vad = WebRTCVADClient()
            print("  ✓ VAD: WebRTC (fallback)")
        except Exception as e2:
            print(f"  ✗ VAD: Both Silero and WebRTC failed")
            print(f"    Install with: pip install torch torchaudio (for Silero)")
            print(f"    Or: pip install webrtcvad (for WebRTC)")
            # Create a dummy VAD for text mode
            class DummyVAD:
                _current_speech_duration = 0
                _current_silence_duration = 0
                async def process_chunk(self, _): return False
                async def detect_speech(self, _): 
                    from vad_client import VADResult
                    return VADResult(is_speech=False, confidence=0, duration_ms=0, rms_level=0)
                def reset(self): pass
            vad = DummyVAD()
            print("  ✓ VAD: Dummy (text mode only)")
    
    # LLM - prefer Claude, then OpenAI, then Gemini
    llm = None
    if anthropic_key:
        llm = AnthropicLLMClient(api_key=anthropic_key, model="claude-sonnet-4-20250514")
        print("  ✓ LLM: Claude Sonnet")
    elif openai_key:
        llm = OpenAILLMClient(api_key=openai_key, model="gpt-4o-mini")
        print("  ✓ LLM: GPT-4o-mini")
    elif gemini_key:
        llm = GeminiLLMClient(api_key=gemini_key, model="gemini-2.0-flash-exp")
        print("  ✓ LLM: Gemini Flash")
    
    # TTS - use voice ID from env if provided
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    tts = ElevenLabsTTSClient(
        api_key=elevenlabs_key,
        voice_id=voice_id,
        model_id="eleven_flash_v2_5"
    )
    print("  ✓ TTS: ElevenLabs Flash")
    
    # Pipeline
    pipeline = VoiceAgentPipeline(
        stt_client=stt,
        vad_client=vad,
        llm_client=llm,
        tts_client=tts,
        system_prompt="""You are a helpful voice assistant. Keep responses concise and natural - 
ideally 1-2 sentences. Speak conversationally as if talking to a friend."""
    )
    
    print("\n✓ Pipeline ready!\n")
    return pipeline


async def text_mode_demo(pipeline):
    """Text-based testing without microphone"""
    from tts_client import TextChunker
    
    print("\n" + "="*60)
    print("Voice Agent - Text Mode")
    print("="*60)
    print("Type messages to simulate voice input.")
    print("Type 'quit' to exit.\n")
    
    audio_buffer = bytearray()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break
        
        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        
        print("\nAssistant: ", end="", flush=True)
        
        # Simulate voice pipeline (skip STT/VAD, go direct to LLM)
        audio_buffer.clear()
        start = time.time()
        
        # Build messages
        messages = [
            {"role": "system", "content": pipeline.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Add conversation history
        for turn in pipeline.conversation_history[-10:]:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_input})
        
        text_buffer = ""
        full_response = ""
        first_token = True
        first_audio = True
        tts_start = None
        
        try:
            # Stream LLM tokens and send to TTS without blocking
            async for token in pipeline.llm.stream_completion(messages):
                if first_token:
                    pipeline.current_metrics.llm_ttfb_ms = (time.time() - start) * 1000
                    first_token = False
                    tts_start = time.time()
                
                print(token, end="", flush=True)
                text_buffer += token
                full_response += token
                
                # Send to TTS at sentence boundaries (non-blocking)
                if TextChunker.should_flush(text_buffer):
                    # Send text to TTS, collect any ready audio
                    async for audio in pipeline.tts.stream_speech(text_buffer):
                        if first_audio and tts_start:
                            pipeline.current_metrics.tts_ttfb_ms = (time.time() - tts_start) * 1000
                            first_audio = False
                        audio_buffer.extend(audio)
                    text_buffer = ""
            
            # Flush remaining text and collect all remaining audio
            if text_buffer:
                async for audio in pipeline.tts.stream_speech(text_buffer, flush=True):
                    if first_audio and tts_start:
                        pipeline.current_metrics.tts_ttfb_ms = (time.time() - tts_start) * 1000
                        first_audio = False
                    audio_buffer.extend(audio)
            
            # Collect any remaining audio
            async for audio in pipeline.tts.flush_audio():
                audio_buffer.extend(audio)
            
            total = (time.time() - start) * 1000
            print(f"\n\n[LLM TTFB: {pipeline.current_metrics.llm_ttfb_ms:.0f}ms | "
                  f"TTS TTFB: {pipeline.current_metrics.tts_ttfb_ms:.0f}ms | "
                  f"Total: {total:.0f}ms | Audio: {len(audio_buffer)/1024:.1f}KB]")
        
        except Exception as e:
            print(f"\n\nError: {e}")
    
    print("\nGoodbye!")


async def benchmark_mode(pipeline):
    """Measure component latencies"""
    print("\n" + "="*60)
    print("Voice Agent - Benchmark")
    print("="*60)
    
    test_phrases = [
        "Hello, how are you?",
        "What's the weather like?",
        "Tell me a joke.",
        "What time is it?",
        "Thank you."
    ]
    
    results = {'llm': [], 'tts': [], 'total': []}
    
    for phrase in test_phrases:
        print(f"\nTesting: '{phrase}'")
        
        messages = [
            {"role": "system", "content": "Keep responses under 30 words."},
            {"role": "user", "content": phrase}
        ]
        
        # Measure LLM
        start = time.time()
        first_token_time = None
        response = ""
        
        try:
            async for token in pipeline.llm.stream_completion(messages):
                if first_token_time is None:
                    first_token_time = time.time()
                response += token
            
            llm_ttfb = (first_token_time - start) * 1000 if first_token_time else 0
            results['llm'].append(llm_ttfb)
            
            # Measure TTS
            start = time.time()
            first_audio_time = None
            
            async for audio in pipeline.tts.stream_speech(response, flush=True):
                if first_audio_time is None:
                    first_audio_time = time.time()
            
            tts_ttfb = (first_audio_time - start) * 1000 if first_audio_time else 0
            results['tts'].append(tts_ttfb)
            
            total = llm_ttfb + tts_ttfb
            results['total'].append(total)
            
            print(f"  LLM: {llm_ttfb:.0f}ms | TTS: {tts_ttfb:.0f}ms | Total: {total:.0f}ms")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    if results['total']:
        print("\n" + "-"*40)
        print("AVERAGES:")
        print(f"  LLM TTFB: {sum(results['llm'])/len(results['llm']):.0f}ms")
        print(f"  TTS TTFB: {sum(results['tts'])/len(results['tts']):.0f}ms")
        print(f"  Total: {sum(results['total'])/len(results['total']):.0f}ms")
        
        avg = sum(results['total'])/len(results['total'])
        print(f"\nTarget: <1000ms → {'✓ PASS' if avg < 1000 else '✗ FAIL'}")


async def voice_mode_demo(pipeline):
    """Real-time voice interaction using microphone"""
    try:
        import pyaudio
    except ImportError:
        print("ERROR: pyaudio not installed. Run: pip install pyaudio")
        print("On Windows, you may need: pip install pipwin && pipwin install pyaudio")
        return
    
    print("\n" + "="*60)
    print("Voice Agent - Voice Mode")
    print("="*60)
    print("Speak into your microphone. Press Ctrl+C to exit.\n")
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024  # ~64ms chunks
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    audio = pyaudio.PyAudio()
    
    # Find input device
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"ERROR: Could not open microphone: {e}")
        return
    
    print("Listening... (speak now)")
    
    async def audio_generator():
        """Generate audio chunks from microphone"""
        try:
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                yield data
                await asyncio.sleep(0.001)  # Yield control
        except asyncio.CancelledError:
            pass
    
    # Set up callbacks
    def on_transcript(text, is_final):
        prefix = "📝 " if is_final else "... "
        print(f"\r{prefix}{text}", end="", flush=True)
    
    def on_response(token):
        print(token, end="", flush=True)
    
    pipeline.on_transcript = on_transcript
    pipeline.on_response_chunk = on_response
    
    try:
        await pipeline.process_audio_stream(audio_generator())
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


async def main():
    parser = argparse.ArgumentParser(description="Voice Agent Demo")
    parser.add_argument(
        "--mode",
        choices=["text", "voice", "benchmark"],
        default="text",
        help="Operation mode: text (simulate), voice (microphone), or benchmark (latency test)"
    )
    args = parser.parse_args()
    
    pipeline = setup_pipeline()
    if not pipeline:
        sys.exit(1)
    
    try:
        if args.mode == "text":
            await text_mode_demo(pipeline)
        elif args.mode == "benchmark":
            await benchmark_mode(pipeline)
        elif args.mode == "voice":
            await voice_mode_demo(pipeline)
    finally:
        await pipeline.tts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
