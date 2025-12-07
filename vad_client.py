"""
Voice Activity Detection Client Module
======================================

Silero VAD for detecting speech/silence and determining when user has finished speaking.
Includes semantic turn-taking analysis for smarter end-of-turn detection.
"""

import asyncio
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class VADResult:
    """Result from VAD processing"""
    is_speech: bool
    confidence: float  # 0.0 to 1.0
    duration_ms: float
    rms_level: float  # Audio level for visualization


class SileroVADClient:
    """
    Silero VAD for voice activity detection.
    
    Why Silero:
    - Most accurate open-source VAD
    - Fast inference (~1ms on CPU)
    - No GPU required
    - Trained on diverse audio including accents, noise
    
    How it works:
    - Neural network classifies 32ms audio windows as speech/non-speech
    - Returns confidence score 0-1
    - We apply threshold (typically 0.5) to make binary decision
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        window_size_ms: int = 32
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_ms = window_size_ms
        
        # State tracking
        self._is_speaking = False
        self._speech_start_time = 0
        self._silence_start_time = 0
        self._current_speech_duration = 0
        self._current_silence_duration = 0
        
        # Load Silero model from torch hub
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model"""
        import torch
        
        self._model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        # Utils for speech timestamp extraction
        (self.get_speech_timestamps, _, _, _, _) = utils
    
    async def process_chunk(self, audio_chunk: bytes) -> bool:
        """
        Process audio chunk and determine if turn is complete.
        
        Returns True when user has finished speaking (end-of-turn detected).
        
        Algorithm:
        1. Run VAD on audio chunk
        2. If speech detected, update speech duration
        3. If silence detected after speech, update silence duration
        4. If silence exceeds threshold AND speech was long enough, return True
        """
        import torch
        
        # Convert bytes to float array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Run Silero VAD
        chunk_samples = int(self.sample_rate * self.window_size_ms / 1000)
        confidences = []
        
        for i in range(0, len(audio_array), chunk_samples):
            window = audio_array[i:i + chunk_samples]
            if len(window) < chunk_samples:
                window = np.pad(window, (0, chunk_samples - len(window)))
            
            tensor = torch.from_numpy(window)
            confidence = self._model(tensor, self.sample_rate).item()
            confidences.append(confidence)
        
        # Average confidence for this chunk
        avg_confidence = np.mean(confidences) if confidences else 0.0
        is_speech = avg_confidence >= self.threshold
        
        current_time = time.time() * 1000  # milliseconds
        
        # State machine for turn detection
        if is_speech:
            if not self._is_speaking:
                # Speech just started
                self._is_speaking = True
                self._speech_start_time = current_time
                self._silence_start_time = 0
            
            self._current_speech_duration = current_time - self._speech_start_time
            self._current_silence_duration = 0
            
        else:  # Silence
            if self._is_speaking:
                if self._silence_start_time == 0:
                    # Silence just started
                    self._silence_start_time = current_time
                
                self._current_silence_duration = current_time - self._silence_start_time
                
                # Check for end-of-turn
                if (self._current_speech_duration >= self.min_speech_duration_ms and
                    self._current_silence_duration >= self.min_silence_duration_ms):
                    
                    # End of turn detected!
                    self._is_speaking = False
                    self._speech_start_time = 0
                    self._silence_start_time = 0
                    
                    return True
        
        return False
    
    async def detect_speech(self, audio_chunk: bytes) -> VADResult:
        """Get detailed VAD result for a single chunk"""
        import torch
        
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_array ** 2))
        
        # Run VAD
        chunk_samples = int(self.sample_rate * self.window_size_ms / 1000)
        
        # Pad if needed
        if len(audio_array) < chunk_samples:
            audio_array = np.pad(audio_array, (0, chunk_samples - len(audio_array)))
        
        tensor = torch.from_numpy(audio_array[:chunk_samples])
        confidence = self._model(tensor, self.sample_rate).item()
        
        return VADResult(
            is_speech=confidence >= self.threshold,
            confidence=confidence,
            duration_ms=self._current_speech_duration if confidence >= self.threshold else self._current_silence_duration,
            rms_level=rms
        )
    
    def reset(self):
        """Reset state for new conversation turn"""
        self._is_speaking = False
        self._speech_start_time = 0
        self._silence_start_time = 0
        self._current_speech_duration = 0
        self._current_silence_duration = 0


class WebRTCVADClient:
    """
    WebRTC VAD as a fallback option.
    
    Lighter weight than Silero but less accurate.
    Good for systems where PyTorch is not available.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,  # 0-3, higher = more aggressive
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500
    ):
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
        except ImportError:
            raise ImportError("webrtcvad not installed. Run: pip install webrtcvad")
        
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        
        self._is_speaking = False
        self._speech_start_time = 0
        self._silence_start_time = 0
        self._current_speech_duration = 0
        self._current_silence_duration = 0
    
    async def process_chunk(self, audio_chunk: bytes) -> bool:
        """Process audio chunk and detect end-of-turn"""
        # WebRTC VAD requires 10, 20, or 30 ms frames
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
        
        is_speech = False
        for i in range(0, len(audio_chunk) - frame_size, frame_size):
            frame = audio_chunk[i:i + frame_size]
            if len(frame) == frame_size:
                if self._vad.is_speech(frame, self.sample_rate):
                    is_speech = True
                    break
        
        current_time = time.time() * 1000
        
        if is_speech:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = current_time
                self._silence_start_time = 0
            
            self._current_speech_duration = current_time - self._speech_start_time
            self._current_silence_duration = 0
        else:
            if self._is_speaking:
                if self._silence_start_time == 0:
                    self._silence_start_time = current_time
                
                self._current_silence_duration = current_time - self._silence_start_time
                
                if (self._current_speech_duration >= self.min_speech_duration_ms and
                    self._current_silence_duration >= self.min_silence_duration_ms):
                    self._is_speaking = False
                    self._speech_start_time = 0
                    self._silence_start_time = 0
                    return True
        
        return False
    
    async def detect_speech(self, audio_chunk: bytes) -> VADResult:
        """Get detailed VAD result"""
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000) * 2
        
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_array ** 2))
        
        is_speech = False
        if len(audio_chunk) >= frame_size:
            is_speech = self._vad.is_speech(audio_chunk[:frame_size], self.sample_rate)
        
        return VADResult(
            is_speech=is_speech,
            confidence=1.0 if is_speech else 0.0,
            duration_ms=self._current_speech_duration if is_speech else self._current_silence_duration,
            rms_level=rms
        )
    
    def reset(self):
        """Reset state for new conversation turn"""
        self._is_speaking = False
        self._speech_start_time = 0
        self._silence_start_time = 0
        self._current_speech_duration = 0
        self._current_silence_duration = 0


class SemanticTurnTaking:
    """
    Advanced turn-taking using semantic analysis of transcript.
    
    Why semantic analysis:
    - VAD alone can't distinguish "um..." (holding turn) from actual end
    - Incomplete sentences should extend the wait time
    - Filler words shouldn't trigger responses
    
    This approximates ElevenLabs' proprietary turn-taking model.
    """
    
    # Words that indicate user is still thinking
    FILLER_WORDS = {
        "um", "uh", "er", "ah", "like", "you know", "i mean",
        "so", "well", "actually", "basically", "literally"
    }
    
    # Sentence endings that indicate incomplete thought
    INCOMPLETE_ENDINGS = {
        "and", "but", "or", "because", "so", "if", "when",
        "while", "although", "however", "therefore", "then",
        "the", "a", "an", "to", "for", "with", "of"
    }
    
    def __init__(self, vad_client):
        self.vad = vad_client
        self._current_text = ""
    
    def update_text(self, text: str):
        """Update with latest transcript"""
        self._current_text = text.strip().lower()
    
    async def should_take_turn(self, audio_chunk: bytes) -> bool:
        """
        Determine if it's appropriate to take a turn.
        
        Combines acoustic VAD with semantic analysis:
        1. Check if VAD detects silence
        2. Check if transcript suggests completion
        3. Apply extended threshold for incomplete sentences
        """
        # Check acoustic VAD first
        vad_result = await self.vad.detect_speech(audio_chunk)
        
        if vad_result.is_speech:
            return False  # Still speaking
        
        # Need minimum silence
        if self.vad._current_silence_duration < 300:
            return False
        
        # Semantic checks
        if self._current_text:
            words = self._current_text.split()
            
            # Filter out filler words
            non_filler = [w for w in words if w not in self.FILLER_WORDS]
            if len(non_filler) == 0:
                return False  # Only filler words, wait for more
            
            # Check for incomplete sentence
            last_word = words[-1] if words else ""
            if last_word in self.INCOMPLETE_ENDINGS:
                # Sentence incomplete - require longer silence
                if self.vad._current_silence_duration < 1000:
                    return False
        
        # All checks passed
        return self.vad._current_speech_duration >= self.vad.min_speech_duration_ms
