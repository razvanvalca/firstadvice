"""
Response processor for handling LLM output.

Handles task completion parsing, text cleanup, and sentence splitting.
"""

import re
from typing import List, Tuple, Set


# Sentence ending punctuation
SENTENCE_ENDINGS: Set[str] = {'.', '!', '?', ':', ';'}

# Task completion pattern
TASK_DONE_PATTERN = re.compile(r'\[TASK_DONE:(\d+)\]')
TASK_DONE_CLEANUP_PATTERN = re.compile(r'\s*\[TASK_DONE:\d+\]\s*')
TASK_DONE_PARTIAL_PATTERN = re.compile(r'\s*\[TASK_DONE:\d*$')


class ResponseProcessor:
    """
    Processes LLM responses for task completion and text-to-speech.
    
    Handles:
    - Task completion marker extraction
    - Marker cleanup for display/TTS
    - Sentence splitting for streaming TTS
    """
    
    @staticmethod
    def extract_task_completions(text: str) -> List[int]:
        """
        Extract task IDs marked as completed in the response.
        
        Args:
            text: Response text containing potential task markers
            
        Returns:
            List of completed task IDs
        """
        return [int(match) for match in TASK_DONE_PATTERN.findall(text)]
    
    @staticmethod
    def strip_task_markers(text: str) -> str:
        """
        Remove task completion markers from text.
        
        Handles both complete and partial markers (for streaming).
        
        Args:
            text: Text potentially containing task markers
            
        Returns:
            Clean text without markers
        """
        # Remove complete markers
        text = TASK_DONE_CLEANUP_PATTERN.sub(' ', text)
        # Remove partial markers at end
        text = TASK_DONE_PARTIAL_PATTERN.sub('', text)
        # Remove partial end markers at start (e.g., "3]")
        text = re.sub(r'^\d*\]\s*', '', text)
        return text.strip()
    
    @staticmethod
    def extract_complete_sentences(buffer: str) -> Tuple[str, str]:
        """
        Extract complete sentences from a buffer.
        
        Sentences are delimited by ., !, ?, :, or ;
        
        Args:
            buffer: Text buffer potentially containing complete sentences
            
        Returns:
            Tuple of (complete_sentences, remaining_buffer)
        """
        # Find the last sentence ending
        last_end = -1
        for ending in SENTENCE_ENDINGS:
            # Check for "ending + space" or "ending at end of string"
            pos_with_space = buffer.rfind(ending + ' ')
            pos_at_end = len(buffer) - 1 if buffer.endswith(ending) else -1
            
            last_end = max(last_end, pos_with_space, pos_at_end)
        
        if last_end >= 0:
            # Include the sentence ending character
            complete = buffer[:last_end + 1].strip()
            remaining = buffer[last_end + 1:].lstrip()
            return complete, remaining
        
        return "", buffer


class ToolRouter:
    """
    Determines when to invoke RAG based on user input.
    
    Uses keyword matching as a fast pre-filter, then LLM classification.
    """
    
    CLASSIFICATION_PROMPT = """You are a tool router. Based on the user's message, decide if you need to look up product information.

Respond with ONLY one of:
- SEARCH: <query> - if user is asking about products, recommendations, or needs specific product details
- NONE - if user is just chatting, giving their name, or not asking about products

Examples:
- "My name is John" → NONE
- "I want to save for retirement" → SEARCH: retirement savings products
- "What products do you recommend?" → SEARCH: product recommendations
- "Hello" → NONE
- "I'm 35 and want growth" → SEARCH: growth investment products for 35 year old"""
    
    def __init__(self, trigger_keywords: str):
        """
        Initialize the tool router.
        
        Args:
            trigger_keywords: Comma-separated list of keywords that trigger RAG
        """
        self._keywords = [
            kw.strip().lower() 
            for kw in trigger_keywords.split(',') 
            if kw.strip()
        ]
    
    def should_check_rag(self, user_text: str) -> bool:
        """
        Quick check if RAG should be considered based on keywords.
        
        Args:
            user_text: User's message
            
        Returns:
            True if keywords match and LLM classification should proceed
        """
        if not self._keywords:
            return True  # No keywords = always check
        
        text_lower = user_text.lower()
        return any(kw in text_lower for kw in self._keywords)
    
    @staticmethod
    def parse_classification(result: str) -> Tuple[bool, str]:
        """
        Parse the LLM classification result.
        
        Args:
            result: Classification result from LLM
            
        Returns:
            Tuple of (should_search, search_query)
        """
        if result.startswith("SEARCH:"):
            query = result[7:].strip()
            return True, query
        return False, ""
