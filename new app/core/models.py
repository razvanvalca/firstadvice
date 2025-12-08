"""
Core domain models for the voice agent.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Task:
    """A task for the agent to complete during the conversation."""
    id: int
    description: str
    completed: bool = False


@dataclass
class ConversationMessage:
    """A message in the conversation history."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class SessionState:
    """State for a single voice agent session."""
    
    # Conversation
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    
    # Processing state
    is_processing: bool = False
    is_speaking: bool = False
    is_interrupted: bool = False
    is_audio_playing: bool = False
    
    # Pending input handling
    pending_interrupt_text: Optional[str] = None
    pending_user_text: Optional[str] = None
    
    # Echo detection
    recent_agent_speech: List[str] = field(default_factory=list)
    last_shown_transcript: Optional[str] = None
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self.conversation_history.append(ConversationMessage(role="user", content=content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self.conversation_history.append(ConversationMessage(role="assistant", content=content))
    
    def get_messages_for_llm(self) -> List[dict]:
        """Get conversation history in LLM-compatible format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]
    
    def track_agent_speech(self, text: str, max_items: int = 3) -> None:
        """Track recent agent speech for echo detection."""
        self.recent_agent_speech.append(text)
        if len(self.recent_agent_speech) > max_items:
            self.recent_agent_speech.pop(0)
    
    def clear_agent_speech_history(self) -> None:
        """Clear the echo detection history."""
        self.recent_agent_speech = []
    
    def is_echo(self, text: str) -> bool:
        """Check if transcribed text is likely echo from agent's speech."""
        text_lower = text.lower().strip()
        for agent_text in self.recent_agent_speech:
            agent_lower = agent_text.lower()
            if text_lower in agent_lower or agent_lower.startswith(text_lower):
                return True
        return False
    
    def mark_task_completed(self, task_id: int) -> Optional[Task]:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of the task to mark complete
            
        Returns:
            The completed task, or None if not found
        """
        for task in self.tasks:
            if task.id == task_id and not task.completed:
                task.completed = True
                return task
        return None
    
    def reset_processing_state(self) -> None:
        """Reset all processing-related state flags."""
        self.is_processing = False
        self.is_speaking = False
        self.is_audio_playing = False
        self.clear_agent_speech_history()


@dataclass
class PromptBuilder:
    """Builds the system prompt with dynamic content."""
    
    base_prompt: str
    product_summary: str = ""
    tasks: List[Task] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, prompt_path: Path) -> "PromptBuilder":
        """
        Create a PromptBuilder from a prompt file.
        
        Args:
            prompt_path: Path to the system prompt markdown file
            
        Returns:
            PromptBuilder instance
        """
        if prompt_path.exists():
            base_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            print(f"[PromptBuilder] Warning: Prompt file not found: {prompt_path}")
            base_prompt = "You are a helpful assistant."
        
        return cls(base_prompt=base_prompt)
    
    def with_product_summary(self, summary: str) -> "PromptBuilder":
        """Set the product summary."""
        self.product_summary = summary
        return self
    
    def with_tasks(self, tasks: List[Task]) -> "PromptBuilder":
        """Set the tasks list."""
        self.tasks = tasks
        return self
    
    def build(self) -> str:
        """Build the complete system prompt."""
        prompt = self.base_prompt
        
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
