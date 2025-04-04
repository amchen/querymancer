import os
import requests
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs.llm_result import LLMResult
from langchain_groq import ChatGroq
from langchain_sambanova import ChatSambaNovaCloud
from querymancer.config import ModelConfig, ModelProvider
from querymancer.optimizations import TokenOptimizationPipeline


class ChatSambaNova(BaseChatModel):
    """SambaNova Cloud chat model integration for Langchain."""

    api_url: str = "https://api.sambanova.ai/v1/chat/completions"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens_to_generate: int = 512,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize SambaNova chat model.

        Args:
            model: Model ID to use
            temperature: Temperature parameter (0.0-1.0)
            max_tokens_to_generate: Maximum tokens to generate
            api_key: SambaNova API key (optional, will use env var if not provided)
        """
        super().__init__(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens_to_generate = max_tokens_to_generate
        self.api_key = api_key or os.getenv("SAMBANOVA_API_KEY")

        if not self.api_key:
            raise ValueError("SambaNova API key is required")

    def _convert_messages_to_sambanova_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Convert Langchain messages to SambaNova API format.

        Args:
            messages: List of Langchain messages

        Returns:
            List of messages in SambaNova format
        """
        sambanova_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                raise ValueError(f"Unknown message type: {type(message)}")

            sambanova_messages.append({"role": role, "content": message.content})

        return sambanova_messages

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Call the SambaNova API.

        Args:
            messages: List of chat messages
            stop: Optional list of stop sequences
            run_manager: Optional callback manager

        Returns:
            Generated text response
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        sambanova_messages = self._convert_messages_to_sambanova_format(messages)

        payload = {
            "model": self.model,
            "messages": sambanova_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens_to_generate,
        }

        # Add stop sequences if provided
        if stop:
            payload["stop"] = stop

        # Add any additional parameters from kwargs
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise ValueError(f"Error from SambaNova API: {response.status_code} - {response.text}")

        try:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected API response format: {e}")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> LLMResult:
        """Generate a response from the model.

        This is used by Langchain for batched generation (not implemented).

        Args:
            messages: List of messages
            stop: Optional list of stop sequences
            run_manager: Optional callback manager

        Raises:
            NotImplementedError: This method is not currently implemented
        """
        raise NotImplementedError(
            "Batch generation is not currently supported for SambaNova models"
        )


class OptimizedChatGroq(ChatGroq):
    """Optimized Groq chat model with token and parameter optimization."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_completion_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize optimized Groq chat model.

        Args:
            model: Model ID to use
            temperature: Temperature parameter (0.0-1.0)
            max_completion_tokens: Maximum completion tokens
            api_key: Groq API key (optional, will use env var if not provided)
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            api_key=api_key,
            **kwargs,
        )

        # Instead of storing as an attribute, create it when needed

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Call the Groq API with optimization.

        Args:
            messages: List of chat messages
            stop: Optional list of stop sequences
            run_manager: Optional callback manager

        Returns:
            Generated text response
        """
        # Create token optimizer when needed instead of storing as an instance attribute
        token_optimizer = TokenOptimizationPipeline()

        # Apply token optimization to the last user message if it exists
        optimized_messages = list(messages)
        for i in range(len(optimized_messages) - 1, -1, -1):
            if isinstance(optimized_messages[i], HumanMessage):
                original_content = optimized_messages[i].content
                optimized_content = token_optimizer.refine_query(original_content)
                optimized_messages[i] = HumanMessage(content=optimized_content)
                break

        # Prune conversation history if needed
        if len(optimized_messages) > 10:  # arbitrary threshold
            system_messages = [m for m in optimized_messages if isinstance(m, SystemMessage)]
            recent_messages = optimized_messages[-9:]  # Keep last 9 messages

            # Make sure we keep at least one system message if it exists
            if system_messages and not any(isinstance(m, SystemMessage) for m in recent_messages):
                optimized_messages = [system_messages[0]] + recent_messages
            else:
                optimized_messages = recent_messages

        # Call the parent implementation with optimized messages
        return super()._call(
            messages=optimized_messages, stop=stop, run_manager=run_manager, **kwargs
        )


def create_llm(model_config: ModelConfig) -> BaseChatModel:
    """Create a language model based on the model configuration.

    Args:
        model_config: Model configuration

    Returns:
        BaseChatModel: Configured language model
    """
    if model_config.provider == ModelProvider.GROQ:
        return OptimizedChatGroq(
            model=model_config.name,
            temperature=model_config.temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif model_config.provider == ModelProvider.SAMBANOVA:
        return ChatSambaNovaCloud(
            model=model_config.name,
            max_tokens=1024,
            temperature=model_config.temperature,
        )
        """
        return ChatSambaNova(
            model=model_config.name,
            temperature=model_config.temperature,
            api_key=os.getenv("SAMBANOVA_API_KEY"),
        )
        """
