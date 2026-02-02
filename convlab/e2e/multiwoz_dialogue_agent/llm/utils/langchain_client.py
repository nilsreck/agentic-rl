import os
from typing import Optional

from langchain_openai import ChatOpenAI


def create_openai_client(
    model: str,
    temperature: float,
    # max_tokens: int,
) -> Optional[ChatOpenAI]:
    """
    Create a ChatOpenAI client with the specified configuration.

    Args:
        model: The OpenAI model to use
        temperature: Temperature setting for the model
        max_tokens: Maximum number of tokens to generate
        proxy: Optional proxy URL

    Returns:
        ChatOpenAI or None: Configured ChatOpenAI instance or None for testing
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "dummy-key-for-testing"

    config = {
        "model": model,
        "temperature": temperature,
        # "max_tokens": max_tokens,
        "api_key": api_key,
    }

    try:
        return ChatOpenAI(**config)
    except Exception as e:
        print(f"Warning: Could not create OpenAI client: {e}")
        # Return a mock client for testing
        return None
