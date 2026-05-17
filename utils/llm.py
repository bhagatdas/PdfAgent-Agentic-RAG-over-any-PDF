"""
LLM abstraction layer with model routing.
Provides a unified interface to switch between light/heavy/vision models.
All calls go through Ollama. Includes retry logic and LangSmith tracing.
"""

import logging
import time
from typing import Optional, Type
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Model routing map ──
MODEL_ROUTING = {
    "light": settings.ollama_model_light,       # Fast: classification, SQL gen, validation
    "heavy": settings.ollama_model_heavy,       # Powerful: reasoning, synthesis
    "vision": settings.ollama_model_vision,     # Multimodal: image captioning (preprocessing)
    "embed": settings.ollama_model_embed,       # Embeddings (handled separately)
}


@lru_cache(maxsize=8)
def get_llm(
    task_type: str = "light",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ChatOllama:
    """
    Get an LLM instance routed by task complexity.

    Args:
        task_type: "light" | "heavy" | "vision" — determines which model to use
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum output tokens

    Returns:
        ChatOllama instance configured for the task
    """
    model_name = MODEL_ROUTING.get(task_type, MODEL_ROUTING["light"])
    logger.info("Initializing LLM — task=%s, model=%s, temp=%.1f", task_type, model_name, temperature)

    return ChatOllama(
        model=model_name,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_predict=max_tokens,
        # Keep alive for 5 minutes to avoid cold starts
        keep_alive="5m",
    )


def get_structured_llm(
    output_schema: Type[BaseModel],
    task_type: str = "light",
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Get an LLM that returns structured (Pydantic) output.

    Args:
        output_schema: Pydantic model class defining the expected output structure
        task_type: "light" | "heavy" — determines which model to use
        temperature: Sampling temperature

    Returns:
        LLM bound to produce structured output matching the schema

    Example:
        structured_llm = get_structured_llm(QueryAnalysis, task_type="light")
        result: QueryAnalysis = structured_llm.invoke(prompt)
    """
    llm = get_llm(task_type=task_type, temperature=temperature)
    logger.info("Binding structured output — schema=%s", output_schema.__name__)
    return llm.with_structured_output(output_schema)


def invoke_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    task_type: str = "light",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> str:
    """
    Invoke an LLM with retry logic. Returns raw text response.

    Args:
        prompt: User prompt
        system_prompt: Optional system instructions
        task_type: "light" | "heavy" | "vision"
        temperature: Sampling temperature
        max_retries: Number of retries on failure

    Returns:
        LLM response as string
    """
    llm = get_llm(task_type=task_type, temperature=temperature)
    messages = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            response = llm.invoke(messages)
            elapsed = (time.time() - start) * 1000

            logger.debug(
                "LLM response — task=%s, tokens=%s, time=%.0fms",
                task_type,
                getattr(response, "usage_metadata", "N/A"),
                elapsed,
            )
            return response.content

        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)
            if attempt == max_retries:
                logger.error("LLM call exhausted retries — task=%s, error=%s", task_type, e)
                raise
            time.sleep(1.0 * (attempt + 1))  # Linear backoff

    return ""  # Unreachable, but satisfies type checker


def invoke_vision_llm(
    prompt: str,
    image_paths: list[str],
    max_retries: int = 2,
) -> str:
    """
    Invoke the vision model with images. Used during preprocessing only.

    Args:
        prompt: Text prompt describing what to extract/caption
        image_paths: List of local image file paths
        max_retries: Number of retries on failure

    Returns:
        Vision model response as string
    """
    import base64

    llm = get_llm(task_type="vision", temperature=0.0)

    # Build multimodal message content
    content = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        except FileNotFoundError:
            logger.warning("Image not found: %s", img_path)

    messages = [HumanMessage(content=content)]

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.warning("Vision LLM failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)
            if attempt == max_retries:
                raise
            time.sleep(1.0 * (attempt + 1))

    return ""
