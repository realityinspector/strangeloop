"""OpenRouter LLM client with robust retry logic.

Adapted from persona2params/main.py robust_llm_call().
"""

import json
import os
import re
import time
from typing import Any, Optional, TypeVar

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel
from rich.console import Console

from strangeloop.schemas.simulation import TokenUsage

console = Console()

T = TypeVar("T", bound=BaseModel)


def extract_json_from_response(content: str) -> Optional[str]:
    """Extract JSON from potentially malformed LLM response."""
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*$", "", content)

    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    json_content = content[start_idx : end_idx + 1]
    json_content = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_content)
    json_content = re.sub(r",\s*}", "}", json_content)
    json_content = re.sub(r",\s*]", "]", json_content)

    open_braces = json_content.count("{")
    close_braces = json_content.count("}")
    if open_braces > close_braces:
        json_content += "}" * (open_braces - close_braces)
    elif close_braces > open_braces:
        json_content = "{" * (close_braces - open_braces) + json_content

    return json_content


class LLMClient:
    """OpenRouter LLM client with retry logic and token tracking."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        fallback_model: str = "meta-llama/llama-3.1-70b-instruct:free",
        api_key: Optional[str] = None,
    ):
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        self.model = model
        self.fallback_model = fallback_model
        self.token_usage = TokenUsage()

    def call(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        json_mode: bool = False,
        max_retries: int = 3,
        fallback_response: Any = None,
        model_override: Optional[str] = None,
    ) -> Any:
        """Make a robust LLM call with retry logic and JSON extraction.

        Adapted from persona2params robust_llm_call.
        """
        model = model_override or self.model
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt > 0 and last_error:
                    error_hint = f"\n\nPrevious attempt failed: {str(last_error)[:100]}. Please ensure valid JSON."
                    if isinstance(messages[-1]["content"], str):
                        messages = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + error_hint}]

                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": min(temperature + attempt * 0.1, 1.0),
                    "max_tokens": max_tokens,
                    "timeout": 30,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                # Track tokens
                if response.usage:
                    cost = 0.005 if "gpt-4" in model else 0.001
                    self.token_usage.add(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        cost_per_1k=cost,
                    )

                if not content or not content.strip():
                    last_error = "Empty response"
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                        continue
                    return fallback_response or {"error": "Empty response after retries"}

                if json_mode:
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        extracted = extract_json_from_response(content)
                        if extracted:
                            try:
                                return json.loads(extracted)
                            except json.JSONDecodeError:
                                pass
                        last_error = "JSON parsing failed"
                        if attempt < max_retries - 1:
                            time.sleep(2**attempt)
                            continue
                        return fallback_response or {"error": "JSON parsing failed", "raw": content[:500]}

                return content

            except (APIError, RateLimitError, APITimeoutError) as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    console.print(f"[dim yellow]API error, retrying in {wait}s...[/dim yellow]")
                    time.sleep(wait)
                    continue
                # Try fallback model on final attempt
                if model != self.fallback_model:
                    console.print(f"[dim yellow]Falling back to {self.fallback_model}[/dim yellow]")
                    return self.call(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        max_retries=1,
                        fallback_response=fallback_response,
                        model_override=self.fallback_model,
                    )
                return fallback_response or {"error": f"API error: {last_error}"}

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                return fallback_response or {"error": f"Unexpected: {last_error}"}

        return fallback_response or {"error": "All retries failed"}

    def call_structured(
        self,
        messages: list[dict[str, str]],
        model_class: type[T],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Optional[T]:
        """Call LLM and parse response into a Pydantic model."""
        result = self.call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        if isinstance(result, dict) and "error" not in result:
            try:
                return model_class(**result)
            except Exception:
                return None
        return None


def robust_llm_call(
    client: LLMClient,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> Any:
    """Module-level convenience wrapper."""
    return client.call(messages, **kwargs)
