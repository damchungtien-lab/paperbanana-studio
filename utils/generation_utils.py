# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
import inspect
import re
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import httpx
import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path
from urllib.parse import urlparse

from utils.console_utils import setup_console

setup_console()

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize a user-provided OpenAI-compatible base URL."""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.path or parsed.path == "/":
        return f"{url}/v1"
    return url


def normalize_google_genai_base_url(base_url: str) -> str:
    """Normalize a Gemini Generate Content base URL root."""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return ""
    for suffix in ("/v1beta", "/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def get_custom_endpoint_settings(kind: str) -> dict[str, str]:
    """Resolve endpoint-specific custom API settings with shared fallback."""
    kind = str(kind or "").strip().lower()
    if kind not in {"text", "image"}:
        raise ValueError(f"Unsupported custom endpoint kind: {kind}")

    shared_base_url = normalize_openai_base_url(
        get_config_val("api_base_urls", "custom_base_url", "CUSTOM_API_BASE_URL", "")
    )
    shared_api_key = get_config_val("api_keys", "custom_api_key", "CUSTOM_API_KEY", "")

    specific_base_url = normalize_openai_base_url(
        get_config_val(
            "api_base_urls",
            f"custom_{kind}_base_url",
            f"CUSTOM_{kind.upper()}_API_BASE_URL",
            "",
        )
    )
    specific_api_key = get_config_val(
        "api_keys",
        f"custom_{kind}_api_key",
        f"CUSTOM_{kind.upper()}_API_KEY",
        "",
    )

    return {
        "base_url": specific_base_url or shared_base_url,
        "api_key": specific_api_key or shared_api_key,
        "specific_base_url": specific_base_url,
        "specific_api_key": specific_api_key,
        "shared_base_url": shared_base_url,
        "shared_api_key": shared_api_key,
    }


def list_openai_compatible_models(
    base_url: str,
    api_key: str = "",
    timeout_seconds: float = 20.0,
) -> list[str]:
    """Fetch model ids from an OpenAI-compatible `/models` endpoint."""
    normalized_base_url = normalize_openai_base_url(base_url)
    if not normalized_base_url:
        return []

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = httpx.get(
        f"{normalized_base_url}/models",
        headers=headers,
        timeout=timeout_seconds,
        follow_redirects=True,
    )
    response.raise_for_status()

    payload = response.json()
    model_items = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = []
    for item in model_items:
        if isinstance(item, dict):
            model_id = str(item.get("id", "")).strip()
            if model_id:
                model_ids.append(model_id)

    return sorted(set(model_ids))


def validate_openai_compatible_endpoint(
    base_url: str,
    api_key: str = "",
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Validate an OpenAI-compatible endpoint and return discovered models."""
    normalized_base_url = normalize_openai_base_url(base_url)
    if not normalized_base_url:
        return {
            "ok": False,
            "base_url": "",
            "models": [],
            "error": "Base URL is empty.",
        }

    try:
        models = list_openai_compatible_models(
            normalized_base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        return {
            "ok": True,
            "base_url": normalized_base_url,
            "models": models,
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "base_url": normalized_base_url,
            "models": [],
            "error": str(exc),
        }


def normalize_generation_image_size(image_size: str) -> str:
    """Normalize image-size labels while preserving Gemini's required `K` casing."""
    normalized = str(image_size or "").strip()
    upper = normalized.upper()
    lower = normalized.lower()
    if upper in {"1K", "2K", "4K"}:
        return upper
    if lower in {"512px", "512"}:
        return "512px"
    return "1K"


def resolve_openai_image_size(aspect_ratio: str, image_size: str) -> str:
    """Map generic generation settings to common OpenAI-compatible image sizes."""
    normalized_ratio = str(aspect_ratio or "1:1").strip()
    normalized_image_size = normalize_generation_image_size(image_size).lower()

    if normalized_ratio in {"21:9", "16:9", "3:2"}:
        return "1536x1024" if normalized_image_size in {"2k", "4k"} else "1024x1024"
    if normalized_ratio in {"9:16", "2:3"}:
        return "1024x1536" if normalized_image_size in {"2k", "4k"} else "1024x1024"
    return "1024x1024"

# Initialize clients lazily or with robust defaults
api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
if api_key:
    gemini_client = genai.Client(api_key=api_key)
    print("Initialized Gemini Client with API Key")
else:
    print("Warning: Could not initialize Gemini Client. Missing credentials.")
    gemini_client = None


anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
if anthropic_api_key:
    anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
    print("Initialized Anthropic Client with API Key")
else:
    print("Warning: Could not initialize Anthropic Client. Missing credentials.")
    anthropic_client = None

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
if openai_api_key:
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    print("Initialized OpenAI Client with API Key")
else:
    print("Warning: Could not initialize OpenAI Client. Missing credentials.")
    openai_client = None

openrouter_api_key = get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")
if openrouter_api_key:
    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    print("Initialized OpenRouter Client with API Key")
else:
    print("Warning: Could not initialize OpenRouter Client. Missing credentials.")
    openrouter_client = None

custom_text_settings = get_custom_endpoint_settings("text")
custom_text_api_key = custom_text_settings["api_key"]
custom_text_base_url = custom_text_settings["base_url"]
if custom_text_base_url:
    custom_text_client = AsyncOpenAI(
        base_url=custom_text_base_url,
        api_key=custom_text_api_key or "paperbanana-local",
    )
    print(f"Initialized Custom Text OpenAI-Compatible Client at {custom_text_base_url}")
else:
    print("Warning: Could not initialize Custom Text OpenAI-Compatible Client. Missing base URL.")
    custom_text_client = None

custom_image_settings = get_custom_endpoint_settings("image")
custom_image_api_key = custom_image_settings["api_key"]
custom_image_base_url = custom_image_settings["base_url"]
if custom_image_base_url:
    custom_image_client = AsyncOpenAI(
        base_url=custom_image_base_url,
        api_key=custom_image_api_key or "paperbanana-local",
    )
    print(f"Initialized Custom Image OpenAI-Compatible Client at {custom_image_base_url}")
else:
    print("Warning: Could not initialize Custom Image OpenAI-Compatible Client. Missing base URL.")
    custom_image_client = None


def get_provider_status() -> dict[str, bool]:
    """Return simple availability flags for UI and routing diagnostics."""
    return {
        "gemini": gemini_client is not None,
        "anthropic": anthropic_client is not None,
        "openai": openai_client is not None,
        "openrouter": openrouter_client is not None,
        "custom_text": custom_text_client is not None,
        "custom_image": custom_image_client is not None,
        "custom": custom_text_client is not None or custom_image_client is not None,
    }


def resolve_model_provider(model_name: str) -> tuple[str, str]:
    """Resolve provider and actual model id from an optional provider-prefixed name."""
    model_name = str(model_name or "").strip()

    prefix_map = {
        "openrouter/": "openrouter",
        "custom/": "custom",
        "openai/": "openai",
        "anthropic/": "anthropic",
        "gemini/": "gemini",
    }
    for prefix, provider in prefix_map.items():
        if model_name.startswith(prefix):
            return provider, model_name[len(prefix):]

    if model_name.startswith("claude-"):
        return "anthropic", model_name
    if any(model_name.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
        return "openai", model_name
    if model_name.startswith("gemini"):
        return "gemini", model_name

    if openrouter_client is not None:
        return "openrouter", _to_openrouter_model_id(model_name)
    if gemini_client is not None:
        return "gemini", model_name
    if anthropic_client is not None:
        return "anthropic", model_name
    if openai_client is not None:
        return "openai", model_name
    if custom_text_client is not None or custom_image_client is not None:
        return "custom", model_name

    raise RuntimeError(
        "No API client available. Please configure at least one API key "
        "or a custom OpenAI-compatible base URL in configs/model_config.yaml."
    )


def uses_openai_images_api(model_name: str) -> bool:
    """Return True when the model should use the OpenAI Images API path."""
    provider, actual_model = resolve_model_provider(model_name)
    return provider in {"openai", "custom"} and actual_model.startswith("gpt-image")



def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["image_base64"]),
                        mime_type="image/jpeg",
                    )
                )
            elif "data" in item:
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["data"]),
                        mime_type=item.get("mime_type", "image/jpeg"),
                    )
                )
    return gemini_parts


def _convert_to_gemini_json_contents(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert the shared content format into Gemini Generate Content JSON payloads."""
    parts = []
    for item in contents:
        item_type = item.get("type")
        if item_type == "text":
            parts.append({"text": item["text"]})
            continue

        if item_type == "image":
            source = item.get("source", {})
            image_b64 = ""
            mime_type = "image/jpeg"
            if source.get("type") == "base64":
                image_b64 = source.get("data", "")
                mime_type = source.get("media_type", mime_type)
            elif "image_base64" in item:
                image_b64 = item.get("image_base64", "")
            elif "data" in item:
                image_b64 = item.get("data", "")
                mime_type = item.get("mime_type", mime_type)

            if image_b64:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    }
                )
    return [{"parts": parts}]


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client was not initialized: missing Google API key. "
            "Please set GOOGLE_API_KEY in environment, or configure api_keys.google_api_key in configs/model_config.yaml."
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client

            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if (
                "nanoviz" in model_name
                or "image" in model_name
            ):
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                    if part.text is not None
                ]
            result_list.extend([r for r in raw_response_list if r and r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            
            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2 ** attempt), 30)
            
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list


async def _call_gemini_generate_content_http_with_retry_async(
    endpoint_root,
    api_key,
    model_name,
    contents,
    config,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """Call Gemini Generate Content over HTTP for custom Gemini-compatible endpoints."""
    if not endpoint_root:
        raise RuntimeError("Gemini-compatible endpoint root is empty.")

    endpoint_url = f"{normalize_google_genai_base_url(endpoint_root)}/v1beta/models/{model_name}:generateContent"
    context_msg = f" for {error_context}" if error_context else ""
    image_size = normalize_generation_image_size(config.get("image_size", "1K"))
    aspect_ratio = str(config.get("aspect_ratio", "1:1")).strip() or "1:1"
    system_prompt = str(config.get("system_prompt", "") or "").strip()
    temperature = config.get("temperature", 1.0)
    response_modalities = config.get("response_modalities") or ["IMAGE"]

    payload = {
        "contents": _convert_to_gemini_json_contents(contents),
        "generationConfig": {
            "responseModalities": response_modalities,
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            },
            "temperature": temperature,
        },
    }
    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key} if api_key else None

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                response = await client.post(
                    endpoint_url,
                    headers=headers,
                    params=params,
                    json=payload,
                )
            response.raise_for_status()
            response_payload = response.json()

            candidates = response_payload.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini Generate Content returned no candidates.")

            images = []
            for candidate in candidates:
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    inline_data = part.get("inlineData") or part.get("inline_data")
                    if inline_data and inline_data.get("data"):
                        images.append(inline_data["data"])

            if images:
                return images

            raise RuntimeError("Gemini Generate Content returned no inline image data.")
        except httpx.HTTPStatusError as exc:
            current_delay = min(retry_delay * (2 ** attempt), 60)
            body = exc.response.text
            print(
                f"Custom Gemini image gen attempt {attempt + 1} failed{context_msg}: "
                f"HTTP {exc.response.status_code} - {body}. Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]
        except Exception as exc:
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"Custom Gemini image gen attempt {attempt + 1} failed{context_msg}: {exc}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    
    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    
    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            elif "data" in item:
                media_type = item.get("mime_type", "image/jpeg")
                data_url = f"data:{media_type};base64,{item['data']}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def _call_openai_compatible_chat_with_retry_async(
    client,
    provider_name,
    model_name,
    contents,
    config,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """Shared async retry wrapper for OpenAI-compatible chat APIs."""
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []
    context_msg = f" for {error_context}" if error_context else ""

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            content = first_response.choices[0].message.content or ""
            if not content.strip():
                print(f"OpenAI returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            print(
                f"{provider_name} validation attempt {attempt + 1} failed{context_msg}: "
                f"{error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} {provider_name} attempts failed to validate "
            f"the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """ASYNC: Call the OpenAI chat API with asynchronous retry logic."""
    if openai_client is None:
        raise RuntimeError(
            "OpenAI client was not initialized: missing API key. "
            "Please set OPENAI_API_KEY in environment, or configure "
            "api_keys.openai_api_key in configs/model_config.yaml."
        )

    provider, actual_model = resolve_model_provider(model_name)
    if provider == "openai":
        model_name = actual_model

    return await _call_openai_compatible_chat_with_retry_async(
        client=openai_client,
        provider_name="OpenAI",
        model_name=model_name,
        contents=contents,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def call_custom_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """ASYNC: Call a custom OpenAI-compatible chat API with asynchronous retry logic."""
    if custom_text_client is None:
        raise RuntimeError(
            "Custom text OpenAI-compatible client was not initialized: missing base URL. "
            "Please set CUSTOM_TEXT_API_BASE_URL/CUSTOM_API_BASE_URL in environment, or configure "
            "api_base_urls.custom_text_base_url/api_base_urls.custom_base_url in configs/model_config.yaml."
        )

    provider, actual_model = resolve_model_provider(model_name)
    if provider == "custom":
        model_name = actual_model

    return await _call_openai_compatible_chat_with_retry_async(
        client=custom_text_client,
        provider_name="Custom endpoint",
        model_name=model_name,
        contents=contents,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def _call_openai_compatible_images_api_with_retry_async(
    client,
    provider_name,
    model_name,
    prompt,
    config,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """Shared retry wrapper for OpenAI-compatible images.generate APIs."""
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via {provider_name}, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for {provider_name} image generation model "
                f"{model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call an OpenAI-compatible Images API (OpenAI or custom endpoint).
    """
    provider, actual_model = resolve_model_provider(model_name)
    if provider == "custom":
        if custom_image_client is None:
            raise RuntimeError(
                "Custom image OpenAI-compatible client was not initialized: missing base URL. "
                "Please set CUSTOM_IMAGE_API_BASE_URL/CUSTOM_API_BASE_URL in environment, or configure "
                "api_base_urls.custom_image_base_url/api_base_urls.custom_base_url in configs/model_config.yaml."
            )
        client = custom_image_client
        provider_name = "Custom endpoint"
    else:
        if openai_client is None:
            raise RuntimeError(
                "OpenAI client was not initialized: missing API key. "
                "Please set OPENAI_API_KEY in environment, or configure "
                "api_keys.openai_api_key in configs/model_config.yaml."
            )
        client = openai_client
        provider_name = "OpenAI"
        actual_model = model_name if provider != "openai" else actual_model

    return await _call_openai_compatible_images_api_with_retry_async(
        client=client,
        provider_name=provider_name,
        model_name=actual_model,
        prompt=prompt,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def call_openrouter_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter API (OpenAI-compatible) with asynchronous retry logic.
    """
    if openrouter_client is None:
        raise RuntimeError(
            "OpenRouter client was not initialized: missing API key. "
            "Please set OPENROUTER_API_KEY in environment, or configure "
            "api_keys.openrouter_api_key in configs/model_config.yaml."
        )

    provider, actual_model = resolve_model_provider(model_name)
    if provider == "openrouter":
        model_name = _to_openrouter_model_id(actual_model)
    else:
        model_name = _to_openrouter_model_id(model_name)

    return await _call_openai_compatible_chat_with_retry_async(
        client=openrouter_client,
        provider_name="OpenRouter",
        model_name=model_name,
        contents=contents,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def _call_openai_compatible_chat_image_generation_with_retry_async(
    endpoint_url,
    provider_name,
    model_name,
    contents,
    config,
    api_key="",
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """Call an OpenAI-compatible chat/completions endpoint with image output."""
    system_prompt = config.get("system_prompt", "")
    temperature = config.get("temperature", 1.0)
    aspect_ratio = config.get("aspect_ratio", "1:1")
    image_size = config.get("image_size", "1K")
    openai_contents = _convert_to_openai_format(contents)

    image_config = {}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config["image_size"] = image_size

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": openai_contents},
        ],
        "temperature": temperature,
        "modalities": ["image", "text"],
    }
    if image_config:
        payload["image_config"] = image_config

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    endpoint_url,
                    headers=headers,
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                print(f"[Warning]: OpenRouter image generation returned no choices, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

            message = choices[0].get("message", {})

            # Try extracting from inline_data in content (Gemini-style)
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "inline_data" in part:
                        b64_data = part["inline_data"].get("data", "")
                        if b64_data:
                            return [b64_data]

            # Try extracting from images field (OpenRouter standard)
            images = message.get("images")
            if images and len(images) > 0:
                img_item = images[0]
                if isinstance(img_item, dict):
                    data_url = img_item.get("image_url", {}).get("url", "")
                else:
                    data_url = str(img_item)
                if "," in data_url:
                    b64_data = data_url.split(",", 1)[1]
                else:
                    b64_data = data_url
                if b64_data:
                    return [b64_data]

            # Try extracting base64 from text content
            if isinstance(content, str):
                if content.startswith("data:image"):
                    if "," in content:
                        b64_data = content.split(",", 1)[1]
                    else:
                        b64_data = content
                    if b64_data:
                        return [b64_data]

                markdown_image_match = re.search(
                    r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)",
                    content,
                )
                if markdown_image_match:
                    return [markdown_image_match.group(1)]

            print(f"[Warning]: {provider_name} image generation returned no images, retrying...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            continue

        except httpx.HTTPStatusError as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"{provider_name} image gen attempt {attempt + 1} failed{context_msg}: "
                f"HTTP {e.response.status_code} - {e.response.text}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"{provider_name} image gen attempt {attempt + 1} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_openrouter_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter image generation via direct chat/completions.
    """
    if not openrouter_api_key:
        raise RuntimeError(
            "OpenRouter client was not initialized: missing API key."
        )

    provider, actual_model = resolve_model_provider(model_name)
    if provider == "openrouter":
        model_name = _to_openrouter_model_id(actual_model)
    else:
        model_name = _to_openrouter_model_id(model_name)

    return await _call_openai_compatible_chat_image_generation_with_retry_async(
        endpoint_url="https://openrouter.ai/api/v1/chat/completions",
        provider_name="OpenRouter",
        model_name=model_name,
        contents=contents,
        config=config,
        api_key=openrouter_api_key,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def call_custom_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call a custom OpenAI-compatible chat/completions endpoint for image generation.
    """
    if not custom_image_base_url:
        raise RuntimeError(
            "Custom image OpenAI-compatible client was not initialized: missing base URL. "
            "Please set CUSTOM_IMAGE_API_BASE_URL/CUSTOM_API_BASE_URL in environment, or configure "
            "api_base_urls.custom_image_base_url/api_base_urls.custom_base_url in configs/model_config.yaml."
        )

    provider, actual_model = resolve_model_provider(model_name)
    if provider == "custom":
        model_name = actual_model

    if model_name.startswith("gemini-") and "image" in model_name:
        return await _call_gemini_generate_content_http_with_retry_async(
            endpoint_root=custom_image_base_url,
            api_key=custom_image_api_key,
            model_name=model_name,
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    endpoint_url = f"{custom_image_base_url.rstrip('/')}/chat/completions"
    return await _call_openai_compatible_chat_image_generation_with_retry_async(
        endpoint_url=endpoint_url,
        provider_name="Custom endpoint",
        model_name=model_name,
        contents=contents,
        config=config,
        api_key=custom_image_api_key,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


def _to_openrouter_model_id(model_name: str) -> str:
    """Convert a bare model name to OpenRouter format (provider/model).

    OpenRouter requires model IDs like 'google/gemini-3-pro-preview'.
    If the name already contains '/', assume it's already qualified.
    Otherwise, prefix with 'google/' for Gemini models.
    """
    if "/" in model_name:
        return model_name
    if model_name.startswith("gemini"):
        return f"google/{model_name}"
    return model_name


async def call_image_model_with_retry_async(
    model_name,
    prompt,
    contents,
    config,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """
    Unified router for image generation and image editing backends.

    Supported routes:
      - Gemini native image generation
      - OpenAI / custom OpenAI-compatible Images API (`gpt-image-*`)
      - OpenRouter / custom OpenAI-compatible chat image generation
    """
    provider, actual_model = resolve_model_provider(model_name)
    image_config = {
        "system_prompt": config.get("system_prompt", ""),
        "temperature": config.get("temperature", 1.0),
        "aspect_ratio": config.get("aspect_ratio", "1:1"),
        "image_size": normalize_generation_image_size(config.get("image_size", "1K")),
        "size": config.get("size", ""),
        "quality": config.get("quality", "high"),
        "background": config.get("background", "opaque"),
        "output_format": config.get("output_format", "png"),
    }
    if not image_config["size"]:
        image_config["size"] = resolve_openai_image_size(
            image_config["aspect_ratio"],
            image_config["image_size"],
        )

    if provider == "gemini":
        if gemini_client is None:
            raise RuntimeError(
                "Gemini client was not initialized: missing GOOGLE_API_KEY."
            )
        return await call_gemini_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=image_config["system_prompt"],
                temperature=image_config["temperature"],
                candidate_count=1,
                max_output_tokens=50000,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=image_config["aspect_ratio"],
                    image_size=image_config["image_size"],
                ),
            ),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if provider in {"openai", "custom"} and actual_model.startswith("gpt-image"):
        return await call_openai_image_generation_with_retry_async(
            model_name=model_name,
            prompt=prompt,
            config=image_config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if provider == "openrouter":
        return await call_openrouter_image_generation_with_retry_async(
            model_name=model_name,
            contents=contents,
            config=image_config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if provider == "custom":
        return await call_custom_image_generation_with_retry_async(
            model_name=model_name,
            contents=contents,
            config=image_config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    raise RuntimeError(
        f"Provider '{provider}' does not currently support image generation/editing "
        f"for model '{model_name}'."
    )


async def call_model_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    Unified router that dispatches to the correct provider based on model_name.

    Routing rules:
      1. Explicit prefix overrides: "openrouter/" -> OpenRouter, "claude-" -> Anthropic,
         "gpt-"/"o1-"/"o3-"/"o4-" -> OpenAI
      2. No prefix: auto-detect based on which API key is configured.
         Priority: OpenRouter > Gemini > Anthropic > OpenAI
    """
    provider, actual_model = resolve_model_provider(model_name)
    if provider == "openrouter":
        actual_model = _to_openrouter_model_id(actual_model)

    if provider == "gemini":
        return await call_gemini_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    # Convert Gemini GenerateContentConfig -> dict for OpenAI/Claude/OpenRouter
    cfg_dict = {
        "system_prompt": config.system_instruction if hasattr(config, "system_instruction") else "",
        "temperature": config.temperature if hasattr(config, "temperature") else 1.0,
        "candidate_num": config.candidate_count if hasattr(config, "candidate_count") else 1,
        "max_completion_tokens": config.max_output_tokens if hasattr(config, "max_output_tokens") else 50000,
    }

    call_fn = {
        "openrouter": call_openrouter_with_retry_async,
        "custom": call_custom_with_retry_async,
        "anthropic": call_claude_with_retry_async,
        "openai": call_openai_with_retry_async,
    }[provider]

    return await call_fn(
        model_name=actual_model,
        contents=contents,
        config=cfg_dict,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )
