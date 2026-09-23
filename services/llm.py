"""
Gemini LLM generation service using the lightweight google-genai SDK.
Handles strict turn alternation and timeout management without LangChain overhead.
"""

from google import genai
from google.genai import types


def generate_chat(
    model_name: str,
    api_key: str,
    system_prompt: str,
    history: list[dict],
    user_message: str,
    timeout_s: float = 25.0,
) -> str:
    """
    Call Gemini generate_content with system prompt, chat history, and user message.
    Ensures strict turn alternation (user -> model -> user) to avoid 400 Bad Request.
    """
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # SDK expects milliseconds
    )

    # Build contents list ensuring strict alternation between user and model
    contents = []
    last_role = None
    for msg in history:
        role = "user" if msg.get("role") in ["user", "human"] else "model"
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        if role == last_role:
            contents[-1]["parts"][0]["text"] += "\n\n" + text
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
            last_role = role

    # Append current user question
    if contents and contents[-1]["role"] == "user":
        contents[-1]["parts"][0]["text"] += "\n\n" + user_message
    else:
        contents.append({"role": "user", "parts": [{"text": user_message}]})

    config_kwargs = {
        "system_instruction": system_prompt,
        "temperature": 0.2,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    try:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    config = types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    return response.text or ""


def generate_text(model_name: str, api_key: str, prompt: str, timeout_s: float = 25.0) -> str:
    """Simple single-turn text generation with Gemini."""
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # SDK expects milliseconds
    )
    config_kwargs = {
        "temperature": 0.2,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    try:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    return response.text or ""
