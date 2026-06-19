"""
llm.py — Single place to get the language model.

Default: local Ollama (free, offline, private).
Fallback: Gemini free tier (set LLM_PROVIDER=gemini + GOOGLE_API_KEY).

Change the model without touching any other file:
  set PARAKH_MODEL=mistral:7b   (or any model you have pulled in Ollama)
"""

import os


def get_llm():
    """Return a LangChain chat model. Reads config from environment."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = os.getenv("PARAKH_MODEL", "llama3.2")
        print(f"[llm] Ollama  model={model}")
        return ChatOllama(model=model, temperature=0)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set. Export it before running.")
        model = os.getenv("PARAKH_MODEL", "gemini-1.5-flash")
        print(f"[llm] Gemini  model={model}")
        return ChatGoogleGenerativeAI(
            model=model, temperature=0, google_api_key=api_key
        )

    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Use 'ollama' or 'gemini'.")
