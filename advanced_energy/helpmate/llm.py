"""Model factory — the ONLY place LLMs / embedding models are instantiated.

`get_llm()` and `get_embeddings()` read their configuration from
:mod:`helpmate.config`, which in turn reads ``.env``. To switch the Gemini
model (or add another provider) you change ``.env`` or extend the single
branch below — never any caller.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from helpmate import config


def with_retries(runnable: Runnable) -> Runnable:
    """Wrap a model/runnable with backoff so it survives free-tier rate limits.

    Gemini's free tier allows only a few requests/minute; the agent makes
    several calls per ticket. We retry on transient API errors (429/503) with
    exponential backoff so a run rides through the limit instead of failing.
    """
    try:
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
        retry_on: tuple = (ChatGoogleGenerativeAIError, Exception)
    except Exception:  # pragma: no cover
        retry_on = (Exception,)
    return runnable.with_retry(
        retry_if_exception_type=retry_on,
        stop_after_attempt=8,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 3, "max": 60, "exp_base": 2, "jitter": 2},
    )


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None) -> BaseChatModel:
    """Return a chat model for the configured provider/model.

    `temperature` overrides the configured default (used e.g. for a slightly
    more deterministic critic). Results are cached per temperature.
    """
    temp = config.LLM_TEMPERATURE if temperature is None else temperature
    provider = (config.LLM_PROVIDER or "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env "
                "(get a free key at https://aistudio.google.com/apikey)."
            )
        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=temp,
        )

    # Extend here for other providers (openai, ollama, ...) — single place.
    raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER!r}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the HuggingFace sentence-transformers embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
