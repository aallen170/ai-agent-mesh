"""
client.py — GatewayClient: thin wrapper around the AIMESH LiteLLM proxy.

Every model in the mesh (Ollama on local devices, RunPod serverless, and
Anthropic Claude) is registered in infra/litellm_config.yaml and exposed
through one OpenAI-compatible endpoint at http://localhost:4000/v1.

Workers and the LangGraph orchestrator use this client so they never need
to know which backend is serving a particular model — they just call
``complete(model="tier2/llama3.1-70b", messages=[...])``.

Configuration
-------------
LITELLM_BASE_URL    URL of the LiteLLM proxy (default: http://localhost:4000/v1)
LITELLM_MASTER_KEY  Auth key for the gateway  (default: sk-aimesh-local)

Both can also be passed directly to the constructor.

Usage
-----
    from src.control_plane.gateway import GatewayClient

    client = GatewayClient()

    # Single completion
    response = client.complete(
        model="tier2/llama3.1-70b-desktop",
        messages=[{"role": "user", "content": "Explain Redis Streams in one paragraph."}],
        temperature=0.3,
        max_tokens=256,
    )
    print(response.choices[0].message.content)

    # List all models currently registered with the gateway
    print(client.list_models())
"""
from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:4000/v1"
_DEFAULT_API_KEY = "sk-aimesh-local"


class GatewayClient:
    """
    OpenAI-compatible client pointed at the AIMESH LiteLLM gateway.

    All five tiers of the mesh are accessible through this single client.
    Model names follow the convention ``tier{N}/{model-slug}``, e.g.:

        tier0/llama3.2-3b-ipad1
        tier1/llama3.1-8b
        tier2/llama3.1-70b-desktop
        tier3/llama3.1-70b-runpod
        tier4/claude-sonnet

    Parameters
    ----------
    base_url    URL of the LiteLLM proxy.  Defaults to the LITELLM_BASE_URL
                environment variable, falling back to http://localhost:4000/v1.
    api_key     Master key for the gateway.  Defaults to the LITELLM_MASTER_KEY
                environment variable, falling back to "sk-aimesh-local".
    timeout     HTTP timeout in seconds for a single model call (default: 120).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL", _DEFAULT_BASE_URL)
        self.api_key = api_key or os.getenv("LITELLM_MASTER_KEY", _DEFAULT_API_KEY)
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )
        logger.debug("GatewayClient initialised (base_url=%s)", self.base_url)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        Send a chat completion request through the LiteLLM gateway.

        Parameters
        ----------
        model       LiteLLM model name from litellm_config.yaml,
                    e.g. ``"tier2/llama3.1-70b-desktop"``.
        messages    OpenAI-format message list,
                    e.g. ``[{"role": "user", "content": "Hello"}]``.
        **kwargs    Additional parameters forwarded to the model
                    (``temperature``, ``max_tokens``, ``stream``, etc.).
                    Unsupported params are silently dropped by LiteLLM
                    (``drop_params: true`` in config).

        Returns
        -------
        openai.types.chat.ChatCompletion

        Raises
        ------
        openai.APIConnectionError   Gateway unreachable.
        openai.AuthenticationError  Bad master key.
        openai.APIStatusError       Model error or bad request.
        """
        logger.debug("GatewayClient.complete model=%s msgs=%d", model, len(messages))
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

    def list_models(self) -> list[str]:
        """
        Return the model IDs currently registered with the gateway.

        Requires the gateway to be running.  Raises openai.APIConnectionError
        if it is not reachable.
        """
        models = self._client.models.list()
        return [m.id for m in models.data]

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------

    @property
    def raw(self) -> OpenAI:
        """
        The underlying ``openai.OpenAI`` client.

        Use this for endpoints not wrapped by GatewayClient (e.g. embeddings,
        streaming, or the /completions endpoint).
        """
        return self._client
