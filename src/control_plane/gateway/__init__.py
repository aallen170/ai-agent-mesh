"""
control_plane.gateway — AIMESH LiteLLM gateway client.

Exports
-------
GatewayClient   OpenAI-compatible client pointed at the LiteLLM proxy.
"""

from .client import GatewayClient

__all__ = ["GatewayClient"]
