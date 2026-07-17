"""Anthropic (Claude) on Google Vertex AI provider profile.

vertex-anthropic: Claude models via Google Cloud, using the Anthropic Messages
API at Vertex's ``publishers/anthropic/models/*:rawPredict`` surface — NOT the
OpenAI-compatible openapi endpoint the Gemini ``vertex`` provider uses.

Auth is the same ADC/OAuth2 credential path as ``vertex`` (short-lived tokens
minted from a service-account JSON or Application Default Credentials), but the
token is handed to the anthropic SDK's ``AnthropicVertex`` client rather than an
OpenAI client. Project/region resolution and the live Credentials object come
from ``agent/vertex_adapter.py`` (``resolve_vertex_anthropic_params``); the
client is constructed in ``agent/agent_init.py``.

``auth_type="vertex"`` marks this as an OAuth-token provider (resolved
specially, like bedrock's ``aws_sdk``) so a credentials-file path is never
mistaken for a static API key.
"""

from providers import register_provider
from providers.base import ProviderProfile


class VertexAnthropicProfile(ProviderProfile):
    """Claude on Vertex AI — Anthropic Messages API, no REST /models route."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Vertex has no ``/models`` listing route; the wizard ships a curated
        list. Model discovery is not available here."""
        return None


vertex_anthropic = VertexAnthropicProfile(
    name="vertex-anthropic",
    aliases=("vertex-claude", "anthropic-vertex", "claude-vertex"),
    api_mode="anthropic_messages",
    env_vars=(),  # OAuth2 via service account / ADC — not a static key env var
    base_url="https://aiplatform.googleapis.com",  # real base_url computed at runtime
    auth_type="vertex",
    supports_vision=True,
    display_name="Vertex AI (Claude)",
    description="Anthropic Claude models via Google Vertex AI (Anthropic Messages API, OAuth2)",
)

register_provider(vertex_anthropic)
