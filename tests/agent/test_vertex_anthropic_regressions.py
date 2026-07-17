"""Regressions for Claude on Vertex across main and auxiliary runtimes."""

from unittest.mock import MagicMock, patch


def test_auxiliary_vertex_anthropic_uses_native_sdk():
    credentials = object()
    native_client = MagicMock()

    with (
        patch(
            "agent.vertex_adapter.resolve_vertex_anthropic_params",
            return_value=(credentials, "test-project", "global"),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_vertex_client",
            return_value=native_client,
        ) as build,
    ):
        from agent.auxiliary_client import (
            AnthropicAuxiliaryClient,
            resolve_provider_client,
        )

        client, model = resolve_provider_client(
            "vertex-anthropic", "claude-opus-4-6",
        )

    build.assert_called_once_with(
        "test-project", "global", credentials=credentials,
    )
    assert isinstance(client, AnthropicAuxiliaryClient)
    assert client._real_client is native_client
    assert client.base_url.startswith("https://aiplatform.googleapis.com/v1/")
    assert model == "claude-opus-4-6"


def test_runtime_vertex_anthropic_global_endpoint_has_no_global_prefix():
    credentials = object()

    with patch(
        "agent.vertex_adapter.resolve_vertex_anthropic_params",
        return_value=(credentials, "test-project", "global"),
    ):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested="vertex-anthropic",
            target_model="claude-opus-4-6",
        )

    assert runtime["base_url"] == (
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/"
        "global/publishers/anthropic/models"
    )
