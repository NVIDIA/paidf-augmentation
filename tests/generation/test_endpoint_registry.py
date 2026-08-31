# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the endpoint-registry config-resolution helpers."""

from types import SimpleNamespace

import pytest

from aug_utils.endpoint_registry import (
    default_adapter_for,
    effective_adapter,
    find_endpoint_by_role,
    resolve_endpoint,
    select_endpoint,
)


def _ep(id, role, adapter=None):
    return SimpleNamespace(id=id, role=role, url="http://x", model="m", adapter=adapter)


class TestResolveEndpoint:
    def test_by_id(self):
        eps = [_ep("a", "vlm"), _ep("b", "image_edit")]
        assert resolve_endpoint(eps, "b").id == "b"

    def test_by_role(self):
        eps = [_ep("a", "vlm"), _ep("b", "image_edit")]
        assert resolve_endpoint(eps, "image_edit").id == "b"

    def test_by_model_name_mapping(self):
        eps = [_ep("a", "vlm"), _ep("b", "image_edit")]
        # selector "image-edit" -> role "image_edit"
        assert resolve_endpoint(eps, "image-edit").id == "b"

    def test_id_wins_over_role(self):
        # An endpoint whose id equals another endpoint's role-selector still
        # resolves by id first.
        eps = [_ep("image_edit", "vlm"), _ep("b", "image_edit")]
        assert resolve_endpoint(eps, "image_edit").id == "image_edit"

    def test_zero_matches_raises_listing_ids(self):
        eps = [_ep("a", "vlm")]
        with pytest.raises(ValueError, match="available ids.*'a'"):
            resolve_endpoint(eps, "nope")

    def test_ambiguous_role_raises(self):
        eps = [_ep("a", "video_transfer"), _ep("b", "video_transfer")]
        with pytest.raises(ValueError, match="disambiguate by id"):
            resolve_endpoint(eps, "cosmos-transfer2.5")

    def test_dict_endpoints_supported(self):
        eps = [{"id": "a", "role": "vlm", "adapter": None}]
        assert resolve_endpoint(eps, "vlm")["id"] == "a"


class TestFindEndpointByRole:
    def test_single(self):
        eps = [_ep("a", "vlm"), _ep("b", "llm")]
        assert find_endpoint_by_role(eps, "llm").id == "b"

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="no endpoint"):
            find_endpoint_by_role([_ep("a", "vlm")], "llm")

    def test_zero_not_required_returns_none(self):
        assert find_endpoint_by_role([_ep("a", "vlm")], "llm", required=False) is None

    def test_multiple_raises_even_when_not_required(self):
        eps = [_ep("a", "vlm"), _ep("b", "vlm")]
        with pytest.raises(ValueError, match="'a'"):
            find_endpoint_by_role(eps, "vlm", required=False)

    def test_multiple_raises_with_ids(self):
        eps = [_ep("a", "vlm"), _ep("b", "vlm")]
        with pytest.raises(ValueError, match="'a'"):
            find_endpoint_by_role(eps, "vlm")


class TestSelectEndpoint:
    def test_picks_by_id(self):
        eps = [_ep("a", "llm"), _ep("b", "llm")]
        assert select_endpoint(eps, "llm", "b").id == "b"

    def test_unknown_id_raises_not_found(self):
        eps = [_ep("a", "llm")]
        with pytest.raises(ValueError, match="not found"):
            select_endpoint(eps, "llm", "nope")

    def test_role_mismatch_raises(self):
        eps = [_ep("a", "vlm"), _ep("b", "llm")]
        with pytest.raises(ValueError, match="has role 'vlm'"):
            select_endpoint(eps, "llm", "a")

    def test_falls_back_to_role_when_id_none(self):
        eps = [_ep("a", "vlm"), _ep("b", "llm")]
        assert select_endpoint(eps, "llm").id == "b"

    def test_fallback_required_false_returns_none(self):
        assert select_endpoint([_ep("a", "vlm")], "llm", required=False) is None


class TestDefaultAdapterFor:
    def test_known_roles(self):
        assert default_adapter_for("vlm") == "openai.chat.completions"
        assert default_adapter_for("llm") == "openai.chat.completions"
        assert default_adapter_for("image_edit") == "nim"
        assert default_adapter_for("video_transfer") == "nim"
        assert default_adapter_for("video_predict") == "nim"

    def test_unknown_role(self):
        assert default_adapter_for("mystery") is None


class TestEffectiveAdapter:
    def test_explicit_adapter_wins(self):
        ep = _ep("a", "image_edit", adapter="passthrough")
        assert effective_adapter(ep) == "passthrough"

    def test_role_default(self):
        ep = _ep("a", "image_edit", adapter=None)
        assert effective_adapter(ep) == "nim"

    def test_vlm_role_default(self):
        ep = _ep("a", "vlm", adapter=None)
        assert effective_adapter(ep) == "openai.chat.completions"

    def test_undeterminable_raises(self):
        ep = _ep("a", "mystery", adapter=None)
        with pytest.raises(ValueError, match="cannot determine adapter"):
            effective_adapter(ep)

    def test_dict_endpoint(self):
        assert effective_adapter({"id": "a", "role": "llm", "adapter": None}) == (
            "openai.chat.completions"
        )
