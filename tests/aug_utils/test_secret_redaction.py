# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secrets must never leak through the CLI override log or a validation error."""

from pydantic import ValidationError

from aug_utils.common import format_validation_error, redact_overrides
from aug_utils.schema import PipelineConfig


class TestRedactOverrides:
    def test_masks_api_key_value(self):
        assert redact_overrides(["endpoints.0.api_key=sk-secret"]) == [
            "endpoints.0.api_key=***"
        ]

    def test_masks_token_secret_password_credential(self):
        assert redact_overrides(
            ["a.token=t", "b.client_secret=s", "c.password=p", "d.credential=k"]
        ) == ["a.token=***", "b.client_secret=***", "c.password=***", "d.credential=***"]

    def test_keeps_api_key_env_name(self):
        # api_key_env names an env var (not the secret itself) -> keep it visible.
        assert redact_overrides(["endpoints.0.api_key_env=LLM_API_KEY"]) == [
            "endpoints.0.api_key_env=LLM_API_KEY"
        ]

    def test_passes_through_non_secret_overrides(self):
        assert redact_overrides(
            ["endpoints.0.url=http://x/v1", "pipeline.retry=3"]
        ) == ["endpoints.0.url=http://x/v1", "pipeline.retry=3"]

    def test_flag_without_value_untouched(self):
        assert redact_overrides(["just.a.flag"]) == ["just.a.flag"]


class TestFormatValidationError:
    def _endpoint_missing_url_error(self) -> ValidationError:
        try:
            PipelineConfig(
                data=[
                    {
                        "inputs": {"rgb": "/tmp/x.mp4"},
                        "output": {
                            "video": "/tmp/o.mp4",
                            "caption": "/tmp/c.txt",
                            "metadata": "/tmp/m.json",
                        },
                    }
                ],
                endpoints=[{"role": "image_edit"}],  # missing required 'url'
                augmentation={"model": {"name": "image-edit"}},
            )
        except ValidationError as exc:
            return exc
        raise AssertionError("expected ValidationError")

    def test_reports_location_and_message(self):
        formatted = format_validation_error(self._endpoint_missing_url_error())
        assert "endpoints.0.url" in formatted

    def test_does_not_echo_complex_input(self):
        exc = self._endpoint_missing_url_error()
        # Pydantic's default rendering dumps the offending input dict verbatim,
        # which could carry a secret; our formatter must not.
        assert "role" in str(exc)
        assert "role" not in format_validation_error(exc)
