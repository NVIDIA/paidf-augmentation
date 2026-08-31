# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generation provenance block written into output metadata."""

import json
from types import SimpleNamespace

import logging

from captioning.text_captioner import TextCaptioner
from modules.cli import (
    _captioning_provenance,
    _fold_repeated_request_fields,
    _pipeline_endpoints,
    _reset_request_records,
)

LOGGER = logging.getLogger("test-provenance")


def _generator(last_request=None, *, adapter_name=None, nested=False):
    endpoint = SimpleNamespace(
        id="cosmos_transfer",
        role="video_transfer",
        adapter=adapter_name,
        url="http://nim:8008",
    )
    adapter = SimpleNamespace(
        endpoint=endpoint,
        request_url="http://nim:8008/v1/infer",
        url="http://nim:8008",
        model=None,
        timeout=600.0,
        recorded_requests=[] if last_request is None else [last_request],
        recorded_call_count=0 if last_request is None else 1,
    )
    if nested:
        # Legacy shape: the CLI also supports generator.executor.adapter.
        return SimpleNamespace(executor=SimpleNamespace(adapter=adapter))
    return SimpleNamespace(adapter=adapter)


class TestGenerationRecord:
    def test_no_records_without_a_generator(self):
        assert _pipeline_endpoints(None, None, None) == []
        assert _pipeline_endpoints(None, SimpleNamespace(), None) == []

    def test_requests_omitted_when_nothing_was_recorded(self):
        """A generator whose calls all failed records nothing, but its endpoint
        identity is still reported."""
        record = _pipeline_endpoints(None, _generator(None), None)[0]
        assert record["stage"] == "generation"
        assert "requests" not in record
        assert "calls" not in record

    def test_records_request_and_duration(self):
        request = {"prompt": "snowy night", "seed": 42, "num_steps": 35}
        record = _pipeline_endpoints(None, _generator(request), None, None, 305.5)[0]

        assert record["requests"] == [request]
        assert record["calls"] == 1
        assert record["duration_seconds"] == 305.5
        assert record["id"] == "cosmos_transfer"
        assert record["role"] == "video_transfer"
        # The concrete route, not the base URL.
        assert record["url"] == "http://nim:8008/v1/infer"
        # Adapter unset on the endpoint -> resolved from the role default.
        assert record["adapter"] == "nim"

    def test_explicit_adapter_wins(self):
        records = _pipeline_endpoints(
            None, _generator({}, adapter_name="passthrough"), None
        )
        assert records[0]["adapter"] == "passthrough"

    def test_reads_through_nested_executor(self):
        records = _pipeline_endpoints(None, _generator({"seed": 1}, nested=True), None)
        assert records[0]["requests"] == [{"seed": 1}]

    def test_duration_omitted_when_unknown(self):
        record = _pipeline_endpoints(None, _generator({"seed": 1}), None)[0]
        assert "duration_seconds" not in record

    def test_duration_only_on_the_generation_stage(self):
        captioner = SimpleNamespace(adapter=_chat_adapter("vlm", "http://v/v1", "m"))
        records = _pipeline_endpoints(
            captioner, _generator({"seed": 1}), None, None, 12.5
        )
        by_stage = {r["stage"]: r for r in records}
        assert by_stage["generation"]["duration_seconds"] == 12.5
        assert "duration_seconds" not in by_stage["captioning.vlm"]

    def test_unresolvable_adapter_does_not_raise(self):
        generator = _generator({})
        generator.adapter.endpoint = SimpleNamespace(id="x", role=None, adapter=None)
        records = _pipeline_endpoints(None, generator, None)
        assert records[0]["adapter"] is None

    def test_records_are_json_serializable(self):
        records = _pipeline_endpoints(None, _generator({"seed": 42}), None)
        assert json.loads(json.dumps(records))[0]["requests"][0]["seed"] == 42


def _chat_adapter(role, url, model, last_request=None):
    """Duck-type of the OpenAIChatAdapter captioners/verifiers hold."""
    return SimpleNamespace(
        endpoint=SimpleNamespace(id=f"{role}-inline", role=role, adapter=None, url=url),
        url=url,
        request_url=url,
        model=model,
        role=role,
        timeout=7200.0,
        recorded_requests=list(last_request or []),
        recorded_call_count=len(last_request or []),
    )


class TestPipelineEndpoints:
    def test_records_captioning_generation_and_evaluation(self):
        vlm = _chat_adapter("vlm", "http://vlm:8000/v1", "Qwen/VL")
        llm = _chat_adapter("llm", "http://llm:8002/v1", "Qwen/Instruct")
        captioner = SimpleNamespace(
            vlm_captioner=SimpleNamespace(adapter=vlm),
            llm_captioner=SimpleNamespace(adapter=llm),
        )
        verifier = SimpleNamespace(
            question_generator=SimpleNamespace(adapter=llm),
            vlm_verifier=SimpleNamespace(adapter=vlm),
        )
        records = _pipeline_endpoints(captioner, _generator({}), verifier)

        assert [r["stage"] for r in records] == [
            "captioning.vlm",
            "captioning.llm",
            "generation",
            "attribute_verification.question_generation",
            "attribute_verification.vlm_verification",
        ]
        by_stage = {r["stage"]: r for r in records}
        assert by_stage["captioning.vlm"]["url"] == "http://vlm:8000/v1"
        assert by_stage["captioning.vlm"]["model"] == "Qwen/VL"
        # Role default resolves the chat contract for vlm/llm consumers.
        assert by_stage["captioning.llm"]["adapter"] == "openai.chat.completions"
        assert by_stage["generation"]["url"] == "http://nim:8008/v1/infer"

    def test_single_model_captioner_labelled_by_role(self):
        captioner = SimpleNamespace(adapter=_chat_adapter("vlm", "http://v/v1", "m"))
        records = _pipeline_endpoints(captioner, None, None)
        assert [r["stage"] for r in records] == ["captioning.vlm"]

    def test_text_captioner_contributes_nothing(self):
        # A text/file captioner calls no endpoint and has no adapter.
        records = _pipeline_endpoints(SimpleNamespace(), None, None)
        assert records == []

    def test_standalone_vlm_verification_recorded(self):
        verifier = SimpleNamespace(adapter=_chat_adapter("vlm", "http://v/v1", "m"))
        records = _pipeline_endpoints(None, None, None, verifier)
        assert [r["stage"] for r in records] == ["vlm_verification"]

    def test_standalone_verifier_ignored_when_attribute_verifier_present(self):
        """Avoids double-recording: the same object hangs off both."""
        vlm = _chat_adapter("vlm", "http://v/v1", "m")
        verifier = SimpleNamespace(
            question_generator=None, vlm_verifier=SimpleNamespace(adapter=vlm)
        )
        records = _pipeline_endpoints(
            None, None, verifier, SimpleNamespace(adapter=vlm)
        )
        assert [r["stage"] for r in records] == [
            "attribute_verification.vlm_verification"
        ]

    def test_adapterless_sub_captioner_falls_back_to_single_model(self):
        """A sub-captioner that exposes no adapter must not suppress the
        single-model fallback and leave the run with no captioning record."""
        captioner = SimpleNamespace(
            vlm_captioner=SimpleNamespace(),  # present, but no .adapter
            adapter=_chat_adapter("vlm", "http://v/v1", "m"),
        )
        records = _pipeline_endpoints(captioner, None, None)
        assert [r["stage"] for r in records] == ["captioning.vlm"]

    def test_adapter_without_role_does_not_raise(self):
        """A bare adapter.role access here would land in cli.py's metadata
        try/except and silently drop the sample."""
        adapter = _chat_adapter("vlm", "http://v/v1", "m")
        del adapter.role
        records = _pipeline_endpoints(SimpleNamespace(adapter=adapter), None, None)
        assert records[0]["stage"] == "captioning.model"

    def test_chat_stage_records_system_and_user_prompts(self):
        """Captioning/verification reach the endpoint through adapter.chat(),
        so their prompts are what that stage's request must show."""
        body = {
            "model": "Qwen/VL",
            "messages": [
                {"role": "system", "content": "You describe video scenes."},
                {"role": "user", "content": "Describe this intersection."},
            ],
            "temperature": 0.3,
        }
        captioner = SimpleNamespace(
            vlm_captioner=SimpleNamespace(
                adapter=_chat_adapter("vlm", "http://v/v1", "Qwen/VL", [body])
            )
        )
        record = _pipeline_endpoints(captioner, None, None)[0]
        assert record["stage"] == "captioning.vlm"
        roles = [m["role"] for m in record["requests"][0]["messages"]]
        assert roles == ["system", "user"]
        assert record["requests"][0]["temperature"] == 0.3

    def test_all_calls_of_a_multi_call_stage_are_recorded(self):
        """A verifier answers one question per variable; recording only the last
        would leave the other prompts unattributed."""
        calls = [
            {"model": "m", "messages": [{"role": "user", "content": f"q{i}"}]}
            for i in range(4)
        ]
        verifier = SimpleNamespace(
            question_generator=SimpleNamespace(
                adapter=_chat_adapter("llm", "http://l/v1", "m", calls)
            ),
            vlm_verifier=None,
        )
        record = _pipeline_endpoints(None, None, verifier)[0]
        assert record["calls"] == 4
        assert [r["messages"][0]["content"] for r in record["requests"]] == [
            "q0",
            "q1",
            "q2",
            "q3",
        ]


class TestCaptioningProvenance:
    def test_text_captioner_records_strategy_and_template(self):
        """A text captioner calls no endpoint, so without this the template that
        produced the prompt is nowhere in metadata."""
        captioner = TextCaptioner(
            text="Change the clothing to a {color} {garment}.",
            logger=LOGGER,
            variables={"color": ["blue"], "garment": ["sweater"]},
        )
        record = _captioning_provenance(captioner)
        assert record["strategy"] == "text"
        assert record["template"] == "Change the clothing to a {color} {garment}."

    def test_chat_captioner_reports_strategy_without_a_template(self):
        class VLMLLMCaptioner:
            pass

        record = _captioning_provenance(VLMLLMCaptioner())
        assert record == {"strategy": "vlm+llm"}

    def test_no_captioner_records_nothing(self):
        assert _captioning_provenance(None) == {}

    def test_unknown_captioner_falls_back_to_its_class_name(self):
        class CustomCaptioner:
            pass

        assert _captioning_provenance(CustomCaptioner())["strategy"] == (
            "CustomCaptioner"
        )


class TestResetRequestRecords:
    def test_resets_every_wired_adapter(self):
        vlm = _chat_adapter("vlm", "http://v/v1", "m", [{"a": 1}])
        llm = _chat_adapter("llm", "http://l/v1", "m", [{"b": 2}])
        vlm.reset_requests = lambda: vlm.recorded_requests.clear()
        llm.reset_requests = lambda: llm.recorded_requests.clear()
        captioner = SimpleNamespace(
            vlm_captioner=SimpleNamespace(adapter=vlm),
            llm_captioner=SimpleNamespace(adapter=llm),
        )
        _reset_request_records(captioner, None, None)
        assert vlm.recorded_requests == [] and llm.recorded_requests == []

    def test_adapter_without_reset_is_skipped(self):
        captioner = SimpleNamespace(adapter=_chat_adapter("vlm", "http://v/v1", "m"))
        _reset_request_records(captioner, None, None)  # must not raise


class TestFoldRepeatedRequestFields:
    def test_single_call_is_left_alone(self):
        shared, per_call = _fold_repeated_request_fields([{"seed": 1}])
        assert shared is None and per_call == [{"seed": 1}]

    def test_identical_params_are_hoisted_once(self):
        requests = [
            {"model": "m", "temperature": 0.2, "messages": [{"role": "user", "u": i}]}
            for i in range(3)
        ]
        shared, per_call = _fold_repeated_request_fields(requests)
        assert shared == {"model": "m", "temperature": 0.2}
        assert per_call == [{"messages": [{"role": "user", "u": i}]} for i in range(3)]

    def test_shared_system_turn_is_hoisted_out_of_messages(self):
        system = {"role": "system", "content": "Answer with one letter."}
        requests = [
            {"messages": [system, {"role": "user", "content": f"q{i}"}]}
            for i in range(3)
        ]
        shared, per_call = _fold_repeated_request_fields(requests)
        assert shared == {"messages": [system]}
        # Each call keeps only its own user turn.
        assert per_call == [
            {"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(3)
        ]

    def test_differing_system_turn_is_not_hoisted(self):
        """A stage that switches system prompt mid-sample (MCQ answers then a
        natural caption) has no shared prefix, and must not claim one."""
        requests = [
            {"messages": [{"role": "system", "content": s}, {"role": "user", "c": 1}]}
            for s in ("answer briefly", "describe the person")
        ]
        shared, per_call = _fold_repeated_request_fields(requests)
        assert shared is None
        assert all(len(r["messages"]) == 2 for r in per_call)

    def test_a_key_missing_from_one_call_is_not_shared(self):
        requests = [{"a": 1, "b": 2}, {"a": 1}]
        shared, per_call = _fold_repeated_request_fields(requests)
        assert shared == {"a": 1}
        assert per_call == [{"b": 2}, {}]

    def test_folding_is_lossless(self):
        requests = [
            {"model": "m", "temperature": 0.2, "messages": [{"role": "u", "c": i}]}
            for i in range(3)
        ]
        shared, per_call = _fold_repeated_request_fields(requests)
        for original, folded in zip(requests, per_call):
            rebuilt = {**shared, **folded}
            rebuilt["messages"] = shared.get("messages", []) + folded["messages"]
            assert rebuilt == original

    def test_model_is_not_repeated_when_it_matches_the_endpoint(self):
        calls = [
            {"model": "m", "messages": [{"role": "user", "content": f"q{i}"}]}
            for i in range(2)
        ]
        verifier = SimpleNamespace(
            question_generator=SimpleNamespace(
                adapter=_chat_adapter("llm", "http://l/v1", "m", calls)
            ),
            vlm_verifier=None,
        )
        record = _pipeline_endpoints(None, None, verifier)[0]
        assert record["model"] == "m"
        assert "model" not in record.get("request_shared", {})

    def test_modelless_adapter_with_several_calls_does_not_crash(self):
        """A single-model NIM has model=None and sends no `model` field, so a
        `None == None` match must not pop a key that was never there."""
        adapter = SimpleNamespace(
            endpoint=SimpleNamespace(
                id="cosmos_transfer",
                role="video_transfer",
                adapter="nim",
                url="http://n",
            ),
            request_url="http://n/v1/infer",
            url="http://n",
            model=None,
            timeout=600.0,
            recorded_requests=[
                {"prompt": "p", "seed": 42, "guidance": 7},
                {"prompt": "p", "seed": 43, "guidance": 7},
            ],
            recorded_call_count=2,
        )
        record = _pipeline_endpoints(None, SimpleNamespace(adapter=adapter), None)[0]
        assert record["request_shared"] == {"prompt": "p", "guidance": 7}
        assert [r["seed"] for r in record["requests"]] == [42, 43]

    def test_model_is_dropped_from_single_call_stages_too(self):
        """One-call and many-call stages of the same config must report the same
        fields, or a consumer reading requests[0].model breaks on one of them."""
        one = _chat_adapter("llm", "http://l/v1", "m", [{"model": "m", "t": 1}])
        many = _chat_adapter(
            "llm", "http://l/v1", "m", [{"model": "m", "t": 1}, {"model": "m", "t": 2}]
        )
        for adapter in (one, many):
            verifier = SimpleNamespace(
                question_generator=SimpleNamespace(adapter=adapter), vlm_verifier=None
            )
            record = _pipeline_endpoints(None, None, verifier)[0]
            assert all("model" not in r for r in record["requests"])
            assert "model" not in record.get("request_shared", {})
