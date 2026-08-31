# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VLM verifier / LLM question generator via ``OpenAIChatAdapter``.

The shared adapter's ``chat`` method is patched to return offline canned
responses (``SimpleNamespace`` stand-ins for OpenAI SDK objects); no network or
live endpoint is involved.
"""

import json
import logging
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from verification.core import AttributeVerifier
from verification.llm_question_generator import LLMQuestionGenerator
from verification.vlm_verifier import VLMVerifier, _extract_mcq_letter

LOGGER = logging.getLogger("test")

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _nonstream_response(content):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(id="resp", choices=[SimpleNamespace(message=message)])


@pytest.fixture
def png_path():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frame.png")
        with open(path, "wb") as f:
            f.write(PNG_BYTES)
        yield path


# ---------------------------------------------------------------------------
# VLMVerifier.verify_question
# ---------------------------------------------------------------------------
class TestVLMVerifier:
    def _make(self):
        return VLMVerifier(
            system_prompt="You answer MCQs.",
            retry=0,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            max_tokens=64,
            stream=False,
            endpoint="http://localhost:9/v1",
            model="some/vlm",
            logger=LOGGER,
        )

    def test_mcq_request_shape_and_correct_answer(self, png_path):
        verifier = self._make()
        options = {"A": "red", "B": "blue", "C": "green", "D": "yellow"}
        with patch.object(
            verifier.adapter, "chat", return_value=_nonstream_response("B")
        ) as chat:
            is_correct, answer, reasoning = verifier.verify_question(
                png_path, "What color is the car?", options, correct_answer="B"
            )

        assert is_correct is True
        assert answer == "B"
        assert reasoning is None
        _, kwargs = chat.call_args
        assert kwargs["model"] == "some/vlm"
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 64
        assert kwargs["stream"] is False
        # The user message carries the formatted MCQ and the image.
        user_content = kwargs["messages"][1]["content"]
        question_text = user_content[0]["text"]
        assert "What color is the car?" in question_text
        assert "A) red" in question_text and "B) blue" in question_text
        assert "single letter" in question_text
        assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_wrong_answer_marked_incorrect(self, png_path):
        verifier = self._make()
        options = {"A": "red", "B": "blue"}
        with patch.object(
            verifier.adapter, "chat", return_value=_nonstream_response("A")
        ):
            is_correct, answer, _ = verifier.verify_question(
                png_path, "Color?", options, correct_answer="B"
            )
        assert is_correct is False
        assert answer == "A"

    def test_reasoning_model_cot_enumeration_not_misparsed(self, png_path):
        """A thinking VLM that enumerates options ("A) ... B) ...") in its CoT
        before the verdict must NOT be pinned to A by a first-letter scan."""
        verifier = self._make()
        options = {"A": "beige", "B": "black", "C": "pink", "D": "yellow"}
        cot = (
            "The user wants the color of the bottoms.\n"
            "4. Evaluate the options:\n"
            "   A) beige: No, the pants are dark.\n"
            "   B) black: Yes, the pants are black.\n"
            "The correct option is B.\n</think>\n\nB"
        )
        with patch.object(
            verifier.adapter, "chat", return_value=_nonstream_response(cot)
        ):
            is_correct, answer, _ = verifier.verify_question(
                png_path, "Bottom color?", options, correct_answer="B"
            )
        assert answer == "B"
        assert is_correct is True

    def test_reasoning_model_think_block_stripped(self, png_path):
        """``<think>`` enumerations are dropped; verdict after the tag wins."""
        verifier = self._make()
        options = {"A": "dress", "B": "jeans", "C": "skirt", "D": "shorts"}
        cot = "<think>Option A) dress? No. It is denim, so B.</think>\nB"
        with patch.object(
            verifier.adapter, "chat", return_value=_nonstream_response(cot)
        ):
            is_correct, answer, _ = verifier.verify_question(
                png_path, "Bottom type?", options, correct_answer="B"
            )
        assert answer == "B"
        assert is_correct is True

    def test_multiframe_video_sends_all_sampled_frames(self, tmp_path):
        # frames=3 -> the video path samples 3 frames and sends all of them.
        verifier = VLMVerifier(
            system_prompt="You answer MCQs.",
            retry=0,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            max_tokens=10,
            stream=False,
            endpoint="http://localhost:9/v1",
            model="some/vlm",
            logger=LOGGER,
            frames=3,
        )
        assert verifier.frames == 3
        frame_paths = []
        for i in range(3):
            p = tmp_path / f"frame_{i}.jpg"
            p.write_bytes(PNG_BYTES)
            frame_paths.append(str(p))
        video = tmp_path / "out.mp4"
        video.write_bytes(b"not-a-real-mp4")  # content irrelevant; extractor is mocked
        with (
            patch.object(
                verifier, "_extract_frames", return_value=frame_paths
            ) as extract,
            patch.object(
                verifier.adapter, "chat", return_value=_nonstream_response("A")
            ) as chat,
        ):
            is_correct, answer, _ = verifier.verify_question(
                str(video),
                "Does a person fall?",
                {"A": "yes", "B": "no"},
                correct_answer="A",
            )
        assert is_correct is True and answer == "A"
        # The extractor was asked for self.frames frames.
        assert extract.call_args[0][1] == 3
        content = chat.call_args[1]["messages"][1]["content"]
        images = [c for c in content if c.get("type") == "image_url"]
        assert len(images) == 3
        assert all(
            im["image_url"]["url"].startswith("data:image/jpeg;base64,")
            for im in images
        )
        # Multi-frame preamble tells the VLM the frames are an ordered sequence.
        assert "frames sampled" in content[0]["text"]

    def test_from_config_reads_frames(self, monkeypatch):
        monkeypatch.setenv("VLM_API_KEY", "k")
        with patch("generation.adapters.openai_chat.OpenAI"):
            verifier = VLMVerifier.from_config(
                config_params={"system_prompt": "", "parameters": {}, "frames": 6},
                endpoint="http://localhost:9/v1",
                model="some/vlm",
                logger=LOGGER,
            )
        assert verifier.frames == 6


# ---------------------------------------------------------------------------
# LLMQuestionGenerator
# ---------------------------------------------------------------------------
class TestLLMQuestionGenerator:
    def _make(self):
        return LLMQuestionGenerator(
            system_prompt="Generate MCQs.",
            retry=0,
            temperature=0.0,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=512,
            stream=False,
            endpoint="http://localhost:9/v1",
            model="some/llm",
            logger=LOGGER,
        )

    def test_guided_json_parsed_into_question(self):
        gen = self._make()
        canned = json.dumps(
            {
                "variable": "weather_condition",
                "value": "cloudy",
                "question": "What are the weather conditions?",
                "options": {"A": "clear", "B": "cloudy", "C": "rainy", "D": "snowy"},
                "correct_answer": "B",
            }
        )
        with patch.object(
            gen.adapter, "chat", return_value=_nonstream_response(canned)
        ) as chat:
            question = gen.generate_question(
                "weather_condition", "cloudy", ["clear", "cloudy", "rainy", "snowy"]
            )

        # The request uses guided-JSON (response_format) and non-streaming.
        _, kwargs = chat.call_args
        assert kwargs["model"] == "some/llm"
        assert kwargs["stream"] is False
        assert "json_schema" in kwargs["response_format"]
        user_prompt = kwargs["messages"][1]["content"]
        assert "weather_condition" in user_prompt
        assert "cloudy" in user_prompt

        # Parsed/repaired question: selected value maps to the correct letter.
        assert question["variable"] == "weather_condition"
        assert question["value"] == "cloudy"
        assert question["question"] == "What are the weather conditions?"
        correct_letter = question["correct_answer"]
        assert question["options"][correct_letter] == "cloudy"

    def test_generate_questions_aggregates(self):
        gen = self._make()
        canned = json.dumps(
            {
                "variable": "vehicle_type",
                "value": "truck",
                "question": "What type of vehicle is shown?",
                "options": {"A": "truck", "B": "sedan", "C": "bus"},
                "correct_answer": "A",
            }
        )
        with patch.object(
            gen.adapter, "chat", return_value=_nonstream_response(canned)
        ):
            questions = gen.generate_questions(
                {"vehicle_type": "truck"},
                {"vehicle_type": ["truck", "sedan", "bus", "van"]},
            )
        assert len(questions) == 1
        q = questions[0]
        assert q["value"] == "truck"
        assert q["options"][q["correct_answer"]] == "truck"

    def test_one_failed_variable_fails_the_whole_set(self):
        """A dropped variable would verify the sample against fewer attributes
        than configured, and the run could then report a pass nothing was
        actually checked against."""
        gen = self._make()
        canned = json.dumps(
            {
                "variable": "vehicle_type",
                "value": "truck",
                "question": "What type of vehicle is shown?",
                "options": {"A": "truck", "B": "sedan", "C": "bus"},
                "correct_answer": "A",
            }
        )

        def flaky(**kwargs):
            # Second variable's call fails, as a partially-down LLM endpoint does.
            if "weather_condition" in kwargs["messages"][1]["content"]:
                raise RuntimeError("503 Service Unavailable")
            return _nonstream_response(canned)

        with (
            patch.object(gen.adapter, "chat", side_effect=flaky),
            pytest.raises(RuntimeError) as excinfo,
        ):
            gen.generate_questions(
                {"vehicle_type": "truck", "weather_condition": "cloudy"},
                {
                    "vehicle_type": ["truck", "sedan", "bus"],
                    "weather_condition": ["cloudy", "clear"],
                },
            )

        # The message must name the variable and carry the endpoint's own error,
        # or the operator cannot tell an outage from a bad config.
        message = str(excinfo.value)
        assert "weather_condition" in message
        assert "503" in message

    def test_generate_options_keeps_llm_distractors(self):
        # generate_options=True -> the LLM's invented options survive (the curated
        # pool is ignored), but the correct answer stays pinned to the selected value.
        gen = LLMQuestionGenerator(
            system_prompt="Generate MCQs.",
            retry=0,
            temperature=0.0,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=512,
            stream=False,
            endpoint="http://localhost:9/v1",
            model="some/llm",
            logger=LOGGER,
            generate_options=True,
        )
        canned = json.dumps(
            {
                "variable": "anomaly_type",
                "value": "person_falling",
                "question": "What anomaly is shown in the video?",
                "options": {
                    "A": "person_falling",
                    "B": "forklift_collision",
                    "C": "fire_smoke",
                    "D": "flooding",
                },
                "correct_answer": "A",
            }
        )
        with patch.object(
            gen.adapter, "chat", return_value=_nonstream_response(canned)
        ) as chat:
            # The pool here is the degenerate single-value list; it must be ignored.
            q = gen.generate_question(
                "anomaly_type", "person_falling", ["person_falling"]
            )
        values = set(q["options"].values())
        # LLM-invented distractors survive (not collapsed to the 1-item pool).
        assert "forklift_collision" in values
        assert len([v for v in values if v != "person_falling"]) >= 2
        # Correct answer stays pinned to the ground-truth selected value.
        assert q["options"][q["correct_answer"]] == "person_falling"
        # The open-mode prompt asked the LLM to invent options.
        user_prompt = chat.call_args[1]["messages"][1]["content"]
        assert "invent" in user_prompt.lower()


# ---------------------------------------------------------------------------
# API key resolution via from_config: api_key_env / literal api_key reach the
# adapter (precedence: literal api_key -> $api_key_env -> role default).
# ---------------------------------------------------------------------------
class TestVerifierApiKeyResolution:
    def test_vlm_verifier_uses_api_key_env(self, monkeypatch):
        monkeypatch.setenv("VLM_API_KEY", "role-default-vlm")
        monkeypatch.setenv("NVCF_API_KEY", "from-nvcf-env")
        with patch("generation.adapters.openai_chat.OpenAI"):
            verifier = VLMVerifier.from_config(
                config_params={"system_prompt": "", "parameters": {}},
                endpoint="http://localhost:9/v1",
                model="some/vlm",
                logger=LOGGER,
                api_key_env="NVCF_API_KEY",
            )
        assert verifier.adapter.api_key == "from-nvcf-env"
        assert verifier.adapter.api_key != "role-default-vlm"

    def test_vlm_verifier_defaults_to_role_env(self, monkeypatch):
        monkeypatch.setenv("VLM_API_KEY", "role-default-vlm")
        with patch("generation.adapters.openai_chat.OpenAI"):
            verifier = VLMVerifier.from_config(
                config_params={"system_prompt": "", "parameters": {}},
                endpoint="http://localhost:9/v1",
                model="some/vlm",
                logger=LOGGER,
            )
        assert verifier.adapter.api_key == "role-default-vlm"

    def test_question_generator_uses_api_key_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "role-default-llm")
        monkeypatch.setenv("NVCF_API_KEY", "from-nvcf-env")
        with patch("generation.adapters.openai_chat.OpenAI"):
            gen = LLMQuestionGenerator.from_config(
                config_params={"parameters": {}},
                system_prompt="Generate MCQs.",
                endpoint="http://localhost:9/v1",
                model="some/llm",
                logger=LOGGER,
                api_key_env="NVCF_API_KEY",
            )
        assert gen.adapter.api_key == "from-nvcf-env"
        assert gen.adapter.api_key != "role-default-llm"


class TestExtractMcqLetter:
    """Think-aware MCQ answer extraction (no raw fallback into <think> blocks)."""

    def test_plain_answer(self):
        assert _extract_mcq_letter("The answer is C") == "C"

    def test_last_standalone_letter_wins(self):
        assert _extract_mcq_letter("Options A) red B) blue. Final answer: B") == "B"

    def test_answer_after_terminated_think(self):
        assert _extract_mcq_letter("<think>maybe A or B</think> Answer: D") == "D"

    def test_reasoning_only_is_unknown_not_last_letter(self):
        # Every A-D lives inside the think block; the model never committed to an
        # answer. Must be UNKNOWN, not "D" (which would be a false pass).
        content = "<think>A is wrong, B maybe, C no, D uncertain</think>"
        assert _extract_mcq_letter(content) == "UNKNOWN"

    def test_unterminated_think_is_unknown(self):
        assert _extract_mcq_letter("<think>A wrong, B maybe, C no, D") == "UNKNOWN"

    def test_empty_is_unknown(self):
        assert _extract_mcq_letter("") == "UNKNOWN"


class TestEmptyQuestionSet:
    """An enabled attribute verifier with zero questions must not pass silently."""

    def test_no_variables_and_no_extra_questions_fails(self):
        verifier = AttributeVerifier(
            question_generator=SimpleNamespace(),  # not called: no variables
            vlm_verifier=SimpleNamespace(),  # not called: no questions
            logger=LOGGER,
        )
        passed, details = verifier.verify_video_attributes(
            video_path="/tmp/out.mp4",
            selected_variables={},
            variable_options={},
            extra_questions=[],
        )
        assert passed is False
        assert details["summary"]["total_checks"] == 0
        assert details["summary"].get("reason") == "no_questions"
        # Same payload shape as the other exit paths (consumers rely on it).
        assert "duration_seconds" in details


class TestVerificationDetailsShape:
    """`results` is the single record of what was asked and how it was answered."""

    def _verifier(self, question, reasoning=None):
        question_generator = SimpleNamespace(
            generate_questions=lambda variables, variable_options: [question]
        )
        vlm_verifier = SimpleNamespace(
            verify_question=lambda *args, **kwargs: (True, "A", reasoning)
        )
        return AttributeVerifier(
            question_generator=question_generator,
            vlm_verifier=vlm_verifier,
            logger=LOGGER,
        )

    def _run(self, verifier):
        return verifier.verify_video_attributes(
            video_path="/tmp/out.mp4",
            selected_variables={"bottom_color": "black"},
            variable_options={"bottom_color": ["black", "blue"]},
        )

    QUESTION = {
        "variable": "bottom_color",
        "value": "black",
        "question": "What is the color of the bottom?",
        "options": {"A": "black", "B": "blue"},
        "correct_answer": "A",
    }

    def test_questions_are_not_duplicated_alongside_results(self):
        """Every field of a question reappears in its result, so listing the
        questions separately recorded each variable twice."""
        _passed, details = self._run(self._verifier(self.QUESTION))
        assert "questions" not in details
        assert [r["variable"] for r in details["results"]] == ["bottom_color"]

    def test_result_carries_every_field_the_question_had(self):
        _passed, details = self._run(self._verifier(self.QUESTION))
        result = details["results"][0]
        for key in ("variable", "value", "question", "options"):
            assert result[key] == self.QUESTION[key]
        # correct_answer is reported under its verification-side name.
        assert result["expected_answer"] == self.QUESTION["correct_answer"]
        assert result["vlm_answer"] == "A"
        assert result["passed"] is True

    def test_request_reasoning_survives_on_the_result(self):
        """The one question-only field; dropping the questions list must not
        lose it."""
        question = {**self.QUESTION, "request_reasoning": True}
        _passed, details = self._run(self._verifier(question, reasoning="because"))
        assert details["results"][0]["request_reasoning"] is True
        assert details["results"][0]["vlm_reasoning"] == "because"

    def test_request_reasoning_recorded_on_the_error_path_too(self):
        def boom(*args, **kwargs):
            raise RuntimeError("vlm down")

        question_generator = SimpleNamespace(
            generate_questions=lambda variables, variable_options: [
                {**self.QUESTION, "request_reasoning": True}
            ]
        )
        verifier = AttributeVerifier(
            question_generator=question_generator,
            vlm_verifier=SimpleNamespace(verify_question=boom),
            logger=LOGGER,
        )
        passed, details = self._run(verifier)
        result = details["results"][0]
        assert passed is False
        assert result["vlm_answer"] == "ERROR"
        assert result["request_reasoning"] is True
        # An error still produces a result, so results can never be shorter
        # than the question set it came from.
        assert len(details["results"]) == details["summary"]["total_checks"]


class TestErroredChecksAreNotQualityFailures:
    """A VLM outage and a video the VLM judged wrong both fail the sample, but
    only one of them says anything about the output."""

    QUESTION = {
        "variable": "bottom_color",
        "value": "black",
        "question": "What is the color of the bottom?",
        "options": {"A": "black", "B": "blue"},
        "correct_answer": "A",
    }

    def _run(self, verify_question):
        verifier = AttributeVerifier(
            question_generator=SimpleNamespace(
                generate_questions=lambda variables, variable_options: [self.QUESTION]
            ),
            vlm_verifier=SimpleNamespace(verify_question=verify_question),
            logger=LOGGER,
        )
        return verifier.verify_video_attributes(
            video_path="/tmp/out.mp4",
            selected_variables={"bottom_color": "black"},
            variable_options={"bottom_color": ["black", "blue"]},
        )

    def test_endpoint_error_counts_as_errored_not_failed(self):
        def boom(*args, **kwargs):
            raise RuntimeError("vlm down")

        passed, details = self._run(boom)
        summary = details["summary"]
        assert passed is False
        assert summary["errored_checks"] == 1
        # Counting this as a quality failure is what made an outage look like a
        # batch of bad generations.
        assert summary["failed_checks"] == 0
        assert summary["passed_checks"] == 0

    def test_wrong_answer_still_counts_as_failed(self):
        passed, details = self._run(lambda *args, **kwargs: (False, "B", None))
        summary = details["summary"]
        assert passed is False
        assert summary["failed_checks"] == 1
        assert summary["errored_checks"] == 0

    def test_summary_is_present_even_when_nothing_errors(self):
        # Consumers can read the field unconditionally.
        _passed, details = self._run(lambda *args, **kwargs: (True, "A", None))
        assert details["summary"]["errored_checks"] == 0
