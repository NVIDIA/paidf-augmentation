# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from typing import Any, Dict, List, Optional

import openai

from generation.adapters.openai_chat import OpenAIChatAdapter
from aug_utils.common import validate_and_cast_config_params


def _format_variable_label(variable_name: str) -> str:
    """Convert a config variable name into a readable label for prompting."""
    return variable_name.replace("_", " ").strip()


def _canonicalize(text: str) -> str:
    """Normalize strings for robust value matching."""
    return str(text).strip().lower()


_CONFUSING_COLOR_DISTRACTORS = {
    "beige": {"brown", "white", "yellow", "orange"},
    "black": {"brown", "grey", "blue"},
    "blue": {"black", "green", "purple"},
    "brown": {"beige", "green", "orange", "yellow", "camouflage"},
    "camouflage": {"brown", "green", "grey"},
    "green": {"blue", "brown", "camouflage", "yellow"},
    "grey": {"black", "white", "blue"},
    "orange": {"beige", "brown", "red", "yellow"},
    "pink": {"beige", "purple", "red"},
    "purple": {"blue", "pink", "red"},
    "red": {"brown", "orange", "pink", "purple"},
    "white": {"beige", "grey", "yellow"},
    "yellow": {"beige", "brown", "green", "orange", "white"},
}

_CONFUSING_VALUE_PAIRS = {
    "bottom_type": {
        "dress": {"skirt"},
        "skirt": {"dress"},
    }
}


def _is_confusing_option(
    variable_name: str, selected_value: str, selected_allowed: str, candidate: str
) -> bool:
    """Return whether a distractor is too close to the selected answer."""
    candidate_norm = _canonicalize(candidate)
    selected_norm = _canonicalize(selected_allowed)
    original_norm = _canonicalize(selected_value)

    if candidate_norm == selected_norm:
        return False

    # Avoid options that are substrings of fine-grained values, e.g. do not
    # offer both "olive" and "brown" as separate options for "olive brown".
    if candidate_norm and original_norm:
        if candidate_norm in original_norm or original_norm in candidate_norm:
            return True

    if variable_name.endswith("_color") or variable_name.endswith(" color"):
        confusing = _CONFUSING_COLOR_DISTRACTORS.get(selected_norm, set())
        return candidate_norm in confusing

    confusing_for_variable = _CONFUSING_VALUE_PAIRS.get(variable_name, {})
    return candidate_norm in confusing_for_variable.get(selected_norm, set())


class LLMQuestionGenerator:
    """Generate multiple choice verification questions using LLM."""

    # Define expected parameter types for validation
    _PARAM_TYPES = {
        "system_prompt": str,
        "retry": int,
        "temperature": float,
        "top_p": float,
        "frequency_penalty": float,
        "presence_penalty": float,
        "max_tokens": int,
        "stream": bool,
        "endpoint": str,
        "model": str,
    }

    # JSON schema for structured output (guided JSON)
    _QUESTION_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "verification_question",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Name of the variable being verified",
                    },
                    "value": {
                        "type": "string",
                        "description": "The selected value to verify",
                    },
                    "question": {
                        "type": "string",
                        "description": "The multiple choice question text",
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                        },
                        "required": ["A", "B"],
                        "additionalProperties": False,
                    },
                    "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                },
                "required": [
                    "variable",
                    "value",
                    "question",
                    "options",
                    "correct_answer",
                ],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        system_prompt: str,
        retry: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        max_tokens: int,
        stream: bool,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        api_key_env: Optional[str] = None,
        generate_options: bool = False,
    ):
        # Validate and cast parameters to ensure correct types
        params = {
            "system_prompt": system_prompt,
            "retry": retry,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "stream": stream,
            "endpoint": endpoint,
            "model": model,
        }

        validated_params = validate_and_cast_config_params(
            params, self._PARAM_TYPES, logger
        )

        self.system_prompt = validated_params["system_prompt"]
        self.retry = validated_params["retry"]
        self.temperature = validated_params["temperature"]
        self.top_p = validated_params["top_p"]
        self.frequency_penalty = validated_params["frequency_penalty"]
        self.presence_penalty = validated_params["presence_penalty"]
        self.max_tokens = validated_params["max_tokens"]
        self.stream = validated_params["stream"]
        self.endpoint = validated_params["endpoint"]
        self.model = validated_params["model"]

        self.logger = logger
        # When True, the LLM invents the distractor options itself (the curated
        # option pool is ignored); only the correct answer stays pinned to the
        # selected ground-truth value in _repair_question_options.
        self.generate_options = bool(generate_options)

        # Route chat.completions through the shared one-client adapter. Key
        # resolved from the endpoint's api_key_env env var when supplied, else
        # the role default (LLM_API_KEY). A 7200s timeout is passed (the SDK
        # default previously applied was less lenient).
        self.adapter = OpenAIChatAdapter.for_chat(
            endpoint,
            self.model,
            logger,
            role="llm",
            timeout=7200,
            api_key_env=api_key_env,
        )

    @classmethod
    def from_config(
        cls,
        config_params: dict,
        system_prompt: str,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        api_key_env: Optional[str] = None,
    ):
        """
        Create LLMQuestionGenerator instance from configuration dictionary with type validation.

        Args:
            config_params: Dictionary containing LLM configuration parameters
            system_prompt: System prompt text
            endpoint: LLM endpoint URL
            model: LLM model name
            logger: Logger instance

        Returns:
            LLMQuestionGenerator: Configured instance with validated parameters
        """
        # Extract and validate parameters
        params = {
            "system_prompt": system_prompt,
            "retry": config_params.get("parameters", {}).get("retry", 1),
            "temperature": config_params.get("parameters", {}).get("temperature", 0.0),
            "top_p": config_params.get("parameters", {}).get("top_p", 0.95),
            "frequency_penalty": config_params.get("parameters", {}).get(
                "frequency_penalty", 0.0
            ),
            "presence_penalty": config_params.get("parameters", {}).get(
                "presence_penalty", 0.0
            ),
            "max_tokens": config_params.get("parameters", {}).get("max_tokens", 2048),
            "stream": config_params.get("parameters", {}).get("stream", True),
            "endpoint": endpoint,
            "model": model,
        }

        validated_params = validate_and_cast_config_params(
            params, cls._PARAM_TYPES, logger
        )

        return cls(
            system_prompt=validated_params["system_prompt"],
            retry=validated_params["retry"],
            temperature=validated_params["temperature"],
            top_p=validated_params["top_p"],
            frequency_penalty=validated_params["frequency_penalty"],
            presence_penalty=validated_params["presence_penalty"],
            max_tokens=validated_params["max_tokens"],
            stream=validated_params["stream"],
            endpoint=validated_params["endpoint"],
            model=validated_params["model"],
            logger=logger,
            api_key_env=api_key_env,
            generate_options=bool(config_params.get("generate_options", False)),
        )

    def _sanitize_model_output(self, text: str) -> str:
        """Remove artifacts from model output."""
        # Remove Markdown code fences
        sanitized = re.sub(r"```[a-zA-Z]*", "", text)
        sanitized = re.sub(r"```", "", sanitized)

        # Remove block comments
        sanitized = re.sub(r"/\*.*?\*/", "", sanitized, flags=re.DOTALL)

        return sanitized

    def _parse_llm_response(self, result: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract a single question JSON.

        Args:
            result (str): The raw LLM response

        Returns:
            Dict[str, Any]: Question dictionary

        Raises:
            ValueError: If the response cannot be parsed as JSON
        """
        # Step 1: cut away any leading chain-of-thought like </think> blocks
        think_close_idx = result.rfind("</think>")
        base_text = (
            result[think_close_idx + len("</think>") :].strip()
            if think_close_idx != -1
            else result.strip()
        )
        self.logger.debug(
            f"Post-think text length={len(base_text)} (think_found={think_close_idx != -1})"
        )

        # Step 2: prefer content inside the first fenced code block, if present
        fence_match = re.search(
            r"```(?:json|javascript|js|python)?\s*([\s\S]*?)\s*```",
            base_text,
            flags=re.IGNORECASE,
        )
        candidate_text = fence_match.group(1).strip() if fence_match else base_text
        self.logger.debug(
            f"Fenced block {'found' if fence_match else 'not found'}; candidate length={len(candidate_text)}"
        )

        # Step 3: sanitize comments and stray fences
        sanitized = self._sanitize_model_output(candidate_text)
        self.logger.debug(f"Sanitized candidate length={len(sanitized)}")

        # Step 4: attempt to decode JSON
        try:
            decoder = json.JSONDecoder()
            # Scan for likely JSON starts
            for match in re.finditer(r"[\[{]", sanitized):
                start_idx = match.start()
                try:
                    obj, end_idx = decoder.raw_decode(sanitized, idx=start_idx)
                    self.logger.debug(
                        f"Decoded JSON segment start={start_idx} end={end_idx}"
                    )
                    return self._normalize_question_output(obj)
                except json.JSONDecodeError:
                    continue
            raise json.JSONDecodeError("No decodable JSON value found", sanitized, 0)

        except json.JSONDecodeError as e:
            self.logger.error(
                f"JSON decode failed: msg={getattr(e, 'msg', '')}, pos={getattr(e, 'pos', None)}"
            )
            error_msg = (
                "Could not parse JSON from model output. "
                "Enable debug logs to inspect sanitized text."
            )
            raise ValueError(error_msg) from e

    def _normalize_question_output(self, obj: Any) -> Dict[str, Any]:
        """
        Normalize the LLM output to expected format for a single question.

        Expected format:
        {
            "variable": "weather_condition",
            "value": "cloudy",
            "question": "What are the weather conditions?",
            "options": {"A": "clear", "B": "cloudy", "C": "rainy", "D": "snowy"},
            "correct_answer": "B"
        }
        """
        required_fields = ["variable", "value", "question", "options", "correct_answer"]

        # If it's already a dict with required fields, validate and return
        if isinstance(obj, dict):
            # Check if it has all required fields directly
            if all(field in obj for field in required_fields):
                return obj

            # Check if it's a wrapper dict
            for key in ("question", "item", "data", "result", "output"):
                inner = obj.get(key)
                if isinstance(inner, dict) and all(
                    field in inner for field in required_fields
                ):
                    return inner

        # If it's a list with one item, extract that item
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            if all(field in obj[0] for field in required_fields):
                return obj[0]

        raise ValueError(
            f"Output must be a dict with required fields: {required_fields}. "
            f"Got: {type(obj).__name__}"
        )

    def _create_user_prompt(
        self, variable_name: str, selected_value: str, all_options: List[str]
    ) -> str:
        """
        Create the user prompt for generating a single question.

        Args:
            variable_name: Name of the variable (e.g., "weather_condition")
            selected_value: Selected value for this variable (e.g., "cloudy")
            all_options: All possible values for this variable

        Returns:
            str: The formatted user prompt
        """
        variable_label = _format_variable_label(variable_name)
        if not all_options:
            # Open mode: the LLM invents the distractors itself. Only the correct
            # value is fixed (ground truth); _repair_question_options keeps these
            # LLM-written options and pins the correct answer to the selected value.
            return f"""Generate a verification question for the following attribute:

Variable: {variable_name}
Readable label: {variable_label}
Correct value: {selected_value}

Create ONE multiple choice question that verifies whether '{selected_value}' is visible in the video frames.

Requirements:
1. The question MUST explicitly name the attribute (use the readable label). No vague wording like "What is shown?".
2. Provide 3-4 total options. Exactly ONE option must be the correct value '{selected_value}'.
3. Invent the OTHER options yourself: plausible but clearly WRONG, visually distinct alternatives for this attribute. Do NOT use near-synonyms, paraphrases, parent/child labels, or overlapping categories of the correct value.
4. Keep options short and mutually exclusive.
5. Format the correct answer as a single letter (A, B, C, or D).

Output MUST be a single JSON object with the following structure:
{{
    "variable": "{variable_name}",
    "value": "{selected_value}",
    "question": "Question text?",
    "options": {{"A": "option1", "B": "option2", "C": "option3", "D": "option4"}},
    "correct_answer": "B"
}}

Do NOT include any explanatory text, markdown fences, or comments. Output ONLY the JSON object."""

        return f"""Generate a verification question for the following variable:

Variable: {variable_name}
Readable label: {variable_label}
Selected value: {selected_value}
All possible values: {all_options}

Create ONE multiple choice question that verifies whether the selected value is present in the image.

Requirements:
1. The question must have 2-4 answer options (A, B, C, D)
2. The selected value '{selected_value}' MUST be one of the options
3. The question should be simple and direct
4. Options should include other possible values from the list above
5. Format the correct answer as a single letter (A, B, C, or D)
6. CRITICAL: The question text MUST explicitly name the variable or attribute being asked about. Use the readable label above when helpful. Do NOT use vague wording like "What is shown?" or "Which option is correct?"
7. Keep the wording domain-agnostic. Do not assume the variable refers to clothing, color, weather, or any other specific use case unless that meaning is already explicit in the variable name or options.
8. Distractor options must be visually and semantically distinct from the selected value. Do NOT include near-synonyms, parent/child labels, or overlapping color families as distractors. For example, if the selected value is "olive brown", do not offer both "olive" and "brown"; choose one correct option and use clearly different colors as distractors. For garment types, avoid ambiguous pairs such as "dress" and "skirt" in the same question when one is the correct answer.

Output MUST be a single JSON object with the following structure:
{{
    "variable": "{variable_name}",
    "value": "{selected_value}",
    "question": "Question text?",
    "options": {{"A": "option1", "B": "option2", "C": "option3", "D": "option4"}},
    "correct_answer": "B"
}}

Do NOT include any explanatory text, markdown fences, or comments. Output ONLY the JSON object."""

    def _resolve_selected_to_allowed(
        self, selected_value: str, all_options: List[str]
    ) -> str:
        """
        Resolve selected value to an allowed option value.

        This handles coarse/base verification values (e.g., "light pink" -> "pink")
        by choosing the closest allowed option when an exact match is unavailable.
        """
        if not all_options:
            return selected_value

        selected_norm = _canonicalize(selected_value)
        allowed_norm_to_raw = {_canonicalize(opt): str(opt) for opt in all_options}

        # Exact normalized match
        if selected_norm in allowed_norm_to_raw:
            return allowed_norm_to_raw[selected_norm]

        # Soft containment matches (useful for "light pink" vs "pink")
        for norm_opt, raw_opt in allowed_norm_to_raw.items():
            if norm_opt and (norm_opt in selected_norm or selected_norm in norm_opt):
                return raw_opt

        # Fallback to original value if nothing better is found
        return selected_value

    def _repair_question_options(
        self,
        question: Dict[str, Any],
        variable_name: str,
        selected_value: str,
        all_options: List[str],
    ) -> Dict[str, Any]:
        """
        Enforce that MCQ options are valid base choices and include selected value.
        """
        repaired = dict(question)
        options = repaired.get("options")
        if not isinstance(options, dict):
            options = {}

        allowed_values = [str(v) for v in all_options] if all_options else []
        allowed_norm = {_canonicalize(v) for v in allowed_values}
        selected_allowed = self._resolve_selected_to_allowed(
            selected_value, allowed_values
        )
        selected_allowed_norm = _canonicalize(selected_allowed)

        # Keep only options that are part of allowed base choices.
        kept_values: List[str] = []
        for _, val in sorted(options.items()):
            val_str = str(val).strip()
            if not val_str:
                continue
            if not allowed_norm or _canonicalize(val_str) in allowed_norm:
                if _is_confusing_option(
                    variable_name, selected_value, selected_allowed, val_str
                ):
                    continue
                if _canonicalize(val_str) not in {
                    _canonicalize(v) for v in kept_values
                }:
                    kept_values.append(val_str)

        # Ensure selected value is included.
        if selected_allowed_norm not in {_canonicalize(v) for v in kept_values}:
            kept_values.insert(0, selected_allowed)

        # Backfill with allowed values so we have 2-4 options.
        for candidate in allowed_values:
            if _canonicalize(candidate) in {_canonicalize(v) for v in kept_values}:
                continue
            if _is_confusing_option(
                variable_name, selected_value, selected_allowed, candidate
            ):
                continue
            kept_values.append(candidate)
            if len(kept_values) >= 4:
                break

        # Guarantee at least 2 options (extreme fallback).
        if len(kept_values) < 2:
            if allowed_values:
                for candidate in allowed_values:
                    if _canonicalize(candidate) != selected_allowed_norm:
                        if _is_confusing_option(
                            variable_name,
                            selected_value,
                            selected_allowed,
                            candidate,
                        ):
                            continue
                        kept_values.append(candidate)
                        break
            if len(kept_values) < 2:
                kept_values.append("other")

        # Limit to 4 options and assign letters.
        final_values = kept_values[:4]
        letters = ["A", "B", "C", "D"]
        final_options = {letters[i]: final_values[i] for i in range(len(final_values))}

        # Set correct_answer to the letter of selected value.
        correct_letter = "A"
        for letter, value in final_options.items():
            if _canonicalize(value) == selected_allowed_norm:
                correct_letter = letter
                break

        repaired["value"] = selected_allowed
        repaired["options"] = final_options
        repaired["correct_answer"] = correct_letter
        return repaired

    def generate_question(
        self, variable_name: str, selected_value: str, all_options: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a verification question for a single variable.

        Args:
            variable_name: Name of the variable (e.g., "weather_condition")
            selected_value: Selected value for this variable (e.g., "cloudy")
            all_options: All possible values for this variable

        Returns:
            Dict[str, Any]: Question dictionary

        Raises:
            RuntimeError: If question generation fails
        """
        self.logger.debug(
            f"Generating question for variable '{variable_name}' with value '{selected_value}'"
        )
        # generate_options mode: pass NO curated pool so the LLM supplies the
        # distractors (an empty pool makes the prompt + repair keep the LLM's
        # options); the correct answer stays pinned to the selected value.
        pool = [] if self.generate_options else all_options
        retries_remaining = self.retry

        while True:
            try:
                user_prompt = self._create_user_prompt(
                    variable_name, selected_value, pool
                )

                self.logger.debug(
                    "Making API call to generate question with structured output"
                )

                # Try structured output first (guided JSON), fall back to regular parsing
                try:
                    completion = self.adapter.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        response_format=self._QUESTION_SCHEMA,
                        stream=False,  # Structured output doesn't support streaming
                    )
                    result = completion.choices[0].message.content
                    question = json.loads(result)
                    self.logger.debug(
                        "Successfully used structured output (guided JSON)"
                    )
                except (
                    openai.APIError,
                    openai.BadRequestError,
                    json.JSONDecodeError,
                    ValueError,
                ) as e:
                    # Fall back to manual parsing if structured output not supported
                    self.logger.debug(
                        f"Structured output not available ({e}), falling back to manual parsing"
                    )
                    completion = self.adapter.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        stream=self.stream,
                    )

                    result = ""
                    if self.stream:
                        # Streaming: iterate over chunks and accumulate content
                        for chunk in completion:
                            if (
                                chunk.choices
                                and chunk.choices[0].delta.content is not None
                            ):
                                result += chunk.choices[0].delta.content
                    else:
                        # Non-streaming: completion is a single response object
                        result = completion.choices[0].message.content

                    # Parse the result
                    question = self._parse_llm_response(result)

                question = self._repair_question_options(
                    question=question,
                    variable_name=variable_name,
                    selected_value=selected_value,
                    all_options=pool,
                )

                self.logger.info(
                    f"Successfully generated question for '{variable_name}'"
                )
                self.logger.debug(f"Generated question: {question}")
                return question

            except Exception as e:
                error_msg = f"Error generating question for '{variable_name}': {e}"
                self.logger.error(error_msg)

                if retries_remaining > 0:
                    retries_remaining -= 1
                    self.logger.info(
                        f"Retrying question generation ({retries_remaining} retries remaining)"
                    )
                    continue

                self.logger.error("Failed to generate question after retries")
                raise RuntimeError(error_msg) from e

    def generate_questions(
        self, variables: Dict[str, str], variable_options: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate verification questions for all given variables (one at a time).

        Args:
            variables: Dictionary of variable names to selected values
            variable_options: Dictionary of variable names to all possible values

        Returns:
            List[Dict[str, Any]]: List of question dictionaries

        Raises:
            RuntimeError: If question generation fails for any variable
        """
        self.logger.info(f"Generating questions for {len(variables)} variables")
        questions = []
        failures: List[str] = []

        for variable_name, selected_value in variables.items():
            all_options = variable_options.get(variable_name, [selected_value])
            try:
                question = self.generate_question(
                    variable_name, selected_value, all_options
                )
                questions.append(question)
            except Exception as e:
                self.logger.error(
                    f"Failed to generate question for '{variable_name}': {e}"
                )
                failures.append(f"{variable_name}: {e}")

        # Verifying only the variables that succeeded lets a sample pass on a
        # subset of the configured checks, so an endpoint outage reads as a clean
        # result. Fail the whole set instead.
        if failures:
            raise RuntimeError(
                f"Question generation failed for {len(failures)} of "
                f"{len(variables)} variable(s), so the sample cannot be fully "
                f"verified: " + "; ".join(failures)
            )

        if not questions:
            raise RuntimeError("Failed to generate any questions")

        self.logger.info(f"Successfully generated {len(questions)} questions")
        return questions
