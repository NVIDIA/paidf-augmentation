# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .llm_question_generator import LLMQuestionGenerator
from .vlm_verifier import VLMVerifier


class AttributeVerifier:
    """
    Orchestrates LLM and VLM to verify video attributes.

    This class coordinates the verification pipeline:
    1. Use LLM to generate verification questions
    2. Use VLM to answer questions based on first frame
    3. Compare answers and determine if all checks pass
    """

    def __init__(
        self,
        question_generator: LLMQuestionGenerator,
        vlm_verifier: VLMVerifier,
        logger: logging.Logger,
    ):
        """
        Initialize the AttributeVerifier.

        Args:
            question_generator: LLM question generator instance
            vlm_verifier: VLM verifier instance
            logger: Logger instance
        """
        self.question_generator = question_generator
        self.vlm_verifier = vlm_verifier
        self.logger = logger

    def verify_video_attributes(
        self,
        video_path: str,
        selected_variables: Dict[str, str],
        variable_options: Dict[str, List[str]],
        extra_questions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that the video contains the selected variable values.

        Args:
            video_path: Path to the generated video
            selected_variables: Dictionary of variable names to selected values
            variable_options: Dictionary of variable names to all possible values
            extra_questions: Optional list of extra questions (e.g. multi-view consistency).
                Each item: variable, question, options (dict A/B/C/D -> text), correct_answer (letter).

        Returns:
            Tuple of (passed_attribute_check: bool, verification_details: dict)

        The verification_details dictionary contains:
            - "results": One entry per question — the question itself (variable,
              value, question, options, expected_answer) alongside how the VLM
              answered it. The generated questions are not listed separately;
              every question yields a result, including on error.
            - "summary": Summary statistics
        """
        self.logger.info(f"Starting attribute verification for video: {video_path}")
        self.logger.debug(f"Selected variables: {selected_variables}")

        start_time = time.time()

        verification_details = {
            "results": [],
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                # VLM call failed, no verdict. Kept apart from failed_checks so
                # an outage is not mistaken for output the VLM judged wrong.
                "errored_checks": 0,
            },
        }

        try:
            # Step 1: Generate verification questions using LLM
            questions = []
            if selected_variables:
                self.logger.info("Generating verification questions...")
                questions = self.question_generator.generate_questions(
                    selected_variables, variable_options
                )
            else:
                self.logger.info(
                    "No variables to verify — skipping LLM question generation"
                )
            # Append extra questions (e.g. multi-view consistency) from config
            if extra_questions:
                for eq in extra_questions:
                    if not isinstance(eq, dict):
                        self.logger.warning(
                            "Skipping invalid extra question entry because it is not a dict: %r",
                            eq,
                        )
                        continue

                    question_text = eq.get("question")
                    options = eq.get("options")
                    correct_answer_raw = (
                        str(eq.get("correct_answer", "")).strip().upper()
                    )
                    correct_answer = (
                        correct_answer_raw[:1] if correct_answer_raw else ""
                    )

                    if not question_text or not isinstance(question_text, str):
                        self.logger.warning(
                            "Skipping extra question because 'question' is missing or invalid: %r",
                            eq,
                        )
                        continue
                    if not isinstance(options, dict) or len(options) < 2:
                        self.logger.warning(
                            "Skipping extra question because 'options' is missing or invalid: %r",
                            eq,
                        )
                        continue
                    if correct_answer not in options:
                        self.logger.warning(
                            "Skipping extra question because 'correct_answer' is missing or "
                            "does not match any option key: %r",
                            eq,
                        )
                        continue

                    questions.append(
                        {
                            "variable": eq.get("variable", "extra"),
                            "value": eq.get("value", "N/A"),
                            "question": question_text,
                            "options": options,
                            "correct_answer": correct_answer,
                            "request_reasoning": eq.get("request_reasoning", False),
                        }
                    )
            # The questions are not recorded separately: each one produces a
            # result below (errors included) carrying every field a question
            # has, so a "questions" list would duplicate "results" outright.
            verification_details["summary"]["total_checks"] = len(questions)

            self.logger.info(f"Generated {len(questions)} verification questions")

            # No variables to verify and no extra_questions -> zero questions.
            # Nothing was checked, so this must NOT count as a pass: under strict
            # evaluation a silent "0 questions -> True" would let unverified output
            # through. Fail explicitly instead.
            if not questions:
                self.logger.warning(
                    "Attribute verification produced no questions (no variables "
                    "and no extra_questions); marking as failed so strict "
                    "evaluation does not report an unverified pass."
                )
                verification_details["summary"]["reason"] = "no_questions"
                verification_details["duration_seconds"] = time.time() - start_time
                return False, verification_details

            # Step 2: Verify each question using VLM
            all_passed = True
            for question_data in questions:
                variable = question_data["variable"]
                value = question_data["value"]
                question = question_data["question"]
                options = question_data["options"]
                correct_answer = question_data["correct_answer"]
                request_reasoning = question_data.get("request_reasoning", False)

                self.logger.info(
                    f"Verifying variable '{variable}' with value '{value}'"
                )
                self.logger.debug(f"Question: {question}")

                try:
                    is_correct, vlm_answer, vlm_reasoning = (
                        self.vlm_verifier.verify_question(
                            video_path,
                            question,
                            options,
                            correct_answer,
                            request_reasoning=request_reasoning,
                        )
                    )

                    result = {
                        "variable": variable,
                        "value": value,
                        "question": question,
                        "options": options,
                        "expected_answer": correct_answer,
                        "request_reasoning": request_reasoning,
                        "vlm_answer": vlm_answer,
                        "passed": is_correct,
                    }
                    if vlm_reasoning is not None:
                        result["vlm_reasoning"] = vlm_reasoning

                    verification_details["results"].append(result)

                    if is_correct:
                        verification_details["summary"]["passed_checks"] += 1
                        self.logger.info(
                            f"Verification passed for '{variable}': "
                            f"expected {correct_answer}, got {vlm_answer}"
                        )
                    else:
                        verification_details["summary"]["failed_checks"] += 1
                        all_passed = False
                        self.logger.warning(
                            f"Verification failed for '{variable}': "
                            f"expected {correct_answer}, got {vlm_answer}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Error verifying question for variable '{variable}': "
                        f"the VLM endpoint did not return a verdict: {e}"
                    )
                    result = {
                        "variable": variable,
                        "value": value,
                        "question": question,
                        "options": options,
                        "expected_answer": correct_answer,
                        "request_reasoning": request_reasoning,
                        "vlm_answer": "ERROR",
                        "passed": False,
                        "error": str(e),
                    }
                    verification_details["results"].append(result)
                    verification_details["summary"]["errored_checks"] += 1
                    all_passed = False

            # Log summary
            self.logger.info("=" * 80)
            self.logger.info("Verification Summary:")
            self.logger.info(
                f"  Total checks: {verification_details['summary']['total_checks']}"
            )
            self.logger.info(
                f"  Passed: {verification_details['summary']['passed_checks']}"
            )
            self.logger.info(
                f"  Failed: {verification_details['summary']['failed_checks']}"
            )
            errored = verification_details["summary"]["errored_checks"]
            if errored:
                self.logger.error(
                    f"  Errored: {errored} (the VLM endpoint failed to answer; "
                    f"these are NOT quality failures)"
                )
            self.logger.info(
                f"  Overall result: {'PASSED' if all_passed else 'FAILED'}"
            )
            self.logger.info("=" * 80)

            verification_details["duration_seconds"] = time.time() - start_time

            return all_passed, verification_details

        except Exception as e:
            self.logger.error(
                f"Error during attribute verification: {e}", exc_info=True
            )
            verification_details["error"] = str(e)
            verification_details["duration_seconds"] = time.time() - start_time
            return False, verification_details
