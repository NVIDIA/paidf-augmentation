# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from typing import Any, Dict, List, Union

from pydantic import ValidationError

from .constants import MEDIA_EXTENSIONS, is_media
from .schema import PipelineConfig


def is_remote_path(path: str) -> bool:
    """Return True when path points to a remote or URI-based location."""
    if path is None:
        return False
    return str(path).startswith(
        ("s3://", "msc://", "gs://", "az://", "http://", "https://")
    )


def replace_words(
    text: str, replacements: List[Dict[str, Union[str, List[str]]]]
) -> str:
    """
    Replace words in a text string based on a list of replacement dictionaries.
    Processes categories in the order they appear in the list.
    Within each category, longer words are replaced first to handle overlapping cases.
    Case-insensitive matching is used, so "Blue", "BLUE", and "blue" will all match.

    Args:
        text (str): The input text string
        replacements (list[dict]): List of dictionaries, each containing:
            - 'category': str - the new word to replace with
            - 'words': list[str] - list of words to be replaced

    Returns:
        str: Text with words replaced according to the mapping

    Example:
        >>> text = "The RED-and-White bright CAT"
        >>> replacements = [
        ...     {"category": "color", "words": ["red", "white"]},  # Note: lowercase
        ...     {"category": "lighting", "words": ["bright"]}      # Will match any case
        ... ]
        >>> replace_words(text, replacements)
        'The {color}-and-{color} {lighting} CAT'
    """
    # Make a copy of the text
    result = text

    # Process each category in the order they appear in the list
    for replacement in replacements:
        category = replacement["category"]
        # Sort words within this category by length (longest first)
        words = sorted(replacement["words"], key=len, reverse=True)

        # Replace each word in this category
        for old_word in words:
            # Create patterns for different cases
            patterns = [
                # Match word with hyphens
                rf"(-{re.escape(old_word)}-)|\b{re.escape(old_word)}\b",
                # Match at start of hyphenated word
                rf"\b{re.escape(old_word)}-",
                # Match at end of hyphenated word
                rf"-{re.escape(old_word)}\b",
            ]

            # Apply each pattern
            for pattern in patterns:
                result = re.sub(pattern, f"{{{category}}}", result, flags=re.IGNORECASE)

    return result


# Type Validation and Casting Functions
def validate_and_cast_config_params(
    params: Dict[str, Any], param_types: Dict[str, type], logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Validate and cast configuration parameters to their expected types.

    Args:
        params (Dict[str, Any]): Dictionary of parameters to validate and cast
        param_types (Dict[str, type]): Dictionary mapping parameter names to expected types
        logger (logging.Logger, optional): Logger for error reporting

    Returns:
        Dict[str, Any]: Dictionary with parameters cast to correct types

    Raises:
        TypeError: If a parameter cannot be cast to the expected type
        ValueError: If a parameter value is invalid for the expected type
    """
    validated_params = {}

    for param_name, expected_type in param_types.items():
        if param_name not in params:
            continue  # Skip missing parameters - let the calling code handle defaults

        value = params[param_name]

        # If already the correct type, use as-is
        if isinstance(value, expected_type):
            validated_params[param_name] = value
            continue

        # Attempt type casting
        try:
            if expected_type is bool:
                # Handle boolean conversion specially
                if isinstance(value, str):
                    if value.lower() in ("true", "1", "yes", "on"):
                        validated_params[param_name] = True
                    elif value.lower() in ("false", "0", "no", "off"):
                        validated_params[param_name] = False
                    else:
                        raise ValueError(f"Cannot convert '{value}' to boolean")
                else:
                    validated_params[param_name] = bool(value)
            elif expected_type in (int, float):
                # Handle numeric conversion
                validated_params[param_name] = expected_type(value)
            elif expected_type is str:
                # Handle string conversion
                validated_params[param_name] = str(value)
            else:
                # For other types, try direct casting
                validated_params[param_name] = expected_type(value)

            if logger:
                logger.debug(
                    f"Cast parameter '{param_name}' from {type(value).__name__} to {expected_type.__name__}: {value} -> {validated_params[param_name]}"
                )

        except (ValueError, TypeError) as e:
            error_msg = f"Cannot cast parameter '{param_name}' with value '{value}' (type: {type(value).__name__}) to {expected_type.__name__}: {e}"
            if logger:
                logger.error(error_msg)
            raise TypeError(error_msg)

    return validated_params


# Field/override-path names whose *values* are secrets and must never be logged.
# ``*_env`` is excluded: it names an env var, not the key itself.
_SECRET_FIELD_RE = re.compile(
    r"(api_?key|token|secret|password|passwd|credential)", re.IGNORECASE
)


def _is_secret_field(name: str) -> bool:
    name = (name or "").lower()
    if not name or name.endswith("_env"):
        return False
    return bool(_SECRET_FIELD_RE.search(name))


def redact_overrides(overrides: List[str]) -> List[str]:
    """Mask the value of any secret-like dotlist override (e.g. ``x.api_key=sk-…``)
    so secrets can't leak into logs or a process-arg dump. Others pass through."""
    out: List[str] = []
    for item in overrides:
        key, sep, _value = item.partition("=")
        if sep and _is_secret_field(key.rsplit(".", 1)[-1]):
            out.append(f"{key}=***")
        else:
            out.append(item)
    return out


def format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic ``ValidationError`` without echoing secret input values.

    Pydantic's default string form embeds each error's ``input`` verbatim, which
    can dump a secret straight into the log. Show ``loc``/``msg``/``type`` always,
    and the input only for a non-secret *scalar* field (complex inputs may nest a
    secret elsewhere, so they're omitted)."""
    lines: List[str] = []
    for err in exc.errors():
        loc_parts = [str(x) for x in err.get("loc", ())]
        loc = ".".join(loc_parts) or "<root>"
        field = loc_parts[-1] if loc_parts else ""
        line = f"  {loc}: {err.get('msg')} [{err.get('type')}]"
        value = err.get("input")
        if (
            "input" in err
            and not _is_secret_field(field)
            and isinstance(value, (str, int, float, bool))
        ):
            line += f" (input={value!r})"
        lines.append(line)
    return "\n".join(lines) or str(exc)


# Configuration Validation Functions
def validate_config_structure(config: dict, logger: logging.Logger):
    """Validate configuration against the unified Pydantic schema.

    Args:
        config: Raw config dict (from YAML + OmegaConf).
        logger: Logger for error reporting.

    Returns:
        A ``PipelineConfig`` instance on success, or ``None`` on failure.
    """
    try:
        pipeline_config = PipelineConfig(**config)
        logger.info("Configuration structure validation passed")
        return pipeline_config
    except ValidationError as exc:
        logger.error(
            "Configuration validation failed:\n%s", format_validation_error(exc)
        )
        return None


def validate_sample_data_availability(sample, config, logger: logging.Logger) -> bool:
    """Validate that required data is available for a specific sample.

    Accepts either a raw dict or a typed ``PipelineConfig`` / ``DataSample``.
    """
    try:
        # Support both raw dicts and Pydantic models
        if hasattr(sample, "inputs"):
            inputs = sample.inputs
            rgb_path = inputs.rgb if inputs is not None else None
            controls = inputs.controls if inputs is not None else None
        else:
            inputs = sample.get("inputs") or {}
            rgb_path = inputs.get("rgb") if isinstance(inputs, dict) else None
            controls = inputs.get("controls") if isinstance(inputs, dict) else None

        if rgb_path is not None and rgb_path:
            if not validate_media_file_type(rgb_path, logger):
                return False

        # Check control inputs required by augmentation modalities
        if hasattr(config, "augmentation"):
            aug = config.augmentation
        else:
            aug = (config or {}).get("augmentation")

        if aug is not None:
            if hasattr(aug, "modalities"):
                modalities_cfg = aug.modalities
            else:
                modalities_cfg = (
                    aug.get("modalities") if isinstance(aug, dict) else None
                )

            if modalities_cfg is not None:
                # Determine which modalities have non-None weights
                modality_names = ["edge", "depth", "seg", "vis"]
                for modality in modality_names:
                    weight = (
                        getattr(modalities_cfg, modality, None)
                        if hasattr(modalities_cfg, modality)
                        else modalities_cfg.get(modality)
                        if isinstance(modalities_cfg, dict)
                        else None
                    )
                    if weight is not None:
                        # This modality is required — check control input exists
                        if controls is None:
                            logger.error(
                                f"Sample missing 'controls' section but augmentation "
                                f"requires modality '{modality}'"
                            )
                            return False

                        control_path = (
                            getattr(controls, modality, None)
                            if hasattr(controls, modality)
                            else controls.get(modality)
                            if isinstance(controls, dict)
                            else None
                        )

                        if control_path is not None:
                            if not validate_media_file_type(str(control_path), logger):
                                return False

        return True

    except Exception as e:
        logger.error(f"Error validating sample data: {e}")
        return False


def validate_media_file_type(file_path: str, logger: logging.Logger) -> bool:
    """
    Validate that file is a supported image or video format.

    Args:
        file_path: Path to the media file
        logger: Logger instance for error reporting

    Returns:
        bool: True if file type is supported, False otherwise
    """
    if not is_media(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        logger.error(
            f"Unsupported file type: {file_ext}. Supported extensions: {sorted(MEDIA_EXTENSIONS)}"
        )
        return False
    return True
