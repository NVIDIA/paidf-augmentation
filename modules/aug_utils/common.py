# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
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


class PrintCaptureHandler(logging.StreamHandler):
    """A custom stream handler that forwards captured output to a logger."""

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def emit(self, record):
        # Forward only if target logger differs from this handler's owner
        if record.name != self.logger.name:
            self.logger.info(self.format(record))
        else:
            super().emit(record)  # fall back to normal stream handling


@contextmanager
def capture_prints(logger):
    """Capture print statements and redirect them to a logger while still printing to terminal.

    Args:
        logger: The logger instance to send captured print statements to.

    Example:
        logger = logging.getLogger(__name__)
        with capture_prints(logger):
            print("This will be both printed and logged")
    """
    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    # Track if we're currently in a logging operation to prevent recursion
    _capturing = False

    class TeeStdout:
        def write(self, data):
            nonlocal _capturing
            new_stdout.write(data)
            # Only log non-empty lines, and prevent recursion
            if data.strip() and not _capturing:
                try:
                    _capturing = True
                    print(f"Capturing print: {data.strip()}")
                    logger.info(data.strip())
                finally:
                    _capturing = False

        def flush(self):
            new_stdout.flush()
            old_stdout.flush()

    sys.stdout = TeeStdout()
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout


def is_template_format(text: str) -> bool:
    """
    Check if text contains template placeholders like {variable}.

    Args:
        text: Text to check

    Returns:
        bool: True if text contains {variable} patterns
    """
    return bool(re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", text))


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


# YAML Configuration Utilities
def read_yaml_config(yaml_file_path: str) -> Dict[str, Any]:
    """
    Read and parse a YAML configuration file.

    Args:
        yaml_file_path (str): Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Parsed YAML configuration

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    if not Path(yaml_file_path).exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file_path}")

    with open(yaml_file_path, "r") as f:
        return yaml.safe_load(f)


def get_output_dir_from_config(config: Dict[str, Any]) -> str:
    """
    Extract the output_dir parameter from Isaac Sim Replicator configuration.

    Args:
        config (Dict[str, Any]): Parsed YAML configuration

    Returns:
        str: The output directory path

    Raises:
        KeyError: If the output_dir parameter is not found
    """
    try:
        return config["isaacsim.replicator.agent"]["replicator"]["parameters"][
            "output_dir"
        ]
    except KeyError as e:
        raise KeyError(
            f"Could not find 'output_dir' parameter in configuration. Missing key: {e}"
        )


def get_output_dir_from_yaml_file(yaml_file_path: str) -> str:
    """
    Read the output_dir parameter directly from a YAML file.

    Args:
        yaml_file_path (str): Path to the YAML configuration file

    Returns:
        str: The output directory path

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        KeyError: If the output_dir parameter is not found
        yaml.YAMLError: If the YAML file is malformed
    """
    config = read_yaml_config(yaml_file_path)
    return get_output_dir_from_config(config)


def get_replicator_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all replicator parameters from Isaac Sim Replicator configuration.

    Args:
        config (Dict[str, Any]): Parsed YAML configuration

    Returns:
        Dict[str, Any]: All replicator parameters

    Raises:
        KeyError: If the replicator parameters section is not found
    """
    try:
        return config["isaacsim"]["replicator"]["agent"]["replicator"]["parameters"]
    except KeyError as e:
        raise KeyError(
            f"Could not find replicator parameters in configuration. Missing key: {e}"
        )


def validate_output_dir(output_dir: str, create_if_missing: bool = False) -> bool:
    """
    Validate that an output directory exists and optionally create it.

    Args:
        output_dir (str): Path to the output directory
        create_if_missing (bool): Whether to create the directory if it doesn't exist

    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    output_path = Path(output_dir)

    if output_path.exists():
        return True

    if create_if_missing:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    return False


def cleanup_oauth_token_file(
    token_file_path: str = "augmentation_steps/captioning/py_llm_oauth_token.json",
) -> bool:
    """
    Remove the OAuth token file for security purposes.

    Args:
        token_file_path (str): Path to the OAuth token file to remove

    Returns:
        bool: True if file was removed or didn't exist, False if removal failed
    """
    try:
        token_path = Path(token_file_path)
        if token_path.exists():
            token_path.unlink()
            print(f"Removed OAuth token file: {token_file_path}")
            return True
        else:
            # File doesn't exist, which is fine
            return True
    except Exception as e:
        # Log the error but don't fail the entire process
        print(f"Warning: Could not remove OAuth token file {token_file_path}: {e}")
        return False


def cleanup_sensitive_files(base_dir: str = ".") -> None:
    """
    Clean up sensitive files before execution.

    Args:
        base_dir (str): Base directory to search for sensitive files
    """
    sensitive_files = [
        "augmentation_steps/captioning/py_llm_oauth_token.json",
        "*.key",
        "*.pem",
        "*.p12",
        "*.pfx",
        "credentials.json",
        "token.json",
    ]

    base_path = Path(base_dir)

    for pattern in sensitive_files:
        try:
            if "*" in pattern:
                # Handle glob patterns
                for file_path in base_path.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"Removed sensitive file: {file_path}")
            else:
                # Handle specific file paths
                file_path = base_path / pattern
                if file_path.exists() and file_path.is_file():
                    file_path.unlink()
                    print(f"Removed sensitive file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not remove sensitive file {pattern}: {e}")


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
        logger.error(f"Configuration validation failed:\n{exc}")
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
