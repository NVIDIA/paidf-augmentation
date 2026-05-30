# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import multistorageclient as msc


def load_system_prompt_from_config(
    config_section: dict, config_dir: str = None, logger: logging.Logger = None
) -> str:
    """
    Load a system prompt from config, supporting both direct text and file references.

    Args:
        config_section: Configuration section containing the system prompt
        config_dir: Directory where the config file is located
        logger: Logger instance for error reporting

    Returns:
        System prompt content as a string
    """
    # Check if there's a file reference
    if "system_prompt_file" in config_section:
        # Load from file
        return _load_prompt_from_file(
            config_section["system_prompt_file"], config_dir, logger
        )
    elif "system_prompt" in config_section:
        # Use direct text from config
        return config_section["system_prompt"]
    else:
        error_msg = (
            "Neither 'system_prompt' nor 'system_prompt_file' found in config section"
        )
        if logger:
            logger.error(error_msg)
        raise KeyError(error_msg)


def _load_prompt_from_file(
    prompt_file_path: str, config_dir: str = None, logger: logging.Logger = None
) -> str:
    """
    Load a prompt from a text file (supports local and remote paths via multistorageclient).

    Args:
        prompt_file_path: Path to the prompt file (can be relative, absolute, or remote URI like s3://)
        config_dir: Directory where the config file is located (for resolving relative paths)
        logger: Logger instance for error reporting

    Returns:
        Content of the prompt file as a string

    Raises:
        FileNotFoundError: If the prompt file cannot be found
        IOError: If the prompt file cannot be read
    """
    # Check if it's a remote path (s3://, msc://, etc.) or absolute local path
    is_remote = prompt_file_path.startswith(("s3://", "msc://", "http://", "https://"))
    is_absolute = os.path.isabs(prompt_file_path)

    if is_remote or is_absolute:
        # Use the path as-is for remote URIs or absolute paths
        full_path = prompt_file_path
    else:
        # If it's a relative path, resolve it relative to the config directory
        if config_dir:
            full_path = os.path.join(config_dir, prompt_file_path)
        else:
            # If no config_dir provided, assume it's relative to current working directory
            full_path = prompt_file_path

    # Normalize the path (only for local paths)
    if not is_remote:
        full_path = os.path.normpath(full_path)

    try:
        # Check if file exists using MSC
        if not msc.is_file(full_path):
            raise FileNotFoundError(f"Prompt file not found: {full_path}")

        # Use multistorageclient to open the file (supports both local and remote)
        with msc.open(full_path, "r") as f:
            content = f.read().strip()

        if logger:
            logger.debug(f"Loaded prompt from file: {full_path}")

        return content

    except FileNotFoundError:
        error_msg = f"Prompt file not found: {full_path}"
        if logger:
            logger.exception(error_msg)
        raise FileNotFoundError(error_msg)

    except Exception as e:
        error_msg = f"Error reading prompt file {full_path}: {e}"
        if logger:
            logger.exception(error_msg)
        raise IOError(error_msg)
