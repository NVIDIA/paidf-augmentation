# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def load_secrets(logger: Optional[logging.Logger] = None):
    """
    Load secrets from /var/secrets/secrets.json and export them to the environment.
    This is useful for Kubernetes deployments where secrets are mounted as files.

    For MULTISTORAGECLIENT_CONFIGURATION, writes config to a temp file and sets
    MSC_CONFIG env var so that msc.open(), msc.is_file(), etc. work directly.
    Also supports MULTISTORAGECLIENT_CONFIGURATION passed directly as an env var
    (secrets file takes priority if both are present).

    Args:
        logger: Optional logger instance for logging messages.
                If not provided, prints to stdout.
    """
    secrets_path = "/var/secrets/secrets.json"
    msc_config_handled = False

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)
        elif level != "debug":
            print(f"[{level.upper()}] {msg}")

    # Process secrets file first (takes priority)
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r") as f:
                secrets = json.load(f)

                for key, value in secrets.items():
                    # Handle MSC config specially - write to file and set MSC_CONFIG
                    if key == "MULTISTORAGECLIENT_CONFIGURATION":
                        _setup_msc_config(value, _log)
                        msc_config_handled = True
                    else:
                        os.environ[key] = str(value)
                        _log(f"Exported secret: {key}")

        except Exception as e:
            _log(f"Failed to load secrets from {secrets_path}: {e}", "error")
    else:
        _log(f"No secrets file found at {secrets_path}", "debug")

    # Fall back to environment variable if not handled by secrets file
    if not msc_config_handled and "MULTISTORAGECLIENT_CONFIGURATION" in os.environ:
        _setup_msc_config(os.environ["MULTISTORAGECLIENT_CONFIGURATION"], _log)


def _setup_msc_config(config: Any, log_fn) -> None:
    """
    Write MSC config to a temp file and set MSC_CONFIG env var.

    This allows msc.open(), msc.is_file(), etc. to work directly with
    any URL (s3://, msc://, local paths) using path_mapping from config.
    """
    try:
        # Parse config if it's a JSON string
        if isinstance(config, str):
            config = json.loads(config)

        # Write to a persistent temp file (not auto-deleted)
        config_path = "/tmp/msc_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Set MSC_CONFIG so shortcuts use this config
        os.environ["MSC_CONFIG"] = config_path
        log_fn(f"MSC config written to {config_path} and MSC_CONFIG env var set")

    except Exception as e:
        log_fn(f"Failed to setup MSC config: {e}", "error")


class NVCFProgressTracker:
    """
    Generic progress tracker for NVCF Task environments.

    Auto-detects NVCF environment and writes progress to the designated file.
    Falls back to no-op when not running in NVCF, making it safe to use anywhere.

    Usage:
        # Simple usage
        tracker = NVCFProgressTracker(logger=logger)
        tracker.update(percent=25, metadata={"step": "preprocessing"})
        tracker.update(percent=50)
        tracker.complete()

        # Context manager usage
        with NVCFProgressTracker(logger=logger) as tracker:
            for i, item in enumerate(items):
                process(item)
                tracker.update(percent=(i + 1) / len(items) * 100)
        # Automatically calls complete() on exit
    """

    # Required environment variables for NVCF Task environment
    REQUIRED_ENV_VARS = [
        "NVCT_TASK_ID",
        "NVCT_TASK_NAME",
        "NVCT_NCA_ID",
        "NVCT_PROGRESS_FILE_PATH",
        "NVCT_RESULTS_DIR",
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the progress tracker.

        Args:
            logger: Optional logger for status messages.
        """
        self._logger = logger
        self._metadata: Dict[str, Any] = {}
        self._current_percent = 0

        # Check if running in NVCF environment
        self._is_nvcf = all(os.getenv(var) for var in self.REQUIRED_ENV_VARS)

        if self._is_nvcf:
            self._task_id = os.getenv("NVCT_TASK_ID")
            self._task_name = os.getenv("NVCT_TASK_NAME")
            self._nca_id = os.getenv("NVCT_NCA_ID")
            self._progress_path = os.getenv("NVCT_PROGRESS_FILE_PATH")
            self._results_dir = os.getenv("NVCT_RESULTS_DIR")
            self._log("NVCF Task environment detected", level="info")
        else:
            self._log(
                "Not running in NVCF Task environment (progress tracking disabled)",
                level="debug",
            )

    def _log(self, message: str, level: str = "info"):
        """Log a message using logger or print."""
        if self._logger:
            getattr(self._logger, level)(message)
        elif level != "debug":  # Don't print debug messages without logger
            print(f"[{level.upper()}] {message}")

    @property
    def is_nvcf(self) -> bool:
        """Check if running in NVCF Task environment."""
        return self._is_nvcf

    @property
    def results_dir(self) -> Optional[str]:
        """Get the NVCF results directory path."""
        return self._results_dir if self._is_nvcf else None

    def set_metadata(self, key: str, value: Any):
        """Set a custom metadata field."""
        self._metadata[key] = value

    def update(self, percent: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Update progress file with current status.

        Args:
            percent: Percentage complete (0-100).
            metadata: Additional metadata to include in this update.
        """
        if not self._is_nvcf:
            return

        # Clamp percent to 0-99 (100 is reserved for complete())
        self._current_percent = max(0, min(99, percent))
        self._write_progress(self._current_percent, metadata)

    def complete(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Mark the task as complete (100%).

        Args:
            metadata: Additional metadata to include in the final update.
        """
        if not self._is_nvcf:
            self._log("Task complete", level="info")
            return

        self._write_progress(100, metadata)
        self._log("Task complete", level="info")

    def _write_progress(
        self, percent: float, extra_metadata: Optional[Dict[str, Any]] = None
    ):
        """Write progress to the NVCF progress file."""
        try:
            payload = {
                "taskId": self._task_id,
                "percentComplete": int(percent),
                "name": self._task_name,
                "metadata": {
                    "resultsDir": self._results_dir,
                    **self._metadata,
                    **(extra_metadata or {}),
                },
                "lastUpdatedAt": datetime.now(timezone.utc).isoformat(),
            }

            with open(self._progress_path, "w") as f:
                json.dump(payload, f, indent=2)

            self._log(f"Progress updated: {int(percent)}%", level="debug")

        except Exception as e:
            self._log(f"Failed to write NVCF progress file: {e}", level="error")

    def __enter__(self) -> "NVCFProgressTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically complete."""
        if exc_type is None:
            self.complete()
        else:
            # Task failed with exception
            self.complete(metadata={"error": str(exc_val)})
        return False  # Don't suppress exceptions
