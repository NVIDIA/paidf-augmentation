# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter base classes for the BYOM multi-endpoint generation layer.

An *adapter* owns exactly one API contract (the wire protocol of an inference
server) and nothing else. It is the only layer that imports a transport SDK
(``openai``, ``requests``, ...). Executors stay transport-agnostic: they build a
normalized request and hand it to ``adapter.invoke``.

Normalized request payload (produced by :class:`BaseExecutor`)::

    {
        "prompt": Optional[str],          # the text instruction / caption
        "media":  {name: {"bytes": b"...", "mime": "image/png", "path": str}},
        "params": {non-null knobs...},    # forwarded to extra_body / JSON body
    }

Adapters return a normalized :class:`Result` (text XOR media bytes), so the
executor can write the output without knowing which contract produced it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

# Fallback per-request timeout (seconds) when an endpoint declares none. A
# wedged/slow endpoint fails after this instead of hanging; pipeline.retry then
# decides whether to re-call. Mirrors the value the image-edit path used.
DEFAULT_REQUEST_TIMEOUT = 120.0

# Cap on requests kept per stage per sample. A VLM verifier calls once per
# question and each call carries an elided-media descriptor, so an unbounded
# list would grow metadata without bound on a wide attribute set. The total is
# still reported, so truncation is visible rather than silent.
MAX_RECORDED_REQUESTS = 16

# Default env var holding the API key for each role, preserving the names that
# already live in examples/.env. Per-endpoint ``api_key`` / ``api_key_env``
# override this (see :func:`resolve_api_key`).
ROLE_API_KEY_ENV: Dict[str, str] = {
    "vlm": "VLM_API_KEY",
    "llm": "LLM_API_KEY",
    "image_edit": "BUILD_NVIDIA_API_KEY",
    "video_transfer": "BUILD_NVIDIA_API_KEY",
    "video_predict": "BUILD_NVIDIA_API_KEY",
    "image2video": "BUILD_NVIDIA_API_KEY",
}


@dataclass
class Result:
    """Normalized adapter result: text XOR media bytes."""

    text: Optional[str] = None
    media_bytes: Optional[bytes] = None
    request_id: Optional[str] = None
    # The raw SDK/HTTP response, for callers (captioners/verifiers) that need
    # full control over parsing. Never required by BaseExecutor.
    raw: Any = field(default=None, repr=False)


def resolve_api_key(endpoint, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """Resolve an endpoint's API key from the environment.

    Precedence: ``$endpoint.api_key_env`` → ``$<role default>`` (see
    :data:`ROLE_API_KEY_ENV`). Keys are **never** read from a config literal: an
    ``api_key`` in YAML/CLI would leak through the override log and validation
    errors, so a key must live in an env var. When ``api_key_env`` names a var
    that happens to be unset, we still fall back to the role default instead of
    silently sending no key (a previous footgun: naming an unset var like
    ``NVIDIA_API_KEY`` yielded ``None`` → SDK sent ``"dummy"`` → 401). Returns
    ``None`` when nothing is configured (some local endpoints need none); a
    *remote* endpoint with no key resolved logs a WARNING, because the next call
    will almost certainly 401.
    """
    role = getattr(endpoint, "role", None)
    # Try the explicitly-named env var first, then the role default; dedupe so a
    # coinciding pair isn't reported twice.
    candidates: List[str] = []
    for name in (getattr(endpoint, "api_key_env", None), ROLE_API_KEY_ENV.get(role)):
        if name and name not in candidates:
            candidates.append(name)
    for name in candidates:
        value = os.getenv(name)
        if value:
            return value

    # Nothing resolved. For a remote endpoint this means a "dummy" key and a
    # near-certain 401 — surface it loudly so it's debuggable, rather than at
    # debug level where it's invisible. Local endpoints usually need no key.
    if candidates and logger:
        url = str(getattr(endpoint, "url", "") or "")
        is_local = (
            "localhost" in url or "127.0.0.1" in url or url.startswith("http://0.0.0.0")
        )
        log_fn = logger.debug if is_local else logger.warning
        log_fn(
            "No API key resolved for endpoint %r (role=%r): env var(s) %s are all "
            "unset. %s",
            getattr(endpoint, "id", "?"),
            role,
            ", ".join(candidates),
            (
                "Local endpoint — usually fine (no key needed)."
                if is_local
                else "Remote endpoint — the request will send no key and almost "
                "certainly 401. Set one of those vars (e.g. --env-file examples/.env)."
            ),
        )
    return None


# A media payload is megabytes of base64; a provenance record wants to say
# *which* bytes were sent, not carry them. Anything shorter than this stays
# verbatim, so prompts and negative prompts are never touched.
ELIDE_MIN_CHARS = 512

# Prose fails this on its first space or period, which is what keeps prompts
# intact. A compiled fullmatch scans in C — a set(value) difference builds a
# per-character set over the whole payload, which is measurable on a 40M-char
# encoded video.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=\r\n]+")

# JSON scalars survive a json.dump; anything else is coerced to its repr so a
# provenance field can never abort the metadata write (and with it the sample).
_JSON_SCALARS = (str, int, float, bool)


# Credential-shaped field names, matched as whole underscore-delimited words at
# the end of the name.
#
# aug_utils' _is_secret_field matches substrings, which is right for its job
# (masking a logged override, where a false positive merely hides a log line)
# but wrong here: it would mask "max_tokens", "stop_token_ids" and
# "skip_special_tokens" — real vLLM sampling knobs — destroying the very
# generation parameters this record exists to preserve. An exemption list can
# never keep up with that family, so match the credential shape instead.
#
# Note "token" is singular-only: "access_token" and "hf_token" are credentials,
# "max_tokens" and "skip_special_tokens" are budgets.
_CREDENTIAL_FIELD_RE = re.compile(
    r"(^|_)(api_?key|apikey|secret|password|passwd|credential|credentials"
    r"|token|bearer|authorization)$",
    re.IGNORECASE,
)


def _is_secret_param(name: Any) -> bool:
    """True when a field name means a credential rather than a generation knob."""
    if not isinstance(name, str) or not name:
        return False
    # *_env names an environment variable, not the secret itself.
    if name.lower().endswith("_env"):
        return False
    return _CREDENTIAL_FIELD_RE.search(name) is not None


def strip_url_credentials(url: str) -> str:
    """Return ``url`` with any ``user:password@`` userinfo removed.

    Basic-auth-in-URL is legal and the endpoint schema does not forbid it, but a
    recorded URL is written into metadata.json alongside the dataset. Masking
    credential-shaped body params while shipping the same secret in the URL
    would be a hole in the same threat model.
    """
    if not url or "@" not in url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.username and not parts.password:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def _looks_base64(value: str) -> bool:
    """True for a long, pure-base64 string (i.e. an encoded media payload)."""
    return len(value) >= ELIDE_MIN_CHARS and _BASE64_RE.fullmatch(value) is not None


def redact_media(value: Any, key: Optional[str] = None) -> Any:
    """Deep-copy a request body with media payloads replaced by descriptors.

    Each elided payload keeps its size and a SHA-256 digest, so a recorded
    request still identifies exactly which input bytes were sent without
    embedding them. Handles the three shapes adapters produce: raw ``bytes``
    (multipart file tuples), ``data:<mime>;base64,...`` URLs, and raw base64.

    ``key`` is the field name ``value`` was found under; a secret-looking name
    is masked outright. ``augmentation.parameters`` is ``extra="allow"`` and is
    merged verbatim into the wire body, so a BYOM server that wants a token as a
    body field would otherwise have it persisted in cleartext to metadata.json.
    """
    if key is not None and _is_secret_param(key):
        return "***"
    if isinstance(value, dict):
        # Keys are not necessarily strings (an OmegaConf/YAML mapping may carry
        # int keys), and JSON needs string keys anyway.
        return {
            str(name): redact_media(item, name if isinstance(name, str) else None)
            for name, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_media(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return {
            "__elided__": "bytes",
            "bytes": len(value),
            "sha256": hashlib.sha256(bytes(value)).hexdigest(),
        }
    if isinstance(value, str):
        prefix, separator, encoded = value.partition(";base64,")
        if separator and prefix.startswith("data:"):
            return {
                "__elided__": "data-url",
                "mime": prefix[len("data:") :],
                "chars": len(encoded),
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            }
        if _looks_base64(value):
            return {
                "__elided__": "base64",
                "chars": len(value),
                "sha256": hashlib.sha256(value.encode()).hexdigest(),
            }
        return value
    if value is None or isinstance(value, _JSON_SCALARS):
        return value
    # Unknown type (Path, datetime, a file handle a caller passed instead of the
    # documented tuple, ...). Keep it describable rather than letting json.dump
    # raise and take the whole sample down with it.
    return repr(value)


def media_items(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return ``(name, media_dict)`` pairs for media present in a payload."""
    media = payload.get("media") or {}
    return [(name, item) for name, item in media.items() if item]


def first_media(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the first media item dict in a payload, or ``None``."""
    items = media_items(payload)
    return items[0][1] if items else None


def control_media(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return ``(name, item)`` pairs for control-modality inputs (kind ``control``)."""
    return [
        (name, item)
        for name, item in media_items(payload)
        if item.get("kind") == "control"
    ]


def primary_media(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the primary (non-control) media item dict, or ``None``.

    The primary input (rgb) is distinct from control modalities (edge/depth/...);
    used by single-file contracts that upload only the primary and reference or
    reject controls separately.
    """
    for _name, item in media_items(payload):
        if item.get("kind") != "control":
            return item
    return None


def reject_control_media(payload: Dict[str, Any], adapter_name: str) -> None:
    """Fail loud if the payload carries control inputs this contract can't send.

    Single-input contracts (chat/images-edit/video sync+async) serialize one
    primary media item; a control modality (edge/depth/seg/vis) would otherwise
    be silently dropped. Raise instead, pointing at the paths that do support it.
    """
    controls = [name for name, _ in control_media(payload)]
    if controls:
        raise ValueError(
            f"{adapter_name}: control inputs {controls} are not supported by this "
            "API contract and would be dropped. Use the 'nim' adapter for "
            "multi-control transfer, or pass controls via parameters "
            "(e.g. a per-modality control_path) rather than as data inputs."
        )


class BaseAdapter(ABC):
    """Base class for all API-contract adapters.

    Args:
        endpoint: The resolved endpoint (any object exposing ``url``, ``model``,
            ``role``, ``api_key``/``api_key_env``, ``timeout`` — typically the
            Pydantic ``Endpoint`` model).
        logger: Logger instance.
    """

    def __init__(self, endpoint, logger: logging.Logger):
        self.endpoint = endpoint
        self.logger = logger
        self.url = str(getattr(endpoint, "url", "") or "").rstrip("/")
        self.model = getattr(endpoint, "model", "") or None
        self.role = getattr(endpoint, "role", None)
        self.api_key = resolve_api_key(endpoint, logger)
        timeout = getattr(endpoint, "timeout", None)
        self.timeout = DEFAULT_REQUEST_TIMEOUT if timeout is None else float(timeout)
        if self.timeout <= 0:
            raise ValueError("endpoint timeout must be > 0 seconds")
        # The concrete route this adapter calls. Subclasses that build a
        # sub-path (``/v1/infer``, ``/v1/videos/sync``, ``/chat/completions``,
        # ...) overwrite it, so provenance reports one comparable field across
        # every contract instead of a base URL for some and a full route for
        # others. Any userinfo is stripped: this string is written into
        # metadata.json, which ships with the dataset.
        self.request_url = strip_url_credentials(self.url)
        # Every *successful* wire request for the current sample, media elided —
        # read by the CLI for output metadata. Holds requests as actually sent
        # (e.g. a seed the retry loop re-rolled), which config cannot
        # reconstruct. A stage may call several times per sample (one question
        # per variable, one answer per question), so this is a list.
        self.recorded_requests: List[Any] = []
        # Successful calls this sample, including any past MAX_RECORDED_REQUESTS,
        # so a truncated list is never mistaken for a complete one. Counts only
        # calls that were recorded: a call that raised is neither counted nor
        # kept, by the same rule that keeps a failed attempt out of the record.
        self.recorded_call_count = 0

    def record_request(self, body: Any) -> None:
        """Store this contract's wire body for provenance, media elided.

        Call this only once the endpoint has accepted the request. A failed
        attempt must not be recorded, or a sample whose output came from an
        earlier successful attempt would be described by the call that produced
        nothing.

        The adapter-resolved API key can't reach here — it rides in a header —
        but a credential-shaped *param* can, so :func:`redact_media` masks
        secret-looking field names.

        Never raises. This runs *after* the endpoint returned, so letting it
        throw would fail an inference that already succeeded — and for a video
        model that is minutes of work discarded for a provenance field.
        """
        try:
            redacted = redact_media(body)
        except Exception as exc:  # noqa: BLE001 - provenance must not fail a call
            self.logger.warning(
                "Could not record request provenance for %s: %s", self.url, exc
            )
            return
        self.recorded_call_count += 1
        if len(self.recorded_requests) < MAX_RECORDED_REQUESTS:
            self.recorded_requests.append(redacted)

    def reset_requests(self) -> None:
        """Drop the recorded requests, called between samples.

        Adapters outlive a sample, so without this a sample's metadata would
        carry the previous sample's calls as well as its own.
        """
        self.recorded_requests = []
        self.recorded_call_count = 0

    @abstractmethod
    def invoke(self, payload: Dict[str, Any]) -> Result:
        """Serialize the normalized ``payload`` onto this contract's wire format,
        call the endpoint, and return a normalized :class:`Result`."""
        raise NotImplementedError
