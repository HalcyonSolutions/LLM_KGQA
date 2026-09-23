"""Load and validate portable model profiles for KGQA experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


MODEL_PROFILE_SCHEMA_VERSION = 1


class ModelProfileError(ValueError):
    """Raised when a model profile is missing or invalid."""


@dataclass(frozen=True)
class ModelProfile:
    """Validated model metadata independent from a particular API server."""

    path: Path
    schema_version: int
    name: str
    model_id: str
    family: str
    source: Dict[str, Any]
    instruction_tuned: bool
    quantized: bool
    quantization_bits: int | None
    quantization_format: str | None
    context_window: int
    supports_thinking: bool
    supports_structured_output: bool

    @property
    def result_name(self) -> str:
        """Return a filesystem-safe profile name for result filenames."""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", self.name).strip("-")
        return safe or "model"

    def to_dict(self) -> Dict[str, Any]:
        """Return the normalized profile representation stored with results."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "model": {
                "id": self.model_id,
                "family": self.family,
            },
            "source": dict(self.source),
            "variant": {
                "instruction_tuned": self.instruction_tuned,
                "quantization": {
                    "enabled": self.quantized,
                    "bits": self.quantization_bits,
                    "format": self.quantization_format,
                },
            },
            "capabilities": {
                "context_window": self.context_window,
                "thinking": self.supports_thinking,
                "structured_output": self.supports_structured_output,
            },
        }


def _require_mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelProfileError(f"'{field}' must be a JSON object.")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelProfileError(f"'{field}' must be a non-empty string.")
    return value.strip()


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ModelProfileError(f"'{field}' must be true or false.")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelProfileError(f"'{field}' must be a positive integer.")
    return value


def load_model_profile(path: str | Path) -> ModelProfile:
    """Load and validate a model profile JSON file."""
    profile_path = Path(path).expanduser()
    if not profile_path.is_file():
        raise ModelProfileError(f"Model profile does not exist: {profile_path}")

    try:
        with profile_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ModelProfileError(
            f"Invalid JSON in model profile {profile_path}: {exc}"
        ) from exc

    root = _require_mapping(raw, "profile")

    schema_version = root.get("schema_version")
    if schema_version != MODEL_PROFILE_SCHEMA_VERSION:
        raise ModelProfileError(
            f"Unsupported model profile schema_version={schema_version!r}; "
            f"expected {MODEL_PROFILE_SCHEMA_VERSION}."
        )

    name = _require_nonempty_string(root.get("name"), "name")

    model = _require_mapping(root.get("model"), "model")
    model_id = _require_nonempty_string(model.get("id"), "model.id")
    family_raw = model.get("family", name)
    family = _require_nonempty_string(family_raw, "model.family")

    source_raw = root.get("source", {})
    source = _require_mapping(source_raw, "source")
    for key, value in source.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ModelProfileError(
                f"'source.{key}' must be a scalar JSON value or null."
            )

    variant = _require_mapping(root.get("variant"), "variant")
    instruction_tuned = _require_bool(
        variant.get("instruction_tuned"),
        "variant.instruction_tuned",
    )

    quantization = _require_mapping(
        variant.get("quantization"),
        "variant.quantization",
    )
    quantized = _require_bool(
        quantization.get("enabled"),
        "variant.quantization.enabled",
    )

    bits = quantization.get("bits")
    quantization_format = quantization.get("format")
    if quantized:
        bits = _require_positive_int(bits, "variant.quantization.bits")
        if quantization_format is not None:
            quantization_format = _require_nonempty_string(
                quantization_format,
                "variant.quantization.format",
            )
    else:
        if bits is not None:
            raise ModelProfileError(
                "'variant.quantization.bits' must be null when quantization is disabled."
            )
        if quantization_format is not None:
            raise ModelProfileError(
                "'variant.quantization.format' must be null when quantization is disabled."
            )

    capabilities = _require_mapping(root.get("capabilities"), "capabilities")
    context_window = _require_positive_int(
        capabilities.get("context_window"),
        "capabilities.context_window",
    )
    supports_thinking = _require_bool(
        capabilities.get("thinking"),
        "capabilities.thinking",
    )
    supports_structured_output = _require_bool(
        capabilities.get("structured_output"),
        "capabilities.structured_output",
    )

    return ModelProfile(
        path=profile_path,
        schema_version=schema_version,
        name=name,
        model_id=model_id,
        family=family,
        source=source,
        instruction_tuned=instruction_tuned,
        quantized=quantized,
        quantization_bits=bits,
        quantization_format=quantization_format,
        context_window=context_window,
        supports_thinking=supports_thinking,
        supports_structured_output=supports_structured_output,
    )


def resolve_backend_model_id(
    profile: ModelProfile,
    override: str | None = None,
) -> str:
    """Resolve the API-facing model ID, optionally overriding the profile value."""
    if override is None:
        return profile.model_id
    return _require_nonempty_string(override, "model_id override")


def validate_runtime_settings(
    profile: ModelProfile,
    *,
    context_window: int,
    use_think: bool = False,
    structured_output: bool = False,
) -> None:
    """Validate experiment-time settings against declared model capabilities."""
    requested_context = _require_positive_int(context_window, "context_window")
    if requested_context > profile.context_window:
        raise ModelProfileError(
            f"Context window {requested_context} exceeds the declared limit for "
            f"'{profile.name}' ({profile.context_window})."
        )

    if use_think and not profile.supports_thinking:
        raise ModelProfileError(
            f"Model profile '{profile.name}' does not declare thinking support."
        )

    if structured_output and not profile.supports_structured_output:
        raise ModelProfileError(
            f"Model profile '{profile.name}' does not declare structured-output support."
        )


def model_result_config(
    profile: ModelProfile,
    *,
    backend_model_id: str,
) -> Dict[str, Any]:
    """Return portable model metadata for inclusion in result configuration."""
    return {
        "model": profile.family,
        "model_profile_name": profile.name,
        "model_config": str(profile.path),
        "backend_model_id": backend_model_id,
        "use_instruct": profile.instruction_tuned,
        "use_quantized": profile.quantized,
        "quantization_bits": profile.quantization_bits,
        "model_profile": profile.to_dict(),
    }


def list_model_profile_names(directory: str | Path = "configs/models") -> list[str]:
    """Return validated, filesystem-safe model profile names from a directory."""
    profile_dir = Path(directory)
    if not profile_dir.is_dir():
        raise ModelProfileError(f"Model profile directory does not exist: {profile_dir}")

    names = []
    for path in sorted(profile_dir.glob("*.json")):
        names.append(load_model_profile(path).result_name)
    return names
