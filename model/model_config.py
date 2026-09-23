"""Load and validate portable model profiles for KGQA experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


MODEL_PROFILE_SCHEMA_VERSION = 2


class ModelProfileError(ValueError):
    """Raised when a model profile is missing or invalid."""


@dataclass(frozen=True)
class ModelProfile:
    """Validated model and artifact metadata independent from a server alias."""

    path: Path
    schema_version: int
    name: str
    model_id: str
    family: str
    parameter_size: str | None
    source: Dict[str, Any]
    runtime_provider: str | None
    connection_type: str | None
    artifact_format: str | None
    artifact_digest: str | None
    artifact_size_bytes: int | None
    instruction_tuned: bool | None
    quantized: bool | None
    quantization_bits: int | None
    quantization_format: str | None
    context_window: int | None
    embedding_length: int | None
    native_capabilities: tuple[str, ...] | None
    supports_completion: bool | None
    supports_tools: bool | None
    supports_thinking: bool | None
    supports_vision: bool | None
    supports_structured_output: bool | None

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
                "parameter_size": self.parameter_size,
            },
            "source": dict(self.source),
            "runtime": {
                "provider": self.runtime_provider,
                "connection_type": self.connection_type,
            },
            "artifact": {
                "format": self.artifact_format,
                "digest": self.artifact_digest,
                "size_bytes": self.artifact_size_bytes,
            },
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
                "embedding_length": self.embedding_length,
                "native": (
                    list(self.native_capabilities)
                    if self.native_capabilities is not None
                    else None
                ),
                "completion": self.supports_completion,
                "tools": self.supports_tools,
                "thinking": self.supports_thinking,
                "vision": self.supports_vision,
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


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field)


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ModelProfileError(f"'{field}' must be true or false.")
    return value


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    return _require_bool(value, field)


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelProfileError(f"'{field}' must be a positive integer.")
    return value


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field)


def _optional_string_list(value: Any, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ModelProfileError(f"'{field}' must be a JSON list or null.")

    normalized = []
    for index, item in enumerate(value):
        normalized.append(
            _require_nonempty_string(item, f"{field}[{index}]")
        )
    return tuple(normalized)


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
    family = _require_nonempty_string(model.get("family", name), "model.family")
    parameter_size = _optional_string(
        model.get("parameter_size"),
        "model.parameter_size",
    )

    source = _require_mapping(root.get("source", {}), "source")
    for key, value in source.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ModelProfileError(
                f"'source.{key}' must be a scalar JSON value or null."
            )

    runtime = _require_mapping(root.get("runtime", {}), "runtime")
    runtime_provider = _optional_string(runtime.get("provider"), "runtime.provider")
    connection_type = _optional_string(
        runtime.get("connection_type"),
        "runtime.connection_type",
    )

    artifact = _require_mapping(root.get("artifact", {}), "artifact")
    artifact_format = _optional_string(artifact.get("format"), "artifact.format")
    artifact_digest = _optional_string(artifact.get("digest"), "artifact.digest")
    artifact_size_bytes = _optional_positive_int(
        artifact.get("size_bytes"),
        "artifact.size_bytes",
    )

    variant = _require_mapping(root.get("variant"), "variant")
    instruction_tuned = _optional_bool(
        variant.get("instruction_tuned"),
        "variant.instruction_tuned",
    )

    quantization = _require_mapping(
        variant.get("quantization"),
        "variant.quantization",
    )
    quantized = _optional_bool(
        quantization.get("enabled"),
        "variant.quantization.enabled",
    )
    bits = _optional_positive_int(
        quantization.get("bits"),
        "variant.quantization.bits",
    )
    quantization_format = _optional_string(
        quantization.get("format"),
        "variant.quantization.format",
    )
    if quantized is False and (bits is not None or quantization_format is not None):
        raise ModelProfileError(
            "Quantization bits/format must be null when quantization is disabled."
        )

    capabilities = _require_mapping(root.get("capabilities"), "capabilities")
    context_window = _optional_positive_int(
        capabilities.get("context_window"),
        "capabilities.context_window",
    )
    embedding_length = _optional_positive_int(
        capabilities.get("embedding_length"),
        "capabilities.embedding_length",
    )
    native_capabilities = _optional_string_list(
        capabilities.get("native"),
        "capabilities.native",
    )
    supports_completion = _optional_bool(
        capabilities.get("completion"),
        "capabilities.completion",
    )
    supports_tools = _optional_bool(
        capabilities.get("tools"),
        "capabilities.tools",
    )
    supports_thinking = _optional_bool(
        capabilities.get("thinking"),
        "capabilities.thinking",
    )
    supports_vision = _optional_bool(
        capabilities.get("vision"),
        "capabilities.vision",
    )
    supports_structured_output = _optional_bool(
        capabilities.get("structured_output"),
        "capabilities.structured_output",
    )

    return ModelProfile(
        path=profile_path,
        schema_version=schema_version,
        name=name,
        model_id=model_id,
        family=family,
        parameter_size=parameter_size,
        source=source,
        runtime_provider=runtime_provider,
        connection_type=connection_type,
        artifact_format=artifact_format,
        artifact_digest=artifact_digest,
        artifact_size_bytes=artifact_size_bytes,
        instruction_tuned=instruction_tuned,
        quantized=quantized,
        quantization_bits=bits,
        quantization_format=quantization_format,
        context_window=context_window,
        embedding_length=embedding_length,
        native_capabilities=native_capabilities,
        supports_completion=supports_completion,
        supports_tools=supports_tools,
        supports_thinking=supports_thinking,
        supports_vision=supports_vision,
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
    if (
        profile.context_window is not None
        and requested_context > profile.context_window
    ):
        raise ModelProfileError(
            f"Context window {requested_context} exceeds the declared limit for "
            f"'{profile.name}' ({profile.context_window})."
        )

    if use_think and profile.supports_thinking is not True:
        state = "unknown" if profile.supports_thinking is None else "unsupported"
        raise ModelProfileError(
            f"Thinking is {state} for model profile '{profile.name}'."
        )

    if structured_output and profile.supports_structured_output is not True:
        state = (
            "unknown"
            if profile.supports_structured_output is None
            else "unsupported"
        )
        raise ModelProfileError(
            f"Structured output is {state} for model profile '{profile.name}'."
        )


def model_result_config(
    profile: ModelProfile,
    *,
    backend_model_id: str,
) -> Dict[str, Any]:
    """Return portable model metadata for inclusion in result configuration."""
    return {
        "model": profile.name,
        "model_family": profile.family,
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
