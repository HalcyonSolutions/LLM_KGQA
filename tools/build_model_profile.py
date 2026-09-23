#!/usr/bin/env python3
"""Build a KGQA model profile from an OpenWebUI/Ollama model export."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a one-model OpenWebUI/Ollama JSON export into the "
            "repository's model-profile schema."
        )
    )
    parser.add_argument("export_json", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/models"),
        help="Directory for the generated profile (default: configs/models).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing generated profile.",
    )
    return parser.parse_args()


def _one_model(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        return raw[0]
    raise ValueError(
        "Expected one exported model object or a JSON list containing exactly one model."
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _quantization_bits(level: str | None) -> int | None:
    if not level:
        return None
    match = re.search(r"(?i)(?:^|[_-])Q(\d+)(?:[_-]|$)", level)
    if match is None:
        match = re.match(r"(?i)^Q(\d+)", level)
    return int(match.group(1)) if match else None


def build_profile(export_path: Path, raw: Any) -> dict[str, Any]:
    entry = _one_model(raw)
    ollama = entry.get("ollama") if isinstance(entry.get("ollama"), dict) else {}
    details = (
        ollama.get("details")
        if isinstance(ollama.get("details"), dict)
        else {}
    )

    model_id = entry.get("id") or ollama.get("model") or ollama.get("name")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Export does not contain a usable model ID.")

    family = details.get("family")
    if not isinstance(family, str) or not family.strip():
        family = model_id.split(":", 1)[0]

    quantization_level = details.get("quantization_level")
    if not isinstance(quantization_level, str) or not quantization_level.strip():
        quantization_level = None
    bits = _quantization_bits(quantization_level)

    native_capabilities = ollama.get("capabilities")
    if not isinstance(native_capabilities, list):
        native_capabilities = []
    capability_set = {
        str(value).strip().lower()
        for value in native_capabilities
        if str(value).strip()
    }

    context_window = _positive_int(details.get("context_length"))
    if context_window is None:
        raise ValueError(
            "Export does not contain ollama.details.context_length; "
            "add the context-window limit manually before using this profile."
        )

    provider = entry.get("owned_by")
    if not isinstance(provider, str) or not provider.strip():
        provider = None

    connection_type = ollama.get("connection_type") or entry.get("connection_type")
    if not isinstance(connection_type, str) or not connection_type.strip():
        connection_type = None

    parameter_size = details.get("parameter_size")
    if not isinstance(parameter_size, str) or not parameter_size.strip():
        parameter_size = None

    artifact_format = details.get("format")
    if not isinstance(artifact_format, str) or not artifact_format.strip():
        artifact_format = None

    digest = ollama.get("digest")
    if not isinstance(digest, str) or not digest.strip():
        digest = None

    return {
        "schema_version": 2,
        "name": export_path.stem,
        "model": {
            "id": model_id.strip(),
            "family": family.strip(),
            "parameter_size": parameter_size,
        },
        "source": {},
        "runtime": {
            "provider": provider,
            "connection_type": connection_type,
        },
        "artifact": {
            "format": artifact_format,
            "digest": digest,
            "size_bytes": _positive_int(ollama.get("size")),
        },
        "variant": {
            "instruction_tuned": None,
            "quantization": {
                "enabled": quantization_level is not None,
                "bits": bits,
                "format": quantization_level,
            },
        },
        "capabilities": {
            "context_window": context_window,
            "embedding_length": _positive_int(details.get("embedding_length")),
            "completion": "completion" in capability_set,
            "tools": "tools" in capability_set,
            "thinking": "thinking" in capability_set,
            "structured_output": None,
        },
    }


def main() -> int:
    args = parse_args()
    export_path = args.export_json.expanduser()
    with export_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    profile = build_profile(export_path, raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{export_path.stem}.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it."
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")

    print(f"Wrote model profile: {output_path}")
    print("Manual fields still to review:")
    print("  variant.instruction_tuned")
    print("  capabilities.structured_output")
    print("  source (optional provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
