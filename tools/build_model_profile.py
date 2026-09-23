#!/usr/bin/env python3
"""Build a KGQA model profile from an OpenWebUI/Ollama model export."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NON_QUANTIZED_LEVELS = {
    "F16",
    "F32",
    "BF16",
    "FP16",
    "FP32",
}


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


def _clean_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _quantization_bits(level: str | None) -> int | None:
    if not level:
        return None
    matches = [
        re.search(r"(?i)(?:^|[_-])Q(\d+)(?:[_-]|$)", level),
        re.search(r"(?i)(?:^|[_-])IQ(\d+)(?:[_-]|$)", level),
        re.search(r"(?i)(?:^|[_-])MXFP(\d+)(?:[_-]|$)", level),
        re.match(r"(?i)^Q(\d+)", level),
        re.match(r"(?i)^IQ(\d+)", level),
        re.match(r"(?i)^MXFP(\d+)", level),
    ]
    for match in matches:
        if match is not None:
            return int(match.group(1))
    return None


def _quantization_state(level: str | None) -> tuple[bool | None, int | None]:
    if level is None:
        return None, None
    normalized = level.upper()
    if normalized in NON_QUANTIZED_LEVELS:
        return False, None
    return True, _quantization_bits(level)


def build_profile(export_path: Path, raw: Any) -> dict[str, Any]:
    entry = _one_model(raw)
    ollama = entry.get("ollama") if isinstance(entry.get("ollama"), dict) else {}
    details = (
        ollama.get("details")
        if isinstance(ollama.get("details"), dict)
        else {}
    )

    model_id = _clean_string(entry.get("id"))
    if model_id is None:
        model_id = _clean_string(ollama.get("model"))
    if model_id is None:
        model_id = _clean_string(ollama.get("name"))
    if model_id is None:
        raise ValueError("Export does not contain a usable model ID.")

    family = _clean_string(details.get("family"))
    if family is None:
        family = model_id.split(":", 1)[0]

    quantization_level = _clean_string(details.get("quantization_level"))
    quantized, bits = _quantization_state(quantization_level)

    raw_capabilities = ollama.get("capabilities")
    if isinstance(raw_capabilities, list):
        native_capabilities = [
            str(value).strip()
            for value in raw_capabilities
            if str(value).strip()
        ]
        capability_set = {value.lower() for value in native_capabilities}
        completion = "completion" in capability_set
        tools = "tools" in capability_set
        thinking = "thinking" in capability_set
        vision = "vision" in capability_set
    else:
        native_capabilities = None
        completion = None
        tools = None
        thinking = None
        vision = None

    provider = _clean_string(entry.get("owned_by"))
    connection_type = (
        _clean_string(ollama.get("connection_type"))
        or _clean_string(entry.get("connection_type"))
    )

    return {
        "schema_version": 2,
        "name": export_path.stem,
        "model": {
            "id": model_id,
            "family": family,
            "parameter_size": _clean_string(details.get("parameter_size")),
        },
        "source": {},
        "runtime": {
            "provider": provider,
            "connection_type": connection_type,
        },
        "artifact": {
            "format": _clean_string(details.get("format")),
            "digest": _clean_string(ollama.get("digest")),
            "size_bytes": _positive_int(ollama.get("size")),
        },
        "variant": {
            "instruction_tuned": None,
            "quantization": {
                "enabled": quantized,
                "bits": bits,
                "format": quantization_level if quantized is not False else None,
            },
        },
        "capabilities": {
            "context_window": _positive_int(details.get("context_length")),
            "embedding_length": _positive_int(details.get("embedding_length")),
            "native": native_capabilities,
            "completion": completion,
            "tools": tools,
            "thinking": thinking,
            "vision": vision,
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
    if profile["capabilities"]["context_window"] is None:
        print("  capabilities.context_window")
    print("  source (optional provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
