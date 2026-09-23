# Model Profiles

Model profiles describe the exact model artifact used by an experiment without relying on repository-specific naming conventions.

## Preferred Workflow: Build from an OpenWebUI/Ollama Export

Export the model metadata from OpenWebUI, then convert the exported JSON:

```bash
python tools/build_model_profile.py /path/to/qwen3.json
```

By convention:

- the **input filename stem becomes the profile `name`**;
- the exported top-level `id` becomes `model.id`, the actual server model identifier, with `ollama.model` and `ollama.name` used only as fallbacks;
- available Ollama artifact/runtime metadata is copied into the profile;
- missing export fields remain `null` instead of being guessed;
- OpenWebUI user/access/UI metadata is intentionally ignored.

This matters because OpenWebUI/Ollama exports are not uniform. Some models include `context_length` and `embedding_length`, while others omit them. Native capabilities also differ across models.

## Export-Derived Fields

When present, the importer records:

- architecture family and parameter size;
- runtime provider and connection type;
- artifact format, digest, and byte size;
- quantization status, bit width when recognizable, and exact quantization label;
- context-window and embedding lengths;
- the complete native Ollama capability list;
- normalized `completion`, `tools`, `thinking`, and `vision` capability flags.

The complete native capability list is retained so newly introduced Ollama capabilities are not lost simply because the repository does not yet have a dedicated normalized field for them.

## Unknown and Manual Fields

Missing export metadata is represented as `null`. In particular, a missing context length does **not** make a profile invalid; it only means the runner cannot verify that the requested `--context-window` is below the model maximum.

Some properties should not be inferred from OpenWebUI environment metadata and remain manual:

- `variant.instruction_tuned`;
- `capabilities.structured_output`;
- `source`, if upstream provenance such as a Hugging Face repository is known.

Runs that explicitly require an unknown capability, such as `--structured-output`, are rejected until that capability is declared.

## Schema

```json
{
  "schema_version": 2,
  "name": "qwen3",
  "model": {
    "id": "qwen3:8b",
    "family": "qwen3",
    "parameter_size": "8.2B"
  },
  "source": {},
  "runtime": {
    "provider": "ollama",
    "connection_type": "local"
  },
  "artifact": {
    "format": "gguf",
    "digest": "...",
    "size_bytes": 5225388164
  },
  "variant": {
    "instruction_tuned": null,
    "quantization": {
      "enabled": true,
      "bits": 4,
      "format": "Q4_K_M"
    }
  },
  "capabilities": {
    "context_window": 40960,
    "embedding_length": 4096,
    "native": [
      "completion",
      "tools",
      "thinking"
    ],
    "completion": true,
    "tools": true,
    "thinking": true,
    "vision": false,
    "structured_output": null
  }
}
```

For an export that omits context information, the corresponding fields are simply:

```json
"context_window": null,
"embedding_length": null
```

## Naming Semantics

The profile `name` is the experiment-facing model name and comes only from the export filename. The `model.id` field is the actual backend/server identifier.

`model.family` is architecture metadata reported by Ollama and must not be treated as the experiment-facing name. For example, a derived model may report a Qwen architecture family while retaining a different profile name.

## Server-Specific ID Overrides

If the same artifact is exposed under a different alias on another server, keep the profile and override only the API-facing identifier:

```bash
--model-config configs/models/qwen3.json \
--model-id another-server-alias
```

Result files record the profile name, architecture family, complete profile metadata, and resolved backend model ID.
