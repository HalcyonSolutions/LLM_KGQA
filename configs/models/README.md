# Model Profiles

Model profiles describe the exact model artifact used by an experiment without relying on repository-specific naming conventions.

## Preferred Workflow: Build from an OpenWebUI/Ollama Export

Export the model metadata from OpenWebUI, then convert the exported JSON:

```bash
python tools/build_model_profile.py /path/to/qwen3.json
```

By convention:

- the **input filename stem becomes the profile `name`**;
- the exported top-level `id` becomes `model.id`, the backend/server identifier observed when the profile was created, with `ollama.model` and `ollama.name` used only as fallbacks;
- available Ollama artifact/runtime metadata is copied into the profile;
- missing export fields remain `null` instead of being guessed;
- OpenWebUI user/access/UI metadata is intentionally ignored;
- `source` is completed manually so that it points to the public source for the exact model variant whenever possible.

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

## Model Source

The `source` block records where another user can obtain or inspect the model variant used by the experiment.

For models distributed through the Ollama library, use:

```json
"source": {
  "provider": "ollama",
  "repository": "qwen3",
  "url": "https://ollama.com/library/qwen3:8b"
}
```

The fields have the following roles:

- `provider`: distribution source, such as `ollama`;
- `repository`: model-family/library name on that source;
- `url`: public page for the exact model variant used whenever one is available.

A separate `tag` field is intentionally not used because the exact variant is already represented by the source URL and `model.id`.

The source URL is intended for portability and discovery. The exact artifact used in an experiment is identified more strongly by the metadata under `artifact`, particularly its `digest`.

For example:

```json
"model": {
  "id": "qwen3:8b",
  "family": "qwen3",
  "parameter_size": "8.2B"
},
"source": {
  "provider": "ollama",
  "repository": "qwen3",
  "url": "https://ollama.com/library/qwen3:8b"
},
"artifact": {
  "format": "gguf",
  "digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
  "size_bytes": 5225388164
}
```

`model.id` is the identifier used by the backend that produced the profile. Another installation may expose the same artifact under a different model ID; in that case, use `--model-id` to override the backend identifier without changing the profile's source or artifact metadata.

## Unknown and Manual Fields

Missing export metadata is represented as `null`. In particular, a missing context length does **not** make a profile invalid; it only means the runner cannot verify that the requested `--context-window` is below the model maximum.

Some properties should not be inferred from OpenWebUI environment metadata and remain manual:

- `variant.instruction_tuned`;
- `capabilities.structured_output`;
- `source`, including the public provider, repository, and exact variant URL.

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
  "source": {
    "provider": "ollama",
    "repository": "qwen3",
    "url": "https://ollama.com/library/qwen3:8b"
  },
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

The profile `name` is the experiment-facing model name and comes from the profile filename.

`model.id` is the backend/server identifier observed when the profile was created. It is not treated as the immutable identity of the model artifact and may be overridden with `--model-id` on another installation.

`model.family` is architecture metadata reported by Ollama and must not be treated as the experiment-facing model name. For example, a derived model may report a Qwen architecture family while retaining a different profile name.

The combination of `source` and `artifact` provides the portable reproducibility information:

- `source` tells another user where to obtain or inspect the model;
- `artifact.digest` identifies the exact artifact used by the reference run;
- `model.id` tells the runner what the original backend called that model.

## Server-Specific ID Overrides

If the same artifact is exposed under a different alias on another server, keep the profile and override only the API-facing identifier:

```bash
--model-config configs/models/qwen3.json \
--model-id another-server-alias
```

Result files record the profile name, architecture family, complete profile metadata, and resolved backend model ID.
