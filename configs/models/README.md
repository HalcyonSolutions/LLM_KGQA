# Model Profiles

Model profiles describe the exact model artifact used by an experiment without relying on repository-specific naming conventions.

## Preferred Workflow: Build from an OpenWebUI/Ollama Export

Export the model metadata from OpenWebUI, then convert the exported JSON:

```bash
python tools/build_model_profile.py /path/to/qwen3-8b-q4_k_m.json
```

By convention:

- the **input filename stem becomes the profile `name`**;
- the exported `id` (falling back to `ollama.model`) becomes `model.id`, the actual server model identifier;
- Ollama artifact metadata such as family, parameter size, GGUF format, digest, size, quantization level, context length, embedding length, and native capabilities are copied when present;
- OpenWebUI user/access/UI metadata is intentionally ignored.

For example, if the input file is named:

```text
qwen3-8b-q4_k_m.json
```

the generated profile contains:

```json
"name": "qwen3-8b-q4_k_m"
```

Rename the export file before conversion if you want a different stable profile name.

The generated file is written to `configs/models/<input-stem>.json` by default.

## Manual Fields

Some properties should not be guessed from the OpenWebUI export. Review and fill these after generation:

- `variant.instruction_tuned`;
- `capabilities.structured_output`;
- `source`, if upstream provenance such as a Hugging Face repository is known.

An unknown manual field is stored as `null`. Runs that explicitly require an unknown capability, such as `--structured-output`, are rejected until the profile declares support.

## Schema

```json
{
  "schema_version": 2,
  "name": "qwen3-8b-q4_k_m",
  "model": {
    "id": "qwen3:8b",
    "family": "qwen3",
    "parameter_size": "8.2B"
  },
  "source": {
    "provider": "huggingface",
    "repo_id": "..."
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
    "completion": true,
    "tools": true,
    "thinking": true,
    "structured_output": null
  }
}
```

## Server-Specific ID Overrides

The generated `model.id` is the ID from the server export. If the same artifact is exposed under a different alias on another server, keep the profile and override only the API-facing identifier:

```bash
--model-config configs/models/qwen3-8b-q4_k_m.json \
--model-id another-server-alias
```

Result files record both the profile metadata and the resolved backend model ID.
