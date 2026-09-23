# Model Profiles

Model profiles describe the model artifact used by an experiment without relying on this repository's historical server naming convention.

A runner receives a profile with:

```bash
--model-config configs/models/qwen2.5-instruct-q4.json
```

The profile's `model.id` is the identifier requested from the configured Ollama/OpenWebUI backend. If another server exposes the same model under a different identifier, override only that deployment-specific value:

```bash
--model-config configs/models/qwen2.5-instruct-q4.json \
--model-id my-server-qwen-alias
```

## Schema

```json
{
  "schema_version": 1,
  "name": "example-model",
  "model": {
    "id": "backend-model-id",
    "family": "model-family"
  },
  "source": {
    "provider": "huggingface",
    "repo_id": "organization/repository"
  },
  "variant": {
    "instruction_tuned": true,
    "quantization": {
      "enabled": true,
      "bits": 4,
      "format": "Q4"
    }
  },
  "capabilities": {
    "context_window": 32768,
    "thinking": false,
    "structured_output": true
  }
}
```

`source` is provenance metadata and may be empty. The experiment does not download or infer deployment settings from an upstream Hugging Face model card.

The loader validates required fields, quantization consistency, context-window limits, thinking support, and structured-output support before a run starts. The API client separately verifies that the resolved backend model ID can be matched against the connected server's advertised models.
