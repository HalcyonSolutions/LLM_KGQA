# Backend Setup

The KGQA runners can use either **Ollama directly** or **Open WebUI backed by Ollama**.

## Ollama

Install Ollama from [ollama.com/download](https://ollama.com/download). A standard local installation exposes the Ollama API at `http://localhost:11434`.

Download the exact model variant referenced by the selected model profile. For example:

```bash
ollama pull qwen3:8b
```

Confirm that the model is installed:

```bash
ollama list
```

Whenever possible, the model ID reported by Ollama should match the profile's `model.id`. If another server exposes the same model under a different alias, the experiment runners accept `--model-id` as an override.

## Repository Backend Configuration

A template is provided at:

```text
configs/openwebui_template.json
```

Copy it before editing:

```bash
cp configs/openwebui_template.json configs/openwebui_config.json
```

The generated `configs/openwebui_config.json` is local configuration and should not be committed.

### Direct Ollama

For the simplest local setup, configure the repository to call Ollama directly:

```json
{
  "backend": "ollama",
  "ollama_url": "http://localhost:11434"
}
```

No API key is required in direct Ollama mode.

## Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) provides a browser interface and authenticated API in front of Ollama. Follow the [official installation guide](https://docs.openwebui.com/getting-started/quick-start/) to install and start it.

Depending on the installation method, Open WebUI may be reachable at a port such as `3000` or `8080`. Use the address at which your own instance is actually available.

### Connect Open WebUI to Ollama

After Open WebUI is running:

1. Open **Settings > Admin > Connections**.
2. Under **Manage Ollama API Connections**, connect the Ollama instance.
3. If Open WebUI and Ollama run directly on the same host, the Ollama URL is typically `http://localhost:11434`.
4. If Open WebUI runs in Docker while Ollama runs on the host, use the host address recommended by the Open WebUI documentation, commonly `http://host.docker.internal:11434`.
5. Open **Manage** on the Ollama connection and verify that the required model is visible.
6. If the connection uses a **Model IDs** filter, leave it empty to expose all models from that Ollama instance or add the exact model ID required by the profile.
7. Under **Settings > Admin > Models**, ensure that the model is enabled and visible to the users who will run the experiments.

For example, after installing:

```bash
ollama pull qwen3:8b
```

the `qwen3:8b` model should be available through Open WebUI before running a profile whose `model.id` is `qwen3:8b`.

## Open WebUI API Keys

The repository needs an Open WebUI API key when `backend` is set to `openwebui`.

API keys must first be enabled by an Open WebUI administrator:

1. Log in as an administrator.
2. Open **Settings > Admin > Authentication**.
3. Enable **API Keys** and save the settings.
4. For non-admin accounts, grant the **API Keys** feature permission through the relevant user/group permissions.

Then create a key for the account used by the experiments:

1. Open the user profile menu.
2. Go to **Settings > Account**.
3. In the **API keys** section, click **Show**.
4. Select **Create new secret key**.
5. Copy the generated key and keep it private.

Do not commit API keys to the repository.

Configure Open WebUI in `configs/openwebui_config.json`:

```json
{
  "backend": "openwebui",
  "base_url": "http://localhost:8080",
  "api_key": "sk-<OPENWEBUI_API_KEY>"
}
```

Set `base_url` to the address at which your Open WebUI instance is reachable.

## Model Profiles

Backend setup and model identity are intentionally separate.

The backend configuration tells the repository **where to send requests**. A model profile under `configs/models/` describes **which model artifact the experiment expects**.

For example:

```bash
python ./kgqa_navigation.py \
  --model-config configs/models/qwen3.json \
  ...
```

If the same artifact is available under a different backend model ID, override only that identifier:

```bash
--model-config configs/models/qwen3.json \
--model-id my-server-model-alias
```

See [../configs/models/README.md](../configs/models/README.md) for the model-profile schema and reproducibility metadata.
