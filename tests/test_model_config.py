import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from model.model_config import (
    ModelProfileError,
    load_model_profile,
    model_result_config,
    resolve_backend_model_id,
    validate_runtime_settings,
)
from tools.build_model_profile import build_profile


class ModelConfigTests(unittest.TestCase):
    def _write_profile(self, root: Path, **overrides):
        profile = {
            "schema_version": 2,
            "name": "test-export",
            "model": {
                "id": "server:test-q4",
                "family": "architecture-family",
                "parameter_size": "8B",
            },
            "source": {},
            "runtime": {
                "provider": "ollama",
                "connection_type": "local",
            },
            "artifact": {
                "format": "gguf",
                "digest": "abc123",
                "size_bytes": 1000,
            },
            "variant": {
                "instruction_tuned": True,
                "quantization": {
                    "enabled": True,
                    "bits": 4,
                    "format": "Q4_K_M",
                },
            },
            "capabilities": {
                "context_window": 8192,
                "embedding_length": 4096,
                "native": ["completion", "tools"],
                "completion": True,
                "tools": True,
                "thinking": False,
                "vision": False,
                "structured_output": True,
            },
        }
        profile.update(overrides)
        path = root / "model.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def test_load_valid_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_model_profile(self._write_profile(Path(tmp)))

        self.assertEqual(profile.name, "test-export")
        self.assertEqual(profile.model_id, "server:test-q4")
        self.assertEqual(profile.family, "architecture-family")
        self.assertEqual(profile.parameter_size, "8B")
        self.assertEqual(profile.native_capabilities, ("completion", "tools"))
        self.assertTrue(profile.quantized)
        self.assertEqual(profile.quantization_format, "Q4_K_M")

    def test_export_omissions_may_remain_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(Path(tmp))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["variant"]["instruction_tuned"] = None
            raw["variant"]["quantization"] = {
                "enabled": None,
                "bits": None,
                "format": None,
            }
            raw["capabilities"]["context_window"] = None
            raw["capabilities"]["embedding_length"] = None
            raw["capabilities"]["native"] = None
            raw["capabilities"]["completion"] = None
            raw["capabilities"]["tools"] = None
            raw["capabilities"]["thinking"] = None
            raw["capabilities"]["vision"] = None
            raw["capabilities"]["structured_output"] = None
            path.write_text(json.dumps(raw), encoding="utf-8")
            profile = load_model_profile(path)

        self.assertIsNone(profile.context_window)
        self.assertIsNone(profile.native_capabilities)
        self.assertIsNone(profile.quantized)
        self.assertIsNone(profile.supports_structured_output)

    def test_unknown_context_does_not_block_runtime_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(Path(tmp))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["capabilities"]["context_window"] = None
            path.write_text(json.dumps(raw), encoding="utf-8")
            profile = load_model_profile(path)

        validate_runtime_settings(profile, context_window=32768)

    def test_known_context_window_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_model_profile(self._write_profile(Path(tmp)))
            validate_runtime_settings(profile, context_window=4096)
            with self.assertRaises(ModelProfileError):
                validate_runtime_settings(profile, context_window=16384)

    def test_structured_output_requires_explicit_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(Path(tmp))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["capabilities"]["structured_output"] = None
            path.write_text(json.dumps(raw), encoding="utf-8")
            profile = load_model_profile(path)

            with self.assertRaises(ModelProfileError):
                validate_runtime_settings(
                    profile,
                    context_window=4096,
                    structured_output=True,
                )

    def test_backend_model_id_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_model_profile(self._write_profile(Path(tmp)))

        self.assertEqual(resolve_backend_model_id(profile), "server:test-q4")
        self.assertEqual(
            resolve_backend_model_id(profile, "another-server-alias"),
            "another-server-alias",
        )

    def test_result_config_uses_profile_name_not_architecture_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_model_profile(self._write_profile(Path(tmp)))

        result = model_result_config(profile, backend_model_id="resolved-id")
        self.assertEqual(result["model"], "test-export")
        self.assertEqual(result["model_family"], "architecture-family")
        self.assertEqual(result["backend_model_id"], "resolved-id")

    def test_openwebui_export_uses_filename_and_server_id(self):
        export = [{
            "id": "qwen3:8b",
            "owned_by": "ollama",
            "ollama": {
                "model": "qwen3:8b",
                "size": 5225388164,
                "digest": "digest-value",
                "connection_type": "local",
                "capabilities": ["completion", "tools", "thinking"],
                "details": {
                    "format": "gguf",
                    "family": "qwen3",
                    "parameter_size": "8.2B",
                    "quantization_level": "Q4_K_M",
                    "context_length": 40960,
                    "embedding_length": 4096,
                },
            },
        }]
        profile = build_profile(Path("renamed-qwen3.json"), export)

        self.assertEqual(profile["name"], "renamed-qwen3")
        self.assertEqual(profile["model"]["id"], "qwen3:8b")
        self.assertEqual(profile["variant"]["quantization"]["bits"], 4)
        self.assertEqual(profile["capabilities"]["native"],
                         ["completion", "tools", "thinking"])
        self.assertTrue(profile["capabilities"]["thinking"])
        self.assertFalse(profile["capabilities"]["vision"])

    def test_missing_context_and_embedding_are_preserved_as_null(self):
        export = [{
            "id": "gemma3:latest",
            "owned_by": "ollama",
            "ollama": {
                "model": "gemma3:latest",
                "capabilities": ["completion"],
                "details": {
                    "format": "gguf",
                    "family": "gemma3",
                    "parameter_size": "4.3B",
                    "quantization_level": "Q4_K_M",
                },
            },
        }]
        profile = build_profile(Path("gemma3.json"), export)

        self.assertIsNone(profile["capabilities"]["context_window"])
        self.assertIsNone(profile["capabilities"]["embedding_length"])
        self.assertEqual(profile["capabilities"]["native"], ["completion"])
        self.assertTrue(profile["capabilities"]["completion"])
        self.assertFalse(profile["capabilities"]["tools"])

    def test_vision_capability_is_preserved(self):
        export = [{
            "id": "ministral-3:8b",
            "ollama": {
                "capabilities": ["vision", "completion", "tools"],
                "details": {"family": "mistral3", "context_length": 262144},
            },
        }]
        profile = build_profile(Path("ministral-3.json"), export)
        self.assertTrue(profile["capabilities"]["vision"])
        self.assertIn("vision", profile["capabilities"]["native"])


if __name__ == "__main__":
    unittest.main()
