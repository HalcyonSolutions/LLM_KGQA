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


class ModelConfigTests(unittest.TestCase):
    def _write_profile(self, root: Path, **overrides):
        profile = {
            "schema_version": 1,
            "name": "test-model-instruct-q4",
            "model": {
                "id": "server:test-q4",
                "family": "test-model",
            },
            "source": {
                "provider": "huggingface",
                "repo_id": "example/test-model",
            },
            "variant": {
                "instruction_tuned": True,
                "quantization": {
                    "enabled": True,
                    "bits": 4,
                    "format": "Q4",
                },
            },
            "capabilities": {
                "context_window": 8192,
                "thinking": False,
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

        self.assertEqual(profile.name, "test-model-instruct-q4")
        self.assertEqual(profile.model_id, "server:test-q4")
        self.assertEqual(profile.family, "test-model")
        self.assertTrue(profile.instruction_tuned)
        self.assertTrue(profile.quantized)
        self.assertEqual(profile.quantization_bits, 4)
        self.assertEqual(profile.context_window, 8192)

    def test_quantized_profile_requires_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(Path(tmp))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["variant"]["quantization"]["bits"] = None
            path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaises(ModelProfileError):
                load_model_profile(path)

    def test_context_window_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_model_profile(self._write_profile(Path(tmp)))

            validate_runtime_settings(profile, context_window=4096)
            with self.assertRaises(ModelProfileError):
                validate_runtime_settings(profile, context_window=16384)

    def test_thinking_capability_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_model_profile(self._write_profile(Path(tmp)))

            with self.assertRaises(ModelProfileError):
                validate_runtime_settings(
                    profile,
                    context_window=4096,
                    use_think=True,
                )

    def test_backend_model_id_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_model_profile(self._write_profile(Path(tmp)))

        self.assertEqual(resolve_backend_model_id(profile), "server:test-q4")
        self.assertEqual(
            resolve_backend_model_id(profile, "another-server-alias"),
            "another-server-alias",
        )

    def test_result_config_keeps_legacy_variant_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_model_profile(self._write_profile(Path(tmp)))

        result = model_result_config(
            profile,
            backend_model_id="resolved-id",
        )
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["model_profile_name"], "test-model-instruct-q4")
        self.assertEqual(result["backend_model_id"], "resolved-id")
        self.assertTrue(result["use_instruct"])
        self.assertTrue(result["use_quantized"])
        self.assertEqual(result["quantization_bits"], 4)


if __name__ == "__main__":
    unittest.main()
