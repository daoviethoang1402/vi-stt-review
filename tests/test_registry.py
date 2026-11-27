"""Test bộ kiểm `registry.yaml` — lỗi cấu hình phải bị chặn trước khi nạp model.

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from harness.registry import RegistryError, load
from tests import controls

controls.register()

BASE_MODEL = {"adapter": "oracle", "hotword_enabled": False}


def write(directory: Path, models: list[dict], **overrides) -> Path:
    doc = {
        "version": 4,
        "run": {"num_threads": 1},
        "models": models,
        **overrides,
    }
    path = directory / "registry.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return path


class ValidationTest(unittest.TestCase):
    def load_error(self, models: list[dict], **overrides) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), models, **overrides)
            with self.assertRaises(RegistryError) as caught:
                load(path)
            return str(caught.exception)

    def test_real_registry_is_valid(self) -> None:
        registry = load()
        self.assertEqual(
            registry.names,
            [
                "chunkformer-ctc-large-vie",
                "parakeet-ctc-0.6b-vi",
                "phowhisper-medium",
                "qwen3-asr-0.6b",
                "qwen3-asr-1.7b",
                "zipformer-30m",
                "zipformer-291m",
            ],
        )

    def test_unknown_config_key_is_rejected(self) -> None:
        message = self.load_error(
            [{"name": "m", "config": {**BASE_MODEL, "hotwords_score": 1.5}}]
        )
        self.assertIn("unknown key", message)
        self.assertIn("hotwords_score", message)

    def test_missing_hotword_enabled_is_rejected(self) -> None:
        message = self.load_error([{"name": "m", "config": {"adapter": "oracle"}}])
        self.assertIn("hotword_enabled", message)

    def test_zero_score_is_rejected(self) -> None:
        message = self.load_error(
            [
                {
                    "name": "m",
                    "config": {
                        "adapter": "sherpa-offline",
                        "hotword_enabled": True,
                        "hotword_score": 0,
                        "hotword_score_param": "hotwords_score",
                        "encoder": "e", "decoder": "d", "joiner": "j", "tokens": "t",
                        "decoding_method": "modified_beam_search", "max_active_paths": 8,
                    },
                }
            ]
        )
        self.assertIn("hotword_score", message)

    def test_prompt_model_needs_prompt(self) -> None:
        message = self.load_error(
            [
                {
                    "name": "m",
                    "config": {
                        "adapter": "qwen3-asr-api",
                        "provider": "remote",
                        "hotword_enabled": True,
                        "url": "http://example.invalid/x",
                    },
                }
            ]
        )
        self.assertIn("hotword_prompt", message)

    def test_missing_required_path_is_rejected(self) -> None:
        message = self.load_error(
            [{"name": "m", "config": {"adapter": "chunkformer", "hotword_enabled": False}}]
        )
        self.assertIn("path", message)

    def test_model_level_key_besides_name_and_config_is_rejected(self) -> None:
        message = self.load_error(
            [{"name": "m", "adapter": "oracle", "config": BASE_MODEL}]
        )
        self.assertIn("unknown key", message)

    def test_dtype_must_match_convert_to_fp32(self) -> None:
        base = {"adapter": "qwen3-asr", "hotword_enabled": False, "path": "models/x",
                "max_new_tokens": 8}
        for flag, dtype, ok in ((True, "fp32", True), (False, "bf16", True),
                                (True, "bf16", False), (False, "fp32", False), (None, "fp32", False)):
            with self.subTest(flag=flag, dtype=dtype):
                config = {**base, "dtype": dtype, **({} if flag is None else {"convert_to_fp32": flag})}
                models = [{"name": "m", "config": config}]
                if ok:
                    with tempfile.TemporaryDirectory() as tmp:
                        self.assertEqual(load(write(Path(tmp), models)).names, ["m"])
                else:
                    self.assertIn("convert_to_fp32" if flag is None else "dtype", self.load_error(models))

    def test_unknown_hotword_case_is_rejected(self) -> None:
        message = self.load_error(
            [{"name": "m", "config": {**BASE_MODEL, "hotword_enabled": True,
                                      "hotword_case": "title"}}]
        )
        self.assertIn("hotword_case", message)

    def test_unknown_provider_is_rejected(self) -> None:
        message = self.load_error([{"name": "m", "config": {**BASE_MODEL, "provider": "tpu"}}])
        self.assertIn("provider", message)

    def test_filter_keeps_requested_order(self) -> None:
        registry = load(only=["zipformer-30m", "chunkformer-ctc-large-vie"])
        self.assertEqual(registry.names, ["zipformer-30m", "chunkformer-ctc-large-vie"])

    def test_unknown_model_in_filter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), [{"name": "m", "config": BASE_MODEL}])
            with self.assertRaises(RegistryError) as caught:
                load(path, only=["khong-co"])
            self.assertIn("khong-co", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class ContractTest(unittest.TestCase):
    """Key khai là "bắt buộc" phải thật sự được adapter đọc.

    Đây là test bắt đúng loại lỗi đã xảy ra một lần: registry khai `path` nhưng
    adapter đọc `config["model"]`, nên 4/5 model chết với `KeyError` — mà chỉ model
    đầu tiên được chạy thử mới lộ ra.
    """

    def test_required_keys_are_read_by_the_adapter(self) -> None:
        import inspect

        import adapters

        for name, deal in adapters.CONTRACTS.items():
            if not deal["required"]:
                continue
            cls = adapters.FACTORIES[name]
            source = inspect.getsource(cls.load) + inspect.getsource(cls.open_session)
            for key in sorted(deal["required"]):
                with self.subTest(adapter=name, key=key):
                    self.assertIn(
                        f'"{key}"',
                        source,
                        f"{name}: `{key}` khai bắt buộc nhưng adapter không đọc",
                    )

    def test_contracts_cover_every_registered_adapter(self) -> None:
        import adapters

        self.assertEqual(set(adapters.CONTRACTS), set(adapters.FACTORIES))
