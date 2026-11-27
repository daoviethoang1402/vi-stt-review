"""Qwen3-ASR adapter, run locally with Hugging Face Transformers (Qwen/Qwen3-ASR-0.6B-hf).

Backend specifics kept here:

1. **Precision**: the checkpoint is bf16. `convert_to_fp32: true` upcasts the weights to
   fp32 at load time (lossless); `false` keeps bf16. A T4 (sm_75) has no bf16 hardware,
   so bf16 there runs through slower fallback kernels: accuracy is meaningful, latency
   is not representative of a GPU with native bf16 (e.g. Jetson Orin).
2. **Request**: `processor.apply_transcription_request(audio, prompt, language)` builds
   the chat template; the output is decoded with `return_format="transcription_only"`,
   which strips the `language …<asr_text>` prefix.
3. **Hotwords**: no boost weight. The canonical hotwords are formatted into
   `hotword_prompt` (e.g. "Vocabulary: {words}.") and passed as `prompt` on `on` only.
4. Greedy decoding (`do_sample=False`), capped by `max_new_tokens`.
"""

from __future__ import annotations

from collections.abc import Sequence

from adapters.base import Segment, check_boost_level, read_waveform, tensor_bytes_mb


class Qwen3AsrSession:
    def __init__(self, adapter: "Qwen3AsrAdapter", prompt: str | None) -> None:
        self._adapter = adapter
        self._prompt = prompt

    def transcribe(self, segment: Segment) -> str:
        import torch

        adapter = self._adapter
        samples, _ = read_waveform(segment.audio)
        inputs = adapter.processor.apply_transcription_request(
            audio=samples, prompt=self._prompt, language=adapter.language
        ).to(adapter.model.device, adapter.model.dtype)
        with torch.inference_mode():
            output = adapter.model.generate(**inputs, max_new_tokens=adapter.max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        return adapter.processor.decode(generated, return_format="transcription_only")[0].strip()

    def close(self) -> None:
        pass


class Qwen3AsrAdapter:
    needs_ground_truth = False

    def load(self, config: dict) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.config = dict(config)
        dtype = torch.float32 if config["convert_to_fp32"] else torch.bfloat16
        device = "cuda" if config.get("provider") == "cuda" else "cpu"
        self.processor = AutoProcessor.from_pretrained(config["path"], local_files_only=True)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            config["path"], dtype=dtype, local_files_only=True
        )
        self.model = model.to(device).eval()
        if self.model.device.type != device or self.model.dtype != dtype:
            raise RuntimeError(f"qwen3-asr: model is {self.model.dtype} on {self.model.device}, "
                               f"expected {dtype} on {device}")
        self.max_new_tokens = int(config["max_new_tokens"])
        self.language = config.get("language") or None

    def weights_mb(self) -> float:
        return tensor_bytes_mb(self.model)

    def open_session(self, hotwords: Sequence[str], boost_level: str) -> Qwen3AsrSession:
        template = self.config.get("hotword_prompt")
        on = check_boost_level(boost_level) == "on"
        prompt = template.format(words=", ".join(hotwords)) if on and hotwords and template else None
        return Qwen3AsrSession(self, prompt)
