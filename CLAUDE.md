# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A model-agnostic harness that benchmarks Vietnamese speech-to-text models on one shared dataset (90 samples, 30 per group: `vi`, `vi-en`, `vi-en-acro`) for an on-device virtual receptionist. Each model is only an entry in `registry.yaml`; evaluating a new model should not require changing harness code.

**`docs/OVERVIEW.md` is the user's requirements document and the source of truth.** `docs/METRIC.md` is the authoritative description of how those requirements are implemented. Docs stay in Vietnamese (handover). Code docstrings, CLI help and messages are in English.

Target machine: **Kaggle T4**. `qwen3-asr-1.7b` is served by a company API on a LAN IP (`http://192.168.0.47:3979/transcribe_qwen3_asr`). It is unreachable from Kaggle; the adapter pings it on load and the harness skips the model (`unavailable`) instead of failing.

## Commands

```bash
python -m unittest discover -s tests -t .              # full suite (no GPU, no models needed)
python -m unittest tests.test_scoring.KeywordTest      # one class
python -m unittest tests.test_harness.ControlTest.test_oracle_is_perfect
python data/validate_dataset.py                        # dataset + hotwords.yaml invariants, keyword/ordinary-word collisions
python scripts/download_models.py [--model NAME ...]   # fetch models (pinned HF commits) into the paths registry.yaml uses
python run.py --list
python run.py --model zipformer-30m --output /tmp/x    # quick real smoke test (~1 min)
python run.py [--hotword yes|no|both] [--warmup true|false] [--manifest ...] [--hotwords ...] [--output DIR]
```

`run.py` and `harness.cli` call `cuda_env.ensure_process_env` to set `LD_LIBRARY_PATH` and re-exec before any CUDA library loads. Skipping it makes sherpa-onnx and CTranslate2 fall back to CPU silently. Model parameters are never CLI flags; they live in each model's `config` in `registry.yaml`.

## Architecture

Layers, calls only go downward: `run.py` → `harness/` (process, knows no backend) → `adapters/` (one file per runtime) → leaves (`scoring.py`, which imports nothing from the project, and `cuda_env.py`). `data/` holds the dataset plus standalone tooling (`prepare_dataset.py`, `validate_dataset.py`); the harness never imports it. `data/acronym_review.csv` is an archive of the human listening pass (readings, pron class); no code reads it.

How a run works:
- `harness/cli.py` validates the registry, then spawns one subprocess per model: `python -m harness.cli --single NAME --out PICKLE` plus the same flags. VmHWM only grows, and VRAM must not be shared between models.
- The child runs `runner.run_model` in order: `VramSampler.start` → `adapter.load` → warm-up (first `run.warmup_samples`, not recorded) → `off` pass → `on` pass. Resources come from the `on` pass.
- Load failures become `ModelResult.status`: `unavailable` (raised as `ModelUnavailable`), `load_failed` or `run_failed`. The parent continues with the next model and exits 1 only if no model was measured.
- There are **no gates**: they were removed on purpose.
- `harness/report.py` writes `summary.md` (4 tables), `accuracy.csv`, `resource.csv`, `per_sample/` and `fingerprint.json`.

Adapter interface (`adapters/base.py`): `load(config)`, `open_session(hotwords, boost_level)`, `session.transcribe(segment) -> str`, `close()`, `weights_mb()`. `adapters/__init__.py` holds `FACTORIES` and `CONTRACTS`. `harness/registry.py` rejects unknown or missing keys, a bad `provider` or `hotword_case`, `hotword_score <= 0`, and a missing prompt.

The control adapters `oracle`, `dummy` and `boostable` live in `tests/controls.py` and are registered into `adapters.FACTORIES` only by the tests (`controls.register()`).

## Scoring rules (decided with the user; do not drift)

- **Canonical only.** A keyword counts only when its canonical form appears as a whole token after normalization (NFC, lowercase, punctuation → space). Spelled or phonetic forms are never credited, and `hotwords.yaml` holds only `canonical` + `distractor` (the old `variants` lists and the `hotword_form` knob were removed).
- **Per-sample sets.** `A` = canonicals of `hotwords.yaml` (distractors included) present in the output. `L` = the record's `spoken_form` (a list of canonical strings in order of appearance, repeats kept; order and repeats do not affect the score).
- **Micro averages only**: WER = ΣE/ΣN, Recall = Σ|A∩L|/Σ|L|, Precision = Σ|A∩L|/Σ|A|, FA/1k = Σ|A\L|/ΣN×1000 plus the raw FA count.
- **Not reported:** FIR (it equals 1 − Precision), macro averages, KER, Recall at `off`, ΔRecall, WER quantiles.
- **Cell markers:** `—` = undefined, needs a look (e.g. Precision when Σ|A| = 0; the model stays in the table with a note). `n/a` = not applicable (Recall/Precision for groups whose labels hold no keyword).
- **`off` reports WER only.** `on` reports all four metrics plus resources (precision, p50/p95 per-utterance latency, throughput = Σaudio/Σtime, peak_rss = VmHWM, per-process peak_vram, model_size).
- **Hotwords fed to decoders are canonical for every model.** `hotword_case` adapts casing to each vocab: zipformer and parakeet upper, chunkformer lower. sherpa-onnx silently ignores hotwords whose casing is absent from its BPE vocab.
- **Same decoder in `off` and `on`.** The CTC adapters use pyctcdecode beam search in both conditions, so the difference is the hotwords alone.
- **No canonical may equal an ordinary word in any label.** `AI` was removed for this reason. `validate_dataset.py` enforces it through one rule: `spoken_form` must equal the sequence of keyword tokens in `ref`.

## Manifest schema

Required fields first, in this order: `id`, `data_class`, `audio`, `ref`, `spoken_form`, `annotation_status`. Optional fields after, never read by the harness: `keywords` (ViMedCSS `cs_terms_list`, incomplete; a possible future label source for English terms), `tags`, `source`, `audio_info`. Unknown fields are rejected by the validator.

## Gotchas

- In `hynt/Zipformer-30M-RNNT-6000h` the sherpa token table is the file named `config.json`; the download script saves it as `tokens.txt`.
- PhoWhisper runs via CTranslate2 converted with `quantization=float32` (`models/phowhisper-medium-ct2-fp32`, produced by the download script from `models/phowhisper-medium`).
- On a small local GPU (the dev laptop has a 2 GB MX450), the fp32 zipformer-291m and PhoWhisper OOM and are skipped as `load_failed`. To exercise them, copy the registry with `provider: cpu` and pass a small `--manifest` (audio paths may be absolute).
- `qwen3-asr-0.6b` runs locally (`adapters/qwen3_asr.py`, `Qwen/Qwen3-ASR-0.6B-hf`). `convert_to_fp32: true|false` picks fp32 or native bf16 and `dtype` must match (registry `dtype_flag` rule). bf16 works on sm_75 (T4/MX450) through fallback kernels: accuracy is fine, latency is not representative.
- The Qwen API request contract (`file_field`, `context_field`, `response_field`) in `registry.yaml` is a guess pending confirmation.
