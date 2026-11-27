# Code base — kiến trúc và phụ thuộc

Đi kèm [PIPELINE.md](PIPELINE.md) (quy trình) và [METRIC.md](METRIC.md) (cách chấm).
File này trả lời ba câu: *code nằm ở đâu*, *tầng nào gọi tầng nào*, *ranh giới nào không được vượt*.
Docstring và thông báo của code viết bằng tiếng Anh; tài liệu trong `docs/` giữ tiếng Việt.

## Bốn tầng

```
  ENTRY      run.py                         scripts/download_models.py (độc lập)
                │
  ĐIỀU PHỐI  harness/  cli · registry · runner · report · probe     biết *quy trình*, không biết backend
                │
  ADAPTER    adapters/  base · sherpa · chunkformer · parakeet_ctc · faster_whisper · qwen3_asr · qwen3_asr_api
                │
  LÁ         scoring.py · cuda_env.py

  DỮ LIỆU    data/  manifest + audio + hotwords.yaml (+ acronym_review.csv: hồ sơ)
                    prepare_dataset.py · validate_dataset.py   (tooling, ngoài đường đo)
```

Mũi tên chỉ **hướng gọi**, không có cạnh đi ngược lên. `scoring.py` không import gì trong dự án.
Harness không import gì trong `data/`. Tooling dữ liệu chỉ mượn `scoring.normalize` để dùng chung
định nghĩa token.

## Vai trò từng file

| File | Trách nhiệm |
|---|---|
| `run.py` | entry: đặt `LD_LIBRARY_PATH` rồi exec lại chính mình, sau đó giao cho `harness.cli` |
| `harness/cli.py` | parse cờ · nạp registry · **spawn mỗi model một process** (`--single`, ẩn) · gom kết quả · ghi report · mã thoát |
| `harness/registry.py` | đọc + **validate** `registry.yaml`, dựng `ModelSpec[]` |
| `harness/runner.py` | nạp manifest và danh sách keyword · `ModelSpec` / `RunContext` / `ModelResult` · warm-up → `off` → `on` cho một model · đo resource · trạng thái `ok` / `unavailable` / `load_failed` / `run_failed` |
| `harness/report.py` | `summary.md`, `accuracy.csv`, `resource.csv`, `per_sample/`, `fingerprint.json` |
| `harness/probe.py` | `peak_rss` (VmHWM), `VramSampler` (per-process hoặc device delta), fingerprint máy |
| `adapters/base.py` | interface `Adapter` / `Session` / `Segment`, `ModelUnavailable`, helper đọc audio, đệm đuôi, đo trọng số |
| `adapters/__init__.py` | `FACTORIES` · `COMMON_KEYS` · `CONTRACTS` (hợp đồng key của `config`) |
| `adapters/sherpa.py` | zipformer offline (`sherpa-offline`) và streaming (`sherpa-online`) dùng chung code |
| `adapters/<runtime>.py` | nạp model, dịch `config` → tham số thật, đưa hotword tới runtime |
| `scoring.py` | `normalize` · `keyword_vocab` · `score_record` (A, L, E) · `Aggregate` (micro) · `by_group` |
| `cuda_env.py` | tìm thư viện CUDA của env rồi `execv` lại tiến trình |
| `scripts/download_models.py` | tải model (pin commit Hugging Face) vào đúng đường dẫn của registry; convert PhoWhisper sang CTranslate2 float32 |
| `data/prepare_dataset.py` | tạo lại dataset và `hotwords.yaml` từ nguồn đã pin revision |
| `data/validate_dataset.py` | kiểm schema manifest (bắt buộc đứng đầu), audio, `ref`, luật `spoken_form` = dãy token keyword trong `ref`, `hotwords.yaml` |
| `tests/controls.py` | adapter đối chứng `oracle` / `dummy` / `boostable`, chỉ đăng ký khi chạy test |
| `tests/test_*.py` | hàm chấm · harness + CLI trên adapter đối chứng · registry · tooling dữ liệu |

## Ranh giới bất biến

| Ranh giới | Vì sao | Test chặn |
|---|---|---|
| `scoring.py` không import gì trong dự án; tập keyword được truyền vào và giống nhau ở mọi model | mỗi model chấm bằng một tập đáp án riêng thì không so được | `test_scoring_imports_nothing_from_the_project` |
| Hàm chấm không có tham số điều kiện (`off`/`on`) | cùng output phải ra cùng điểm ở mọi điều kiện | `test_score_has_no_condition_parameter` |
| Chỉ canonical được tính, không bao giờ dạng đánh vần | yêu cầu gốc của khảo sát | `test_spelled_or_phonetic_forms_never_count`, `test_boostable_needs_hotwords_and_spelled_forms_never_count` |
| `harness/` không có `if model == ...` | thêm model không phải sửa harness | rà soát thủ công |
| Key "bắt buộc" trong `adapters.CONTRACTS` phải thật sự được adapter đọc | chặn kiểu khai `path` mà adapter đọc `config["model"]` | `test_required_keys_are_read_by_the_adapter` |
| Chỉ adapter đối chứng đặt `needs_ground_truth = True` | adapter thật không bao giờ được thấy nhãn | đối chứng nằm ngoài `adapters/` |
| Canonical không trùng từ thường trong nhãn | tránh FA oan | `test_keyword_equal_to_an_ordinary_word_is_caught` |

## Một lần chạy, theo thứ tự gọi

1. `run.py` → `cuda_env.ensure_process_env(...)`: exec lại chính nó với `LD_LIBRARY_PATH` trỏ vào thư
   viện CUDA của env, **trước** khi bất kỳ thư viện CUDA nào được nạp.
2. `harness.cli.main()` → `registry.load(path, only=--model)`: đọc YAML, validate, dựng `ModelSpec[]`.
3. `run_isolated()` spawn `python -m harness.cli --single <name> --out <pickle> ...` cho **từng** model,
   vì `VmHWM` chỉ tăng và hai model GPU chung process sẽ làm bẩn số VRAM.
4. Trong process con: `build_context()` dựng `RunContext` (records, keyword, điều kiện, warm-up) →
   `run_model()`:
   - `VramSampler.start()` lấy mốc;
   - `adapter.load()`, rồi warm-up;
   - `off`, rồi `on`;
   - `_resource()` trên lượt `on`.
5. Process cha gom `ModelResult`. Model không đo được thì in lý do và chạy tiếp. Sau đó
   `report.write_all()`, rồi thoát mã 1 nếu không model nào đo được.

## Thêm một model

1. Runtime đã có adapter: thêm một mục vào `models` trong `registry.yaml` (`name` + `config`), và một
   recipe trong `scripts/download_models.py` nếu cần tải. Không sửa harness.
2. Runtime mới: viết `adapters/<runtime>.py` (`load`, `open_session`, `weights_mb`; session có
   `transcribe(segment) -> str` và `close()`), khai `CONTRACTS` và `FACTORIES` trong
   `adapters/__init__.py`. Không sửa `harness/`.
3. Chạy `python -m unittest discover -s tests -t .`.
