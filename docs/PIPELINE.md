# Pipeline khảo sát mô hình STT — model-agnostic

Metric: [METRIC.md](METRIC.md) là ground truth, tài liệu
này không định nghĩa lại. Kiến trúc code: [CODEBASE.md](CODEBASE.md).

## Yêu cầu

Chọn model speech-to-text cho lễ tân ảo chạy on-device:

| # | Yêu cầu |
|---|---|
| 1 | Chạy được trên **Jetson Orin NX 8 GB Unified Memory**, không cần TensorRT (FP32/FP16/INT8) |
| 2 | Bắt được **từ tiếng Anh** lẫn trong câu tiếng Việt |
| 3 | Nạp được **keyword** (hiện chỉ là từ viết tắt) liên quan đến công ty/tổ chức/phòng ban |

Khảo sát **không** tìm "model tốt nhất". Nó trả lời *mỗi model đánh đổi những gì*, nên output là
bảng nhiều chiều, không phải một điểm tổng hợp. Không có gate loại model: model nào nạp được thì được
đo và báo cáo đầy đủ.

**Máy chạy: Kaggle T4** (background process). Riêng `qwen3-asr-1.7b` chạy qua API trên mạng công ty,
nên chỉ đo được khi chạy trên máy có kết nối tới mạng đó (xem *Model qua API*). Số tài nguyên là số
của máy chạy, không phải số Jetson: dùng để so giữa các model, không suy ra tuyệt đối cho Orin.

## Nguyên tắc

1. **Harness không biết model.** Không có `if model == ...` ngoài `adapters/`.
2. **Thêm model = thêm 1 mục trong `models` của `registry.yaml`** (+ 1 adapter nếu là runtime mới). Không sửa dữ liệu, không sửa metric.
3. **Khác biệt backend nằm ở adapter**: precision, cách đưa hotword tới runtime, hoa thường của vocab.
4. **Đo cái quan sát được.** Peak RAM/VRAM đọc từ OS và driver, không lấy từ log của model.
5. **"API nhận hotword" ≠ "hotword có tác dụng".** `sherpa-onnx` nhận `hotwords_file` rồi bỏ qua im lặng nếu `decoding_method` sai hay hotword không có trong vocab. Registry chặn cấu hình sai; adapter `boostable` trong test chứng minh harness thật sự truyền hotword đi.
6. **Không gộp điểm.**

## Quy trình trên mỗi model

| Bước | Làm gì | Báo cáo |
|---|---|---|
| warm-up | chạy `run.warmup_samples` mẫu đầu (mặc định 3) một lần, không ghi lại. Tắt bằng `--warmup false` | — |
| `off` | **1 lượt** toàn dataset, không hotword | **chỉ WER**: toàn bộ + 3 nhóm, bảng riêng |
| `on` | **1 lượt** toàn dataset, có hotword | WER, Recall, Precision, FA/1k (toàn bộ + phụ lục theo nhóm) và resource footprint |

`--hotword no` chỉ chạy `off`, `--hotword yes` chỉ chạy `on`, `--hotword both` (mặc định) chạy cả
hai theo thứ tự trên. Model có `hotword_enabled: false` thì không chạy `on`, và report ghi chú.

**Cả hai lượt dùng cùng một decoder**, chỉ khác hotword. Với CTC (chunkformer, parakeet), `off` cũng
chạy beam search pyctcdecode, chỉ là không có hotword. Nếu `off` dùng greedy và `on` dùng beam thì
chênh lệch giữa hai lượt trộn lẫn hiệu ứng của beam search với hiệu ứng của hotword.

## Kiến trúc

```
   run.py  →  đọc registry.yaml, kiểm cấu hình, mỗi model một process
                              │
   data/   manifest + audio + hotwords.yaml   ← dùng chung cho mọi model
                              │
   HARNESS   nạp → warm-up → off → on → đo tài nguyên → report
                              │  chỉ gọi qua interface adapter (mục dưới)
   ADAPTER   1 file / runtime: sherpa (offline + streaming) · chunkformer · parakeet_ctc
                              · faster_whisper · qwen3_asr · qwen3_asr_api   (đối chứng nằm trong tests/)
                              │
   MODEL     chỉ là dữ liệu trong registry.yaml; tải bằng scripts/download_models.py
```

Model nạp lỗi (thiếu file, hết bộ nhớ) hay không truy cập được (API) thì bị **bỏ qua**, harness chạy
tiếp model sau và ghi lý do vào bảng *Models not measured* của `summary.md`. Không model nào đo được ⇒
`run.py` thoát mã 1.

## Các mô hình

| `name` trong registry | Repo nguồn (pin commit trong `scripts/download_models.py`) | Adapter | Precision | Đường hotword | `hotword_case` |
|---|---|---|---|---|---|
| `chunkformer-ctc-large-vie` | [khanhld/chunkformer-ctc-large-vie](https://huggingface.co/khanhld/chunkformer-ctc-large-vie) | `chunkformer` | fp32 | pyctcdecode `hotwords=` trên logits CTC | `lower` |
| `parakeet-ctc-0.6b-vi` | [nvidia/parakeet-ctc-0.6b-Vietnamese](https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese) | `parakeet-ctc` | fp32 | NeMo strategy `pyctcdecode`, `hotwords` | `upper` |
| `phowhisper-medium` | [vinai/PhoWhisper-medium](https://huggingface.co/vinai/PhoWhisper-medium) → CTranslate2 float32 | `faster-whisper` | fp32 | tham số `hotwords` (prompt của decoder) | `as_is` |
| `qwen3-asr-0.6b` | [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) (bản Transformers của Qwen/Qwen3-ASR-0.6B) | `qwen3-asr` | fp32 hoặc bf16 (`convert_to_fp32`) | `prompt` của `apply_transcription_request` | `as_is` |
| `qwen3-asr-1.7b` | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), API công ty | `qwen3-asr-api` | bf16 (server) | trường context của request | `as_is` |
| `zipformer-30m` | [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) | `sherpa-offline` | fp32 | `hotwords_file` của sherpa-onnx | `upper` |
| `zipformer-291m` | [stdo/PengChengStarling](https://huggingface.co/stdo/PengChengStarling) (streaming, 8 ngôn ngữ) | `sherpa-online` | fp32 | `hotwords_file` của sherpa-onnx | `upper` |

Tải model: `python scripts/download_models.py` (cả bộ, khoảng 11,5 GB, gồm 3 GB checkpoint PhoWhisper
gốc và 3 GB bản CTranslate2 float32) hoặc `--model <tên>`.
Script tải đúng commit đã pin, bỏ qua file đã có đủ kích thước, và convert PhoWhisper sang CTranslate2
float32 một lần. Repo `hynt/Zipformer-30M-RNNT-6000h` để bảng token của sherpa-onnx dưới tên
`config.json`; script lưu lại thành `tokens.txt`.

### Precision của `qwen3-asr-0.6b`

Checkpoint gốc là bf16. Cờ `convert_to_fp32` trong `config` chọn precision lúc chạy:

| `convert_to_fp32` | `dtype` (phải khớp) | Trọng số | Ghi chú |
|---|---|---:|---|
| `true` | `fp32` | ~2 985 MB | nâng bf16 → fp32 lúc nạp, không mất thông tin; an toàn về số học |
| `false` | `bf16` | ~1 492 MB | giữ nguyên bf16. T4 (sm_75) không có bf16 bằng phần cứng: chạy được qua kernel dự phòng (đã thử trên MX450, cũng sm_75), độ chính xác dùng được nhưng **latency không đại diện** cho GPU có bf16 thật như Orin |

Registry từ chối cấu hình lệch (`convert_to_fp32: true` mà `dtype: bf16`), nên cột `precision` của
report luôn là precision thật đã chạy. Muốn so cả hai thì khai hai mục registry, chỉ khác cặp này.
Bản `qwen3-asr-0.6b` từng bị loại ở lần khảo sát trên MX450 (phải ép int8, sát trần VRAM); nay chạy
lại trên T4.

### Hotword nạp vào decoder

Mọi model nạp **canonical**: `ADH`, không phải `a dê hát`. Lý do:

- Model boost đường giải mã (sherpa, pyctcdecode) mà được nạp dạng đánh vần thì càng hay **in dạng
  đánh vần**, mà dạng đó lại không bao giờ được tính đúng (METRIC.md).
- Model nhận prompt (Whisper, Qwen) tự hiểu ngữ cảnh: nghe "a dê hát" rồi viết `ADH` nhờ prompt.
  Prompt chứa hàng nghìn cách đánh vần chỉ dài vô ích (Whisper giới hạn khoảng 224 token prompt).

`hotword_case` đổi hoa thường cho khớp vocab của model. Đây là thuộc tính của model, không phải hotword
khác: hotword viết sai hoa thường so với vocab sẽ bị sherpa-onnx bỏ qua im lặng. Model boost đường
giải mã mà vẫn không in được canonical (vd. vocab thiếu token Latin) là một **kết quả khảo sát**,
không lách bằng cách nạp dạng đánh vần. (Danh sách cách đánh vần `variants` và nút `hotword_form`
đã bị gỡ.)

### Model qua API

`qwen3-asr-1.7b` gọi `http://192.168.0.47:3979/transcribe_qwen3_asr`. Lúc nạp, adapter gửi
`ping_attempts` (mặc định 2) request thật (1 giây im lặng). Nếu không request nào thành công thì model
có trạng thái `unavailable` và bị bỏ qua. Đây là tình huống bình thường khi chạy trên Kaggle. Chạy
riêng model này trên máy có mạng công ty:

```bash
python run.py --model qwen3-asr-1.7b --output report-qwen
```

Hợp đồng request (tên trường audio, trường context, khoá JSON chứa transcript) khai trong `config` của
model: `file_field`, `context_field`, `response_field`.

## Interface adapter & Registry

**Interface** (`adapters/base.py`). Harness chỉ gọi:

| Thao tác | Vào | Ra |
|---|---|---|
| `load(config)` | `config` của registry + `num_threads` | — (lỗi ⇒ `load_failed`; `ModelUnavailable` ⇒ `unavailable`) |
| `open_session(hotwords, boost_level)` | danh sách canonical đã đổi hoa thường theo `hotword_case` (rỗng ở `off`) + `off`/`on` | `Session` |
| `session.transcribe(segment)` | `Segment(id, audio)`: wav 16 kHz mono | chuỗi output nguyên văn: **đây là vùng đo latency** |
| `session.close()` | — | — |
| `weights_mb()` | — | kích thước trọng số, `None` với model remote |

Output được lưu nguyên văn ở trường `text_raw` của `per_sample/`, **trước** mọi chuẩn hoá, để audit.
Adapter không được tự chuẩn hoá hay sửa output.

Ba adapter đối chứng chỉ dùng trong test, không có trong registry: `oracle` (trả đúng `ref`),
`dummy` (trả rỗng), `boostable` (chỉ in canonical khi acronym có trong hotword được cấp). Chỉ chúng
được đặt `needs_ground_truth = True` để nhận nhãn.

**Registry**: `registry.yaml` là nơi duy nhất biết về model. Khoá mức gốc: `version`, `run`
(`num_threads`, `warmup_samples`), `models`. Mỗi model chỉ có `name` + `config`; **mọi tham số nằm
trong `config`**, không có cờ CLI nào ghi đè. Hợp đồng key của từng adapter nằm ở
`adapters.CONTRACTS` và được kiểm lúc nạp ([harness/registry.py](../harness/registry.py)). Những lỗi
sau làm dừng ngay:

- key lạ hoặc thiếu key bắt buộc;
- `provider` lạ;
- `hotword_score ≤ 0`;
- `hotword_case` lạ;
- thiếu `hotword_prompt` ở model chỉ nhận prompt;
- `decoding_method` khác `modified_beam_search` khi bật hotword cho sherpa.

| Key trong `config` | Ý nghĩa |
|---|---|
| `adapter` · `dtype` | runtime, và precision báo cáo ở cột `precision` |
| `note` | chú thích cho người đọc |
| `provider` | `cpu` · `cuda` · `remote` |
| `hotword_enabled` | `false` ⇒ không chạy `on` |
| `hotword_case` | `as_is` · `upper` · `lower`: khớp hoa thường của vocab |
| `hotword_score` · `hotword_score_param` | trọng số boost và tên thật của tham số runtime; `null` với model không có thang |
| `hotword_prompt` | template có `{words}` cho model nhận prompt |
| `convert_to_fp32` | riêng `qwen3-asr`: `true` = chạy fp32, `false` = giữ bf16; `dtype` phải khớp |

## Artifact

`python run.py --output <dir>` ghi vào `<dir>` (mặc định `report/`). Mỗi thư mục là **một lần chạy
tự chứa**:

| File | Nội dung |
|---|---|
| `summary.md` | 4 bảng: WER `off` · độ chính xác `on` toàn bộ · phụ lục `on` theo nhóm · resource. Kèm bảng model không đo được và ghi chú |
| `accuracy.csv` | 1 dòng / *(model, điều kiện, nhóm)* với nhóm ∈ {`all`, `vi`, `vi-en`, `vi-en-acro`}: số thô `n e a l tp fa` và các tỉ lệ. Ở `off` chỉ có `wer` |
| `resource.csv` | 1 dòng / model: trạng thái và mọi cột tài nguyên, kèm `vram_method` |
| `per_sample/<model>.jsonl` | từng mẫu × điều kiện: `text_raw`, output đã chuẩn hoá, `heard` (A), `label` (L), số thô, latency |
| `fingerprint.json` | máy (GPU, CPU, RAM, OS), Python, git sha, `--hotword`, warm-up, sha256 của manifest và hotwords |

## Dữ liệu dùng chung

Chi tiết nguồn và cách tạo lại: [data/README.md](../data/README.md).

### `data/manifest.jsonl` — 1 dòng / mẫu

| Trường | Nội dung |
|---|---|
| `id` | theo `data_class`, vd. `vi_003` · `vi-en_007` · `vi-en-acro_010` |
| `data_class` | `vi` · `vi-en` · `vi-en-acro` (30 mẫu mỗi nhóm) |
| `audio` | wav 16 kHz mono, đường dẫn tương đối với thư mục chứa manifest |
| `ref` | nhãn đã chuẩn hoá: chữ thường, bỏ dấu câu, giữ dấu thanh; acronym ghi dạng chuẩn tắc (`nmn`) |
| `spoken_form` | canonical các acronym trong `ref`, theo thứ tự xuất hiện, giữ lần lặp: `["NADH", "NADH"]` |
| `annotation_status` | `reviewed` / `pending` |

Sáu trường trên là **bắt buộc và đứng đầu** theo đúng thứ tự đó. Sau chúng là các trường **tuỳ
chọn**, harness không đọc: `keywords` (thuật ngữ code-switch của ViMedCSS, chưa đầy đủ, chưa dùng khi
chấm), `tags`, `source` (provenance), `audio_info`. Chi tiết: [data/README.md](../data/README.md).

**Acronym bị cắt khỏi audio thì không khai báo.** ViMedCSS cắt segment theo `end_time` nguyên giây
nên đuôi audio thiếu tới khoảng 1 s. Hai mẫu `vi-en-acro_016`, `vi-en-acro_023` vì thế không có lượt
acronym nào nghe được, không đóng góp vào `L`, nhưng vẫn thuộc nhóm `vi-en-acro`.

Cả 34 lượt acronym đã được nghe lại trên audio. Cách đọc nghe được lưu ở `data/acronym_review.csv`
**làm hồ sơ**; không code nào đọc. `validate_dataset.py` chặn mọi `spoken_form` không bằng đúng dãy
token keyword đọc trong `ref`.

### `data/hotwords.yaml`

`canonical` · `distractor`. **44 mục: 31 acronym thực sự được nói và 13 distractor**
(có trong vocab, không được nói ở mẫu nào; một distractor bị in ra là một lần bịa).

- Vocab chỉ chứa acronym. Mỗi canonical phải chuẩn hoá ra đúng một token và không được trùng từ thường
  trong nhãn (`validate_dataset.py` kiểm qua luật dãy acronym).
- `validate_dataset.py` kiểm thêm: canonical 2–6 ký tự A-Z/0-9, không trùng, cờ `distractor` khớp
  với `spoken_form` của dataset.

## Kiểm soát công bằng

| Hạng mục | Quy tắc |
|---|---|
| Audio | cùng file, 16 kHz mono. Hai adapter sherpa đệm thêm 500 ms im lặng cuối câu (`TAIL_PADDING_MS`) vì transducer cần audio sau khi hết tiếng để nhả token cuối; các runtime khác đọc nguyên file |
| Hotword | cùng danh sách canonical cho mọi model; chỉ được đổi hoa thường, khai bằng `hotword_case` trong registry |
| Chấm điểm | cùng `hotwords.yaml`, cùng một hàm, không hậu xử lý output |
| Decoder | `off` và `on` dùng cùng decoder, chỉ khác hotword |
| Đo đạc | mỗi model một process; cùng máy, cùng `num_threads`; warm-up không tính |
| Revision | model pin commit trong `scripts/download_models.py`; dataset pin revision trong `data/prepare_dataset.py` |

## Rủi ro

| Rủi ro | Xử lý |
|---|---|
| Adapter nhận hotword nhưng thực chất bỏ qua | registry chặn cấu hình sai đã biết; so `off`/`on` trên `per_sample/`; test `boostable` |
| Hotword sai hoa thường so với vocab ⇒ sherpa bỏ qua im lặng | `hotword_case` khai theo vocab của từng model |
| Keyword trùng từ thường ⇒ FA oan | `validate_dataset.py` chặn trên nhãn; soi `per_sample/` khi FA cao |
| Runtime tự rơi về CPU mà không báo | adapter tự kiểm (CTranslate2, torch); `cuda_env.py` đặt `LD_LIBRARY_PATH` trước khi nạp CUDA; model `cuda` có `peak_vram` < 50 MB được report ghi chú *likely fell back to CPU*. Với sherpa-onnx còn cần CUDA runtime ≥ 12.9 (xem README, mục *Cài đặt*) |
| API công ty không truy cập được | adapter ping trước, model bị bỏ qua với trạng thái `unavailable` |
| Số của T4 bị đọc thành số Jetson | `fingerprint.json` ghi GPU; chỉ so giữa các model |
| Recall với 33 keyword | một keyword = 3 điểm %; luôn đọc kèm Σ\|L\| và số tuyệt đối |
