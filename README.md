# Khảo sát mô hình STT tiếng Việt

Chọn model speech-to-text cho **lễ tân ảo** chạy on-device:

| # | Yêu cầu |
|---|---|
| 1 | Chạy được trên **Jetson Orin NX 8 GB Unified Memory**, không cần TensorRT |
| 2 | Bắt được **từ tiếng Anh** lẫn trong câu tiếng Việt |
| 3 | Nạp được **keyword** (hiện là từ viết tắt) liên quan đến công ty/tổ chức/phòng ban |

Khảo sát **không** tìm "model tốt nhất". Nó trả lời *mỗi model đánh đổi những gì*, nên output là
bảng nhiều chiều, không phải một điểm tổng hợp. Harness cho phép đánh giá nhiều model trên cùng một
dataset mà không sửa code: mỗi model chỉ là một mục trong `registry.yaml`.

Máy chạy: **Kaggle T4**. `qwen3-asr-1.7b` chạy qua API trên mạng công ty và được tự động bỏ qua khi
không truy cập được.

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | **Yêu cầu gốc**: model, dataset, metric, quy trình, harness |
| [docs/METRIC.md](docs/METRIC.md) | **Ground truth** về cách chấm: WER, Recall, Precision, FA/1k, resource footprint |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Quy trình, danh sách model, interface adapter, `registry.yaml`, artifact, dữ liệu |
| [docs/CODEBASE.md](docs/CODEBASE.md) | Kiến trúc code và phụ thuộc giữa các module |
| [docs/COMPARE.md](docs/COMPARE.md) | So sánh các model về lý thuyết (không có số đo) |
| [data/README.md](data/README.md) | Bộ dữ liệu: nguồn, cột manifest, trạng thái kiểm định, cách tạo lại |

## Cài đặt

Python 3.11, CUDA 12.4. Bộ version đã pin nằm trong [requirements.txt](requirements.txt), kèm hai
`--extra-index-url` bắt buộc (torch `+cu124` và wheel CUDA của sherpa-onnx) và ghi chú vì sao
`chunkformer` / `pyctcdecode` phải cài `--no-deps`. `run.py` tự đặt `LD_LIBRARY_PATH` rồi exec lại
chính mình, không cần `export` tay.

```bash
pip install -r requirements.txt
```

```bash
pip install --no-deps chunkformer==1.2.2 pyctcdecode==0.5.0 pygtrie==2.6.2 hypothesis==6.170.15
```

```bash
python scripts/download_models.py            # tải mọi model vào models/ (~11,5 GB)
python scripts/download_models.py --list     # danh sách model và commit đã pin
```

## Chạy

```bash
python run.py                                  # mọi model trong registry → report/
python run.py --model zipformer-30m            # chỉ một (hoặc vài) model
python run.py --list                           # liệt kê model
python run.py --hotword no                     # chỉ chạy không hotword (WER)
python run.py --output report-t4 --warmup false
```

| Cờ | Ý nghĩa | Mặc định |
|---|---|---|
| `--model NAME ...` | chỉ chạy các model này; gọi `--model` mà không có tên là lỗi | chạy tất cả |
| `--list` | liệt kê model rồi thoát | |
| `--hotword yes\|no\|both` | `no`: chỉ `off` (WER) · `yes`: chỉ `on` · `both`: `off` rồi `on` | `both` |
| `--warmup true\|false` | chạy `run.warmup_samples` mẫu đầu một lần trước khi đo | `true` |
| `--manifest` | dataset đầu vào | `data/manifest.jsonl` |
| `--hotwords` | danh sách keyword (vừa nạp vào decoder, vừa để chấm) | `data/hotwords.yaml` |
| `--output` | thư mục report | `report/` |
| `--registry` · `--threads` | registry khác · ghi đè `run.num_threads` | |

**Không có cờ nào cho tham số của model.** Mọi tham số nằm trong `config` của từng model trong
`registry.yaml`, được kiểm chặt lúc nạp ([harness/registry.py](harness/registry.py)).

Kiểm tra:

```bash
python -m unittest discover -s tests -t .      # hàm chấm, harness, registry, CLI
python data/validate_dataset.py                # invariant của dataset và hotwords.yaml
```

## Kết quả

Mỗi thư mục `--output` là một lần chạy tự chứa:

- `summary.md`: 4 bảng.
  1. WER khi không gắn hotword (toàn bộ + 3 nhóm).
  2. WER · Recall · Precision · FA/1k khi gắn hotword, trên toàn bộ dataset.
  3. Phụ lục bảng 2 theo từng nhóm dữ liệu.
  4. Resource footprint.
- `accuracy.csv`, `resource.csv`: số thô và các tỉ lệ.
- `per_sample/`: output từng mẫu để audit.
- `fingerprint.json`: thông tin máy và lần chạy.

Định nghĩa và cách đọc từng cột: [METRIC.md](docs/METRIC.md).

## Cấu trúc

```
run.py            entry duy nhất
registry.yaml     khai báo model — nguồn duy nhất biết về model
harness/          điều phối: nạp → warm-up → off → on → resource → report
adapters/         1 file / runtime — nơi duy nhất biết về backend
scoring.py        hàm chấm
scripts/          tải model
data/             dataset, hotwords.yaml, tooling validate/prepare
tests/            test + adapter đối chứng (oracle, dummy, boostable)
docs/             OVERVIEW · METRIC · PIPELINE · CODEBASE · COMPARE
```
