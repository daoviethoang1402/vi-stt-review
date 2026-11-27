# Metric — định nghĩa và cách tính

Đi kèm [PIPELINE.md](PIPELINE.md). File này là
**ground truth** về cách cài đặt yêu cầu đó trong code (`scoring.py`).

| Nhóm | Metric | Đo khi nào |
|---|---|---|
| Độ chính xác | **WER** | `off` **và** `on` |
| Độ chính xác | **Keyword Recall** · **Keyword Precision** · **FA/1k** | **chỉ `on`** |
| Tài nguyên | `precision` · p50/p95 latency · throughput · `peak_rss` · `peak_vram` · `model_size` | lượt `on` |

Không gộp điểm. Một model tốt ở metric này và tệ ở metric kia là **thông tin**, không phải lỗi.
Không có FIR (bằng đúng `1 − Precision`), không có KER, không có ΔRecall.

## Quy ước

| Khái niệm | Nghĩa |
|---|---|
| **nhãn** (reference) | câu chữ đúng, trường `ref` trong manifest |
| **output** (hypothesis) | câu chữ model in ra cho cùng audio, trường `text_raw` |
| **chuẩn hoá** | `scoring.normalize()`: NFC → chữ thường → mọi ký tự `category P` thành khoảng trắng → gộp khoảng trắng. Áp cho **cả nhãn và output**, và chỉ có vậy |
| **token** | đơn vị sau `.split()` theo khoảng trắng, sau chuẩn hoá. Từ tiếng Anh, keyword và từng âm tiết tiếng Việt có giá trị như nhau |
| **keyword** | hiện chỉ là **từ viết tắt** (acronym) trong `data/hotwords.yaml`, gồm cả distractor |
| **canonical** | dạng chuẩn tắc của keyword, vd. `ADH`. Sau chuẩn hoá là **đúng một token** (`adh`) |

**Không hậu xử lý output.** Không ghép cụm tên chữ cái, không ánh xạ `nờ → n`, không sửa chính tả.
Model nói gì chấm nấy. **Giữ dấu thanh**, chỉ bỏ dấu câu: `hóa` ≠ `hoa`.

Nhãn ghi canonical, không ghi phiên âm: người nói đọc `nờ mờ nờ` hay `en em en` đều ghi `nmn`.
Cách đọc thật được lưu ở `data/acronym_review.csv` làm hồ sơ; không dùng khi chấm.

## Năm số thô của một mẫu

| Số | Nghĩa |
|---|---|
| `N` | số token của **nhãn** |
| `E` | `S + D + I`: khoảng cách Levenshtein mức token giữa nhãn và output |
| `A` | tập keyword **model nghe được**: canonical (của mọi entry trong `hotwords.yaml`) xuất hiện **nguyên token** trong output |
| `L` | tập keyword **có trong nhãn**: các canonical trong `spoken_form` của mẫu, giao với `hotwords.yaml` |
| `A∩L`, `A \ L` | nghe đúng · nghe nhầm (bịa, false alarm) |

`N` và `L` **luôn lấy từ nhãn**, nên `off` và `on` so được dù output dài ngắn khác nhau.

### Quy tắc so khớp keyword

- **Chỉ canonical.** Dạng đánh vần hay phiên âm (`a dê hát`, `A D H`, `ây di aitch`) **không bao giờ
  được tính** là nghe đúng.
- **Không phân biệt hoa thường.** Vì cả nhãn và output đều được chuẩn hoá về chữ thường, `ADH`,
  `adh`, `Adh` là một.
- **Nguyên token.** `adhd` hay `siadh` không chứa lượt `adh`.
- **Theo tập trên từng mẫu.** Một keyword xuất hiện hai lần trong cùng câu vẫn chỉ tính một phần tử
  của `A` (hay `L`). Lần lặp dư vẫn bị WER phạt.

| Nhãn | Output | `A` | `L` | `A∩L` | `A \ L` |
|---|---|---|---|---:|---:|
| `… nồng độ adh tăng` | `… nồng độ ADH tăng` | {adh} | {adh} | 1 | 0 |
| `… nồng độ adh tăng` | `… nồng độ a dê hát tăng` | ∅ | {adh} | 0 | 0 |
| `… nồng độ adh tăng` | `… nồng độ ADN tăng` | {adn} | {adh} | 0 | 1 |
| `… xét nghiệm hdl` | `… xét nghiệm HDL và MRI` | {hdl, mri} | {hdl} | 1 | 1 |
| `… châu âu là lục địa …` | `… châu âu là LH địa …` | {lh} | ∅ | 0 | 1 |

Nghe nhầm acronym này thành acronym khác (dòng 3) bị tính **hai lần**: sót ở Recall và bịa ở FA.
Đúng như thực tế: model vừa bỏ mất acronym thật, vừa đưa một acronym sai vào câu cho LLM.

### Từ trùng mặt chữ

Vì so khớp không phân biệt hoa thường, một canonical **không được trùng** với một từ thường xuất hiện
trong nhãn. `AI` đã bị bỏ khỏi `hotwords.yaml` vì "ai" là từ tiếng Việt có trong nhãn nhóm `vi`: giữ
lại thì mọi output đúng chữ "ai" thành một lần bịa. `data/validate_dataset.py` chặn trường hợp này.
`HA`, `VA` (va chạm) và `WHO` (tiếng Anh) hiện chưa trùng trong nhãn, nhưng output **có thể** in ra
chúng như từ thường. Khi đó chúng bị tính là FA; soi `per_sample/` trước khi kết luận.

## Bốn metric

Mọi metric là **micro**: cộng tử của mọi mẫu, cộng mẫu của mọi mẫu, rồi mới chia. Không báo macro.

| Metric | Công thức | Không xác định khi |
|---|---|---|
| **WER** | Σ(S + D + I) / ΣN | không bao giờ (ΣN > 0) |
| **Recall** | Σ\|A∩L\| / Σ\|L\| | Σ\|L\| = 0 (nhóm không có keyword) |
| **Precision** | Σ\|A∩L\| / Σ\|A\| | Σ\|A\| = 0 (model không in keyword nào) |
| **FA/1k** | Σ\|A \ L\| / ΣN × 1000 | không bao giờ |

Kèm theo là hai số đếm tuyệt đối: **TP** = Σ\|A∩L\| (keyword nghe đúng) và **FA** = Σ\|A \ L\|
(keyword bịa). Từ hai số này kiểm lại được cả hai tỷ lệ: Recall = TP / Σ\|L\|, Precision = TP / (TP + FA).
Σ\|L\| là hằng số của dataset nên chỉ ghi một lần dưới tiêu đề bảng 2, không thành cột; Σ\|A\| = TP + FA
nên cũng không cần cột riêng. Bảng 2 và bảng 3 nối với nhau bằng phép cộng: TP toàn bộ = TP của nhóm
`vi-en-acro` (hai nhóm kia không có nhãn keyword), FA toàn bộ = tổng FA của 3 nhóm. Vì vậy Precision
toàn bộ có thể thấp hơn Precision của nhóm `vi-en-acro`: FA ở nhóm `vi-en` kéo nó xuống.

- **WER micro** phản ánh đúng tỷ lệ lỗi trên tổng lượng từ; câu dài có trọng số lớn hơn câu ngắn.
- **Recall micro** cho mẫu số Σ\|L\| > 0 ở cấp toàn dataset (hiện 33 keyword, tất cả ở nhóm
  `vi-en-acro`), nên không bị undefined như khi tính từng mẫu. Một keyword đổi trạng thái = 3 điểm %.
- **FA/1k** chia cho số token **nhãn**, một hằng số của dataset, không phụ thuộc output. Nhờ vậy nó
  so được giữa các model, và không bao giờ undefined. Hệ số 1000 chỉ để đổi đơn vị cho dễ đọc
  ("số lần bịa trên mỗi 1 000 từ"). Chia cho độ dài output thì ngược lại: model chèn nhiều từ rác lại
  có FA thấp hơn.
- **Precision** trả lời câu khác: *khi model in ra một acronym, tin được bao nhiêu*. Mẫu số phụ thuộc
  output: hai model cùng bịa 5 lần nhưng in đúng 30 và 5 acronym có Precision 86 % và 50 %.

### Ô `—` và `n/a`

| Ký hiệu | Nghĩa | Ví dụ |
|---|---|---|
| `—` | **không xác định, cần xem lại** | Precision khi Σ\|A\| = 0 trên toàn dataset: model không in nổi một canonical nào |
| `n/a` | **không áp dụng** | Recall, Precision của nhóm `vi` / `vi-en`: nhãn của nhóm này không có keyword nào |

Σ\|A\| = 0 trên toàn dataset là trường hợp tệ nhất (Recall khi đó tự động bằng 0 %). Model **vẫn ở
trong bảng**, kèm một ghi chú. Đọc `text_raw` trong `per_sample/` để phân biệt hai nguyên nhân:

- output có dạng đánh vần (`a đê hát`) → **giới hạn của model**, đó là một kết quả khảo sát;
- output trống, lạ, hay sai hoa thường so với vocab → **lỗi adapter**, sửa rồi chạy lại.

### Theo nhóm dữ liệu

| Metric | `vi` | `vi-en` | `vi-en-acro` | toàn bộ |
|---|---|---|---|---|
| WER | ✅ | ✅ | ✅ | ✅ |
| Recall | `n/a` | `n/a` | ✅ | ✅ (= nhóm `vi-en-acro`) |
| Precision | `n/a` | `n/a` | ✅ | ✅ |
| FA/1k, FA | ✅ | ✅ | ✅ | ✅ |

Ở nhóm `vi` và `vi-en`, FA/1k là con số duy nhất cho biết model có **bịa acronym khi không có
acronym nào để nghe** hay không.

## Cách chấm một mẫu

```
hyp = normalize(text_raw)          ref = normalize(record.ref)
E   = edit_distance(ref.split(), hyp.split())
A   = { token ∈ hyp.split() | token ∈ canonical(hotwords.yaml) }
L   = set(spoken_form) ∩ canonical(hotwords.yaml)
```

Cùng một hàm cho **mọi model** và **cả hai điều kiện**. Danh sách keyword truyền vào hàm chấm là
canonical của `--hotwords` (mặc định `data/hotwords.yaml`), giống nhau ở mọi model.

| File | Vai trò khi chấm |
|---|---|
| `data/hotwords.yaml` → `canonical` | định nghĩa tập keyword dùng để dựng `A` và `L` |
| `manifest.jsonl` → `spoken_form` | nguồn của `L`: danh sách canonical theo thứ tự xuất hiện, giữ lần lặp |

**Invariant dữ liệu**, kiểm lúc validate manifest chứ không phải lúc chấm: `spoken_form` phải
bằng đúng dãy token keyword đọc trong `ref`. Nhờ vậy L luôn khớp với nhãn, và không keyword nào
trùng một từ thường.

Thứ tự và lần lặp trong `spoken_form` **không ảnh hưởng điểm** (A, L là tập). Chúng được giữ để
dữ liệu không mất thông tin, nếu sau này cần đếm theo số lần.

### Test chứng minh hàm chấm không hỏng

`tests/test_scoring.py` và `tests/test_harness.py`. Test nào fail ⇒ **hàm chấm hay harness sai**,
không phải model sai.

| Test | Kỳ vọng |
|---|---|
| chuẩn hoá: `HÔM NAY, em đi học!` vs `hôm nay em đi học` | `E = 0` |
| đánh vần: nhãn `mti`, output `em ti ai` | `E = 3`, `A = ∅` |
| phiên âm không được tính: output `mờ tê i`, `M T I` | `A = ∅` |
| nhầm acronym: nhãn `hdl`, output `mri` | sót 1, bịa 1 |
| adapter `oracle` (trả đúng `ref`) | WER 0, Recall 100 %, Precision 100 %, FA 0 |
| adapter `dummy` (trả rỗng) | WER 100 %, Σ\|A\| = 0 ⇒ Precision `—` |
| adapter `boostable` (chỉ in canonical khi được cấp hotword) | Recall `off` 0 %, `on` 100 % |

## Resource footprint

Latency và throughput tính trên **lượt `on`** (điều kiện triển khai thật); khi chạy `--hotword no`
thì tính trên lượt `off`, và cột *Measured on* ghi rõ. `peak_rss` và `peak_vram` là đỉnh của **cả
process**: mỗi model một process riêng, nên con số gồm lúc nạp model, warm-up và cả hai lượt.

| Cột | Cách đo |
|---|---|
| `precision` | dtype lúc chạy, khai ở `config.dtype` trong `registry.yaml` |
| `model_size` | PyTorch: `Σ numel × element_size` của tham số và buffer. ONNX Runtime, CTranslate2: dung lượng file trọng số |
| p50 / p95 latency | phân phối thời gian wall của **từng lần** `transcribe()` trên toàn bộ mẫu; nội suy tuyến tính |
| throughput | Σ thời lượng audio / Σ thời gian xử lý (RTF nghịch đảo, tính micro). `rtf` có trong `resource.csv` |
| `peak_rss` | `VmHWM` của process: toàn bộ RAM host đã dùng, gồm interpreter, framework, thư viện CUDA |
| `peak_vram` | đỉnh VRAM của process, poll `nvidia-smi` mỗi 0,1 s từ trước khi nạp model đến hết lượt cuối |

`peak_vram` có hai cách đo, ghi ở cột `vram_method` của `resource.csv`:

- `per_process`: `nvidia-smi --query-compute-apps` thấy PID của process. Đo đúng phần của process.
- `device_delta`: trong container (vd. Kaggle) driver thường không thấy PID. Khi đó lấy đỉnh
  `memory.used` của card trừ mốc đo **trước khi nạp model**. Đúng khi không có gì khác chạy trên GPU,
  điều mà cách ly process đảm bảo.

`peak_rss` và `peak_vram` là **hai pool bộ nhớ khác nhau**, không cộng lại. Model qua API
(`provider: remote`) không có `model_size`, `peak_rss`, `peak_vram`; latency của nó gồm cả mạng.

Latency đo bằng cách nạp **cả câu** một lần cho mọi model, kể cả model streaming. Đây là chi phí xử lý
một câu, không phải độ trễ cảm nhận khi stream.
