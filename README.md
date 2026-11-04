# MusicGenAI SIC

MusicGenAI SIC là hệ thống **text-to-music được huấn luyện từ đầu** trên FMA Small.
Mô hình sinh nhạc là Transformer Decoder do dự án tự xây dựng; T5-base chỉ mã hóa
văn bản và EnCodec 32 kHz chỉ chuyển đổi giữa waveform và acoustic tokens.

## Tính năng

- Sinh nhạc từ prompt tiếng Việt hoặc tiếng Anh.
- Transformer tự hồi quy với delay pattern cho 4 EnCodec codebook.
- Classifier-free guidance (CFG) giúp kết quả bám prompt tốt hơn.
- Chia mỗi track thành 6 đoạn để khai thác toàn bộ dữ liệu.
- Cân bằng genre bằng `WeightedRandomSampler`.
- Cache theo batch trên CUDA và có thể tiếp tục nếu bị gián đoạn.
- Tự lưu latest/best checkpoint và tiếp tục huấn luyện.
- Giao diện web dạng chat, nghe WAV trực tiếp và lưu lịch sử bằng SQLite.

## Kiến trúc

```text
Text prompt
    │
    ▼
T5-base encoder ──► text embeddings [64, 768]
                         │
Audio 32 kHz             ▼
    │              Transformer Decoder
    ▼              (384 × 6 layers, CFG)
EnCodec encoder          │
    │                    ▼
    └────────────► 4 acoustic codebooks ──► EnCodec decoder ──► WAV
```

| Thành phần | Cấu hình |
|---|---:|
| Sample rate | 32.000 Hz |
| Độ dài mẫu | 5 giây |
| Acoustic token rate | 50 token/giây |
| Codebook | 4 × 2.048 token |
| `d_model` | 384 |
| Decoder layers | 6 |
| Attention heads | 8 |
| Feed-forward dimension | 1.536 |
| Maximum sequence length | 512 |

## Cấu trúc dự án

```text
Music-GenAI/
├── config.py                 # Cấu hình data/model/train/generation
├── data.py                   # FMA loader, multi-crop và encoded cache
├── model.py                  # Transformer Decoder và CFG generation
├── train.py                  # Train, validation, test và checkpoint
├── test.py                   # Đánh giá test set hoặc sinh WAV bằng CLI
├── frontend/
│   ├── app.py                # FastAPI backend, SQLite và model inference
│   ├── index.html            # Giao diện chat
│   ├── app.js                # Logic frontend và API calls
│   └── style.css             # Responsive dark theme
├── dataset/                  # FMA và cache, không commit
├── checkpoints/              # Checkpoint, không commit
├── outputs/                  # WAV sinh ra, không commit
├── pyproject.toml
└── uv.lock
```

## Yêu cầu

- Windows 10/11 và Python 3.12.
- [uv](https://docs.astral.sh/uv/) để quản lý môi trường.
- GPU NVIDIA hỗ trợ CUDA. Pipeline đã được kiểm thử trên RTX 2050 4 GB.
- Internet cho lần đầu tải FMA, T5-base và EnCodec.
- Đủ dung lượng cho FMA Small và khoảng 48.000 file cache.

## Cài đặt

```powershell
git clone https://github.com/dinhmanhctnk33/MusicGenAI_SIC.git
cd MusicGenAI_SIC
uv sync
```

Kiểm tra CUDA:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dữ liệu và cache

FMA metadata và FMA Small được tải tự động khi chạy training. Sau khi làm sạch,
pipeline hiện giữ khoảng 7.996 track. Mỗi track tạo 6 crop cố định từ đầu tới cuối,
tương đương khoảng 47.976 mẫu.

Mỗi cache item gồm:

- `audio_tokens`: `[4, 250]`
- `text_embeddings`: `[64, 768]`
- `text_attention_mask`: `[64]`
- `prompt`, `genre`, `track_id`, `segment_index`

Cache v2 nằm tại:

```text
dataset/encoded_cache_v2_multicrop/sr32000_duration5000ms/
```

Cache được tạo theo batch 8 trên CUDA FP16. Có thể dừng bằng `Ctrl+C` và chạy lại;
những item đã tạo sẽ được tái sử dụng.

Ép thiết bị encoder khi cần:

```powershell
$env:MUSICGEN_ENCODER_DEVICE="cuda"  # hoặc "cpu"
uv run train.py
```

## Huấn luyện

```powershell
uv run train.py
```

Cấu hình mặc định:

- 30 epoch, early stopping sau 5 epoch không cải thiện.
- Batch size 1, gradient accumulation 16.
- AdamW, cosine learning-rate scheduler và gradient clipping.
- AMP FP16 trên CUDA.
- CFG dropout `0.1`.
- Split theo track: 90% train, 5% validation, 5% test.
- Mọi crop của cùng một track luôn nằm trong cùng split.
- Weighted sampling để cân bằng genre.

Checkpoint được lưu riêng cho scratch v2:

```text
checkpoints/scratch_v2/latest_musicgen_model.pt
checkpoints/scratch_v2/best_musicgen_model.pt
```

Nếu latest checkpoint tồn tại, `train.py` tự tiếp tục từ epoch đã lưu. Để cố ý
train lại từ đầu:

```powershell
$env:MUSICGEN_RESUME="none"
uv run train.py
```

Xóa biến sau khi dùng:

```powershell
Remove-Item Env:MUSICGEN_RESUME -ErrorAction SilentlyContinue
```

## Đánh giá và sinh nhạc bằng CLI

Đánh giá best checkpoint trên test split:

```powershell
uv run test.py
```

Sinh nhạc:

```powershell
uv run test.py --mode generate `
  --prompt "Energetic Rock music with electric guitar and strong drums" `
  --duration 5 `
  --temperature 0.8 `
  --top-k 50 `
  --guidance-scale 2.5 `
  --output rock.wav
```

Prompt ngắn, gần với genre trong FMA thường cho kết quả ổn định hơn:

```text
Rock music.
Hip-Hop music.
Electronic music.
Classical music.
```

## Giao diện web

Khởi động hệ thống:

```powershell
uv run frontend/app.py
```

Trình duyệt tự mở tại [http://127.0.0.1:7860](http://127.0.0.1:7860).

Frontend sử dụng trực tiếp source trong `frontend/index.html`, `frontend/style.css`
và `frontend/app.js`. FastAPI backend cung cấp:

| API | Chức năng |
|---|---|
| `GET /api/health` | Trạng thái CUDA và checkpoint |
| `POST /api/generate` | Sinh nhạc từ prompt |
| `GET /api/history` | Danh sách lịch sử |
| `GET /api/history/{id}` | Chi tiết một hội thoại |
| `DELETE /api/history/{id}` | Xóa hội thoại và WAV tương ứng |
| `DELETE /api/history` | Xóa toàn bộ lịch sử |

Lịch sử được lưu trong `musicgen_history.sqlite3`; WAV được lưu trong `outputs/`.
Backend ưu tiên checkpoint scratch v2, sau đó mới fallback về checkpoint cũ.

## Lưu ý

- Lần tạo cache v2 đầu tiên mất nhiều giờ; các lần sau đọc cache trực tiếp.
- Khi thay đổi kiến trúc hoặc cache signature, phải train checkpoint mới.
- Nếu gặp CUDA out-of-memory, đóng các ứng dụng đang dùng GPU rồi chạy lại.
- Nếu gặp `CUDA device-side assert`, đóng hoàn toàn tiến trình Python cũ.
- Dataset, cache, checkpoint, WAV và SQLite local không được đưa lên Git.
- T5-base hiểu prompt tiếng Anh tốt hơn tiếng Việt với cấu hình hiện tại.

## Giấy phép dữ liệu

FMA có giấy phép riêng cho metadata và từng track. Người sử dụng phải tuân thủ
giấy phép của [Free Music Archive](https://github.com/mdeff/fma) và giấy phép
tương ứng của từng bản nhạc.
